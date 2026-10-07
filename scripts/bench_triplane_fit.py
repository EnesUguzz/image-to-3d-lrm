"""AYIRT EDICI TEST: encoder+transformer YOK. Her objeye KENDI serbest triplane'i
verilir (oracle conditioning). Ayni renderer + NeRF MLP + loss.

  - PSNR yuksek (>25dB) => temsil/render/loss saglikli, sorun goruntu->triplane yolu.
  - PSNR ~17dB'de takilir => sorun temsilde (triplane cozunurlugu / NeRF MLP kapasitesi /
    ornekleme / loss), conditioning'i duzeltmek TEK BASINA ise yaramaz.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras
from lrm import defaults
from lrm.nerf import TriplaneNeRF
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render
from lrm.losses import LRMLoss
from bench_overfit import load_view, SUP_VIEWS

DEV = "cuda"; OUT = "dataset/lrm_bench"


class DeepNeRF(nn.Module):
    """Daha derin NeRF MLP (OpenLRM/TripoSR 10 katman kullanir; bizde 2)."""
    def __init__(self, in_dim, hidden=64, layers=10):
        super().__init__()
        L = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(layers - 1):
            L += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        self.backbone = nn.Sequential(*L)
        self.density_head = nn.Linear(hidden, 1)
        self.rgb_head = nn.Linear(hidden, 3)

    def forward(self, f):
        h = self.backbone(f)
        return torch.nn.functional.softplus(self.density_head(h)), torch.sigmoid(self.rgb_head(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obj", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--tp_res", type=int, default=64)
    ap.add_argument("--tp_ch", type=int, default=32)
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=defaults.N_SAMPLES)
    # 2026-09-02 kod incelemesi B3: varsayilan 2.0 idi. CLAUDE.md'de
    # OLCULEREK terk edilmis deger: LPIPS baskin olunca model "ortalama
    # obje" havzasinda kaliyor (32 obje: top-1 %25 -> %62, 0.25'te).
    # train_lrm ve fit_teacher 0.25 kullaniyor; KAPI ARACI egitimden
    # farkli bir tarifeyi olcuyordu. tests/test_kunye_uyumu.py hizayi tutar.
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--bound", type=float, default=0.6)
    ap.add_argument("--tag", default="oracle")
    ap.add_argument("--save_tp", action="store_true",
                    help="ogrenilen triplane'leri + NeRF'i kaydet (distilasyon testi icin)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    uids = json.load(open("dataset/train_list.json", encoding="utf-8"))["train"]
    uids = uids[::max(1, len(uids) // a.n_obj)][:a.n_obj]
    data = []
    for u in uids:
        prem, alpha, c2w, K = [], [], [], []
        for v in SUP_VIEWS:
            arr, c, k = load_view(u, v, a.res)
            prem.append(arr[:3] * arr[3:4]); alpha.append(arr[3:4]); c2w.append(c); K.append(k)
        data.append(dict(prem=torch.stack(prem).to(DEV), alpha=torch.stack(alpha).to(DEV),
                         c2w=torch.stack(c2w).to(DEV), K=torch.stack(K).to(DEV)))

    triplanes = nn.Parameter(torch.randn(len(uids), 3, a.tp_ch, a.tp_res, a.tp_res, device=DEV) * 0.1)
    nerf = (TriplaneNeRF(in_dim=3 * a.tp_ch, hidden=64) if a.mlp_layers == 2
            else DeepNeRF(3 * a.tp_ch, 64, a.mlp_layers)).to(DEV)
    loss_fn = LRMLoss(use_lpips=True, w_lpips=a.w_lpips).to(DEV)
    opt = torch.optim.Adam([{"params": [triplanes], "lr": a.lr},
                            {"params": nerf.parameters(), "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    print(f"[{a.tag}] ORACLE triplane | {len(uids)} obje | tp {a.tp_res}^2x{a.tp_ch} "
          f"| mlp_layers={a.mlp_layers} | n_samples={a.n_samples} | res {a.res} "
          f"| w_lpips={a.w_lpips} | bound={a.bound}", flush=True)

    def render(i, c2w, K, res, bg):
        tp = triplanes[i]
        os_, ds_ = [], []
        for v in range(c2w.shape[0]):
            o, d = cameras.rays_from_camera(c2w[v], K[v], res, res)
            os_.append(o); ds_.append(d)
        o, d = torch.cat(os_), torch.cat(ds_)
        def q(pts):
            den, rgb = nerf(sample_triplane(tp, pts, bound=a.bound))
            ins = (pts.abs().amax(-1, keepdim=True) <= a.bound).to(den.dtype)
            return den * ins, rgb
        rgb, acc = volume_render(o, d, defaults.NEAR, defaults.FAR, a.n_samples, q, bg_color=bg)
        V = c2w.shape[0]
        return (rgb.reshape(V, res, res, 3).permute(0, 3, 1, 2),
                acc.reshape(V, res, res, 1).permute(0, 3, 1, 2))

    rng = np.random.default_rng(0); t0 = time.time(); hist = []
    for step in range(a.steps):
        opt.zero_grad()
        for _ in range(4):
            i = int(rng.integers(len(data))); d = data[i]
            c = torch.rand(3, device=DEV)
            target = d["prem"] + (1 - d["alpha"]) * c[None, :, None, None]
            rgb, acc = render(i, d["c2w"], d["K"], a.res, c)
            total, _ = loss_fn(rgb, acc, target, d["alpha"])
            (total / 4).backward()
        opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            with torch.no_grad():
                ps = []
                for i, d in enumerate(data):
                    rgb, acc = render(i, d["c2w"][:1], d["K"][:1], a.res, None)
                    pred = rgb + (1 - acc)          # beyaz zemine kompozit
                    gt = d["prem"][:1] + (1 - d["alpha"][:1])
                    ps.append((pred - gt).pow(2).mean().item())
                mse = float(np.mean(ps)); psnr = -10 * math.log10(max(mse, 1e-9))
            hist.append(dict(step=step, psnr=psnr, mse=mse))
            print(f"  step {step:5d} PSNR={psnr:.2f}dB mse={mse:.4f} "
                  f"[{(step+1)/(time.time()-t0):.2f} it/s]", flush=True)
    with torch.no_grad():
        imgs = []
        for i, d in enumerate(data[:8]):
            rgb, acc = render(i, d["c2w"][:1], d["K"][:1], a.res, None)
            rgb = rgb + (1 - acc)
            imgs.append(torch.cat([d["prem"][0] + (1 - d["alpha"][0]), rgb[0]], -1))
        Image.fromarray((torch.cat(imgs, 1).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                         * 255).astype(np.uint8)).save(f"{OUT}/{a.tag}.png")
    json.dump(dict(cfg=vars(a), hist=hist), open(f"{OUT}/{a.tag}.json", "w", encoding="utf-8"), indent=1)
    if a.save_tp:
        torch.save({"triplanes": triplanes.detach().cpu(), "nerf": nerf.state_dict(),
                    "uids": uids, "cfg": vars(a)}, f"{OUT}/{a.tag}_tp.pt")
        print(f"  triplane+nerf kaydedildi: {OUT}/{a.tag}_tp.pt", flush=True)
    print(f"[{a.tag}] BITTI PSNR={hist[-1]['psnr']:.2f}dB", flush=True)


if __name__ == "__main__":
    main()
