"""ASAMA 1: ogretmen triplane'lerini AYRI olarak oturt.

Neden ayri: egitim dongusu icinde ogretmen, obje basina adim basina batch/N_dataset
(=8/2635=0.003) guncelleme aliyor; 15000 adimda sadece ~34 guncelleme. Olculdu
(diag_teacher_quality.py, step 4000): ogretmen 15.02 dB, ogrenci 16.38 dB --
yani ogretmen ogrenciyi ASAGI cekiyordu.

Burada transformer/encoder yok, adim basina 4 obje dogrudan guncelleniyor:
1024 obje icin 25600 adim = obje basina 100 guncelleme (oracle testinde 94
guncelleme 23.8 dB veriyordu). Sonuc: ogrenciden cok daha iyi bir hedef.

Cikti: dataset/lrm_ckpts/teacher_init.pt  {triplanes, nerf, uids, cfg}
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras
from lrm.dataset import LRMDataset
from lrm.nerf import TriplaneNeRF
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render
from lrm.losses import LRMLoss

DEV = "cuda"
OUT = "dataset/lrm_ckpts/teacher_init.pt"
PREVIEW = "dataset/lrm_val_previews/fit_teacher.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list.json")
    ap.add_argument("--renders_dir", default="dataset/renders")
    ap.add_argument("--n_obj", type=int, default=1024, help="ilk N train objesi")
    ap.add_argument("--steps", type=int, default=25600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--nerf_lr", type=float, default=1e-3)
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--bound", type=float, default=0.6)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # deterministic=False => her erisimde farkli supervision gorunumleri
    # (ogretmen coklu-gorunum denetimi alsin, tek aciya ezberlemesin)
    ds = LRMDataset(a.train_list, a.renders_dir, split="train", render_res=a.res,
                    n_sup=4, augment=False, deterministic=False)
    n = min(a.n_obj, len(ds))
    triplanes = nn.Parameter(torch.randn(n, 3, 32, 64, 64, device=DEV) * 0.1)
    nerf = TriplaneNeRF(in_dim=96, hidden=64).to(DEV)
    opt = torch.optim.Adam([{"params": [triplanes], "lr": a.lr},
                            {"params": nerf.parameters(), "lr": a.nerf_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    loss_fn = LRMLoss(use_lpips=True, w_lpips=a.w_lpips).to(DEV)
    per_obj = a.steps * a.batch / n
    print(f"ogretmen fit: {n} obje | {a.steps} adim x batch {a.batch} "
          f"=> obje basina ~{per_obj:.0f} guncelleme | res {a.res} "
          f"| bellek {triplanes.numel()*4/1e9:.2f} GB", flush=True)

    def render(tp, c2w, K, bg):
        os_, ds_ = [], []
        for v in range(c2w.shape[0]):
            o, d = cameras.rays_from_camera(c2w[v], K[v], a.res, a.res)
            os_.append(o); ds_.append(d)
        o, d = torch.cat(os_).to(DEV), torch.cat(ds_).to(DEV)

        def q(pts):
            den, rgb = nerf(sample_triplane(tp, pts, bound=a.bound))
            ins = (pts.abs().amax(-1, keepdim=True) <= a.bound).to(den.dtype)
            return den * ins, rgb
        rgb, acc = volume_render(o, d, 0.8, 2.2, a.n_samples, q, bg_color=bg)
        V = c2w.shape[0]
        return (rgb.reshape(V, a.res, a.res, 3).permute(0, 3, 1, 2),
                acc.reshape(V, a.res, a.res, 1).permute(0, 3, 1, 2))

    rng = np.random.default_rng(0); t0 = time.time()
    for step in range(a.steps):
        opt.zero_grad(); agg = 0.0
        for _ in range(a.batch):
            i = int(rng.integers(n))
            it = ds[i]
            alpha = it["sup_alpha"].to(DEV); prem = it["sup_premult"].to(DEV)
            c = torch.rand(3, device=DEV)
            target = prem + (1 - alpha) * c[None, :, None, None]
            rgb, acc = render(triplanes[i], it["sup_c2w"].to(DEV),
                              it["sup_K"].to(DEV), c)
            total, _ = loss_fn(rgb, acc, target, alpha)
            (total / a.batch).backward(); agg += float(total) / a.batch
        opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            with torch.no_grad():
                ps = []
                for i in range(0, min(n, 24), 4):
                    it = ds[i]
                    gt = it["sup_rgb"][:1].to(DEV)
                    r, ac = render(triplanes[i], it["sup_c2w"][:1].to(DEV),
                                   it["sup_K"][:1].to(DEV), None)
                    ps.append(float(((r + (1 - ac) - gt) ** 2).mean()))
                psnr = -10 * math.log10(max(float(np.mean(ps)), 1e-9))
            el = time.time() - t0
            eta = (a.steps - step - 1) / max((step + 1) / el, 1e-6) / 3600
            print(f"  step {step:6d}/{a.steps} loss={agg:.4f} PSNR={psnr:.2f}dB "
                  f"[{(step+1)/el:.2f} it/s, kalan ~{eta:.1f} sa]", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"triplanes": triplanes.detach().cpu(), "nerf": nerf.state_dict(),
                "uids": ds.uids[:n], "cfg": vars(a)}, a.out)
    print(f"kaydedildi: {a.out}", flush=True)
    with torch.no_grad():
        rows = []
        for i in range(6):
            it = ds[i]
            gt = it["sup_rgb"][0].to(DEV)
            r, ac = render(triplanes[i], it["sup_c2w"][:1].to(DEV),
                           it["sup_K"][:1].to(DEV), None)
            rows.append(torch.cat([gt, (r + (1 - ac))[0].clamp(0, 1)], -1).cpu())
    Image.fromarray((torch.cat(rows, 1).permute(1, 2, 0).numpy() * 255
                     ).astype(np.uint8)).save(PREVIEW)
    print(f"gorsel: {PREVIEW}", flush=True)


if __name__ == "__main__":
    main()
