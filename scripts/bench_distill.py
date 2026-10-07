"""DISTILASYON TESTI: oracle'in ogrendigi (bilinen-iyi) triplane'leri hedef alip
encoder+transformer'a DOGRUDAN regresyonla ogret. Render gradyani devrede degil.

  - Basarili (render PSNR oracle'a yaklasir) => goruntu->triplane esleme OGRENILEBILIR;
    sorun render uzerinden gelen gradyanin zayif/gurultulu olmasi (loss/optimizasyon).
  - Basarisiz => encoder+transformer bu eslemeyi kuramiyor (kapasite/donuk encoder/mimari).
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras
from lrm import defaults
from lrm.model import LRM
from lrm.nerf import TriplaneNeRF
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render
from bench_overfit import load_view, SUP_VIEWS, IN_VIEWS
from bench_triplane_fit import DeepNeRF

DEV = "cuda"; OUT = "dataset/lrm_bench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", required=True, help="oracle triplane .pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--arch", default="cross", choices=["joint", "cross"])
    ap.add_argument("--unfreeze_last", type=int, default=0)
    ap.add_argument("--tag", default="T_distill")
    ap.add_argument("--save_model", action="store_true",
                    help="distile edilmis encoder+transformer+head agirliklarini kaydet")
    a = ap.parse_args()
    torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(a.tp, map_location="cpu", weights_only=False)
    target = ck["triplanes"].to(DEV)          # (N,3,C,H,W)
    uids, ocfg = ck["uids"], ck["cfg"]
    nerf = (TriplaneNeRF(in_dim=3 * ocfg["tp_ch"], hidden=64) if ocfg["mlp_layers"] == 2
            else DeepNeRF(3 * ocfg["tp_ch"], 64, ocfg["mlp_layers"])).to(DEV)
    nerf.load_state_dict(ck["nerf"]); nerf.eval()
    for p in nerf.parameters():
        p.requires_grad_(False)

    data = []
    for u in uids:
        ii, ic, ik = [], [], []
        for v in IN_VIEWS:
            arr, c, k = load_view(u, v, 224)
            ii.append(arr[:3] * arr[3:4] + (1 - arr[3:4])); ic.append(c); ik.append(k)
        prem, alpha, sc, sk = [], [], [], []
        for v in SUP_VIEWS[:1]:
            arr, c, k = load_view(u, v, a.res)
            prem.append(arr[:3] * arr[3:4]); alpha.append(arr[3:4]); sc.append(c); sk.append(k)
        data.append(dict(ii=torch.stack(ii).to(DEV), ic=torch.stack(ic).to(DEV),
                         ik=torch.stack(ik).to(DEV), prem=torch.stack(prem).to(DEV),
                         alpha=torch.stack(alpha).to(DEV), sc=torch.stack(sc).to(DEV),
                         sk=torch.stack(sk).to(DEV)))

    model = LRM(n_samples=defaults.N_SAMPLES, cross_attn=(a.arch == "cross")).to(DEV)
    if a.unfreeze_last > 0:
        for blk in model.encoder.model.blocks[-a.unfreeze_last:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        model.encoder.forward = model.encoder.forward.__wrapped__.__get__(model.encoder)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / 200 if s < 200 else
        0.5 * (1 + math.cos(math.pi * (s - 200) / max(1, a.steps - 200))))
    tvar = target.var().item()
    print(f"[{a.tag}] distilasyon | {len(uids)} obje | arch={a.arch} "
          f"| hedef triplane {tuple(target.shape[1:])} var={tvar:.3f} "
          f"| unfreeze_last={a.unfreeze_last}", flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval(); rel, ps = [], []
        for i, d in enumerate(data):
            tp = model.make_triplane(d["ii"], d["ic"], d["ik"])
            rel.append(((tp - target[i]) ** 2).mean().item() / tvar)
            o, dd = cameras.rays_from_camera(d["sc"][0], d["sk"][0], a.res, a.res)
            def q(pts):
                den, rgb = nerf(sample_triplane(tp, pts, bound=ocfg["bound"]))
                ins = (pts.abs().amax(-1, keepdim=True) <= ocfg["bound"]).to(den.dtype)
                return den * ins, rgb
            rgb, acc = volume_render(o.to(DEV), dd.to(DEV), defaults.NEAR, defaults.FAR,
                                     defaults.N_SAMPLES, q)
            pred = (rgb + (1 - acc)).reshape(a.res, a.res, 3).permute(2, 0, 1)
            gt = d["prem"][0] + (1 - d["alpha"][0])
            ps.append((pred - gt).pow(2).mean().item())
        model.train()
        m = float(np.mean(ps))
        return float(np.mean(rel)), -10 * math.log10(max(m, 1e-9))

    rng = np.random.default_rng(0); t0 = time.time(); hist = []
    model.train()
    for step in range(a.steps):
        opt.zero_grad(); agg = 0.
        for _ in range(a.batch):
            i = int(rng.integers(len(data)))
            tp = model.make_triplane(data[i]["ii"], data[i]["ic"], data[i]["ik"])
            loss = ((tp - target[i]) ** 2).mean() / tvar
            (loss / a.batch).backward(); agg += float(loss) / a.batch
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            rel, psnr = evaluate()
            hist.append(dict(step=step, train_rel=agg, eval_rel=rel, psnr=psnr))
            print(f"  step {step:5d} triplane_rel_mse={agg:.4f} eval_rel={rel:.4f} "
                  f"renderPSNR={psnr:.2f}dB [{(step+1)/(time.time()-t0):.2f} it/s]", flush=True)
    json.dump(dict(cfg=vars(a), hist=hist), open(f"{OUT}/{a.tag}.json", "w", encoding="utf-8"), indent=1)
    if a.save_model:
        torch.save({"model": model.state_dict(), "arch": a.arch, "uids": uids},
                   f"{OUT}/{a.tag}_model.pt")
        print(f"  model kaydedildi: {OUT}/{a.tag}_model.pt", flush=True)
    print(f"[{a.tag}] BITTI eval_rel={hist[-1]['eval_rel']:.4f} "
          f"renderPSNR={hist[-1]['psnr']:.2f}dB (oracle tavani ~{ocfg.get('_',0) or 24.7}dB)", flush=True)


if __name__ == "__main__":
    main()
