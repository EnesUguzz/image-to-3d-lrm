"""Ortak egitim: hizli ogrenen 'ogretmen' triplane + LRM 'ogrenci'.

Teshis (docs/lrm-blob-diagnosis.md): render hedefi DOGRU (E1: iyi cozumun kaybi
0.23, blob'un 0.47) ama model rastgele baslangictan 'ortalama obje' yerel
minimumuna dusuyor. Serbest triplane ise ayni render kaybiyla 25 dB'ye cikiyor
(O1) cunku araya transformer girmiyor.

Fikir: ikisini AYNI kosuda calistir.
  - ogretmen: obje basina serbest triplane, render kaybi -> hizla dogru cozume gider
  - ogrenci: LRM, ogretmenin triplane'ine distilasyon + kendi render kaybi
  - ogretmenin agirligi zamanla soner; ogrenci render kaybiyla devam eder (E1 bunun
    calistigini gosterdi: distile agirliklardan render kaybi 24.10 -> 24.99 dB)
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras
from lrm import defaults
from lrm.model import LRM
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render
from lrm.losses import LRMLoss
from bench_overfit import build, evaluate, sample_fg_crop, OUT, DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obj", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--teacher_lr", type=float, default=1e-2)
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--fg_weight", type=float, default=5.0)
    ap.add_argument("--w_tv", type=float, default=5e-4)
    ap.add_argument("--w_distill", type=float, default=1.0)
    ap.add_argument("--teacher_subset", type=int, default=0,
                    help="ogretmen sadece ilk N objede (0=hepsi). Uretimde bellek "
                         "yuzunden kismi kapsama kullanacagiz; bu bayrak onu test eder.")
    ap.add_argument("--teacher_off", type=float, default=0.75,
                    help="egitimin bu oranindan sonra ogretmen/distilasyon kapanir")
    ap.add_argument("--n_samples", type=int, default=defaults.N_SAMPLES)
    ap.add_argument("--fg_crop", type=int, default=0,
                    help="TripoSR tarzi on-plana yanli rastgele kirpma: hedefleri bu "
                         "cozunurlukte sakla (ör. 256) ve her adim rastgele kirp. "
                         "0=kapali")
    ap.add_argument("--crop_min", type=float, default=0.45)
    ap.add_argument("--tag", default="TCH")
    ap.add_argument("--eval_every", type=int, default=250)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True

    all_uids = json.load(open("dataset/train_list.json", encoding="utf-8"))["train"]
    uids = all_uids[::max(1, len(all_uids) // a.n_obj)][:a.n_obj]
    data = build(uids, a.res, hi_res=a.fg_crop)

    model = LRM(n_samples=a.n_samples).to(DEV)
    tp_shape = (3, 32, 64, 64)
    n_tch = a.teacher_subset if a.teacher_subset > 0 else len(uids)
    n_tch = min(n_tch, len(uids))
    teacher = nn.Parameter(torch.randn(n_tch, *tp_shape, device=DEV) * 0.1)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    opt_t = torch.optim.Adam([teacher], lr=a.teacher_lr)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    loss_fn = LRMLoss(use_lpips=True, w_lpips=a.w_lpips, fg_weight=a.fg_weight).to(DEV)
    off_at = int(a.teacher_off * a.steps)
    print(f"[{a.tag}] ogretmen+ogrenci | {len(uids)} obje | w_lpips={a.w_lpips} "
          f"fg={a.fg_weight} w_distill={a.w_distill} fg_crop={a.fg_crop}@{a.crop_min} "
          f"ogretmen_kapsama={n_tch}/{len(uids)} "
          f"ogretmen {off_at}. adimda kapanir",
          flush=True)

    def render_tp(tp, c2w, K, res, bg):
        os_, ds_ = [], []
        for v in range(c2w.shape[0]):
            o, d = cameras.rays_from_camera(c2w[v], K[v], res, res)
            os_.append(o); ds_.append(d)
        o, d = torch.cat(os_).to(DEV), torch.cat(ds_).to(DEV)

        def q(pts):
            den, rgb = model.nerf(sample_triplane(tp, pts, bound=model.bound))
            ins = (pts.abs().amax(-1, keepdim=True) <= model.bound).to(den.dtype)
            return den * ins, rgb
        rgb, acc = volume_render(o, d, model.near, model.far, a.n_samples, q, bg_color=bg)
        V = c2w.shape[0]
        return (rgb.reshape(V, res, res, 3).permute(0, 3, 1, 2),
                acc.reshape(V, res, res, 1).permute(0, 3, 1, 2))

    rng = np.random.default_rng(0); t0 = time.time(); hist = []
    model.train()
    for step in range(a.steps):
        use_teacher = step < off_at
        opt.zero_grad(); opt_t.zero_grad()
        agg = 0.0
        for _ in range(a.batch):
            i = int(rng.integers(len(data))); d = data[i]
            c = torch.rand(3, device=DEV)
            if a.fg_crop:
                prem_c, alpha_c, K_c = sample_fg_crop(d, a.res, rng, smin=a.crop_min)
            else:
                prem_c, alpha_c, K_c = d["prem"], d["alpha"], d["sk"]
            target = prem_c + (1 - alpha_c) * c[None, :, None, None]
            total = 0.0
            if use_teacher and i < n_tch:         # ogretmen: serbest triplane
                rgb_t, acc_t = render_tp(teacher[i], d["sc"], K_c, a.res, c)
                lt, _ = loss_fn(rgb_t, acc_t, target, alpha_c)
                total = total + lt
            rgb, acc = model(d["ii"], d["ic"], d["ik"], d["sc"], K_c,
                             (a.res, a.res), bg_color=c)
            ls, parts = loss_fn(rgb, acc, target, alpha_c)
            total = total + ls + a.w_tv * model.tv_loss(model._last_triplane)
            if use_teacher and i < n_tch:         # distilasyon: ogrenci -> ogretmen
                tt = teacher[i].detach()
                # ogretmenin varyansina normalize: w_distill olcekten bagimsiz kalir
                dist = ((model._last_triplane - tt) ** 2).mean() / tt.var().clamp_min(1e-6)
                total = total + a.w_distill * dist
            (total / a.batch).backward()
            agg += float(ls) / a.batch
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if use_teacher:
            opt_t.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            psnr, top1, mse, mmse, P, G = evaluate(model, data, a.res)
            npsnr, ntop1, nmse, nmmse, _, _ = evaluate(model, data, a.res, novel=True)
            hist.append(dict(step=step, psnr=psnr, top1=top1, mse=mse, mean_mse=mmse,
                             novel_psnr=npsnr, novel_top1=ntop1, teacher=use_teacher))
            print(f"  step {step:5d} ogrenci_loss={agg:.4f} gorulen={psnr:.2f}dB/"
                  f"top1={top1:.0%} YENI-GORUNUM={npsnr:.2f}dB/top1={ntop1:.0%} "
                  f"mse={mse:.4f} ortalama-baseline={mmse:.4f} "
                  f"{'[ogretmen ACIK]' if use_teacher else '[ogretmen kapali]'} "
                  f"[{(step+1)/(time.time()-t0):.2f} it/s]", flush=True)
    psnr, top1, mse, mmse, P, G = evaluate(model, data, a.res)
    npsnr, ntop1, nmse, nmmse, NP, NG = evaluate(model, data, a.res, novel=True)
    n = min(8, len(data))
    # satir: [gorulen GT | gorulen tahmin | YENI GT | YENI tahmin]
    grid = torch.cat([torch.cat([G[i], P[i], NG[i], NP[i]], -1) for i in range(n)], 1)
    Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                     ).astype(np.uint8)).save(f"{OUT}/{a.tag}.png")
    json.dump(dict(cfg=vars(a), hist=hist,
                   final=dict(psnr=psnr, top1=top1, mse=mse, mean_mse=mmse,
                              novel_psnr=npsnr, novel_top1=ntop1)),
              open(f"{OUT}/{a.tag}.json", "w", encoding="utf-8"), indent=1)
    print(f"[{a.tag}] BITTI gorulen={psnr:.2f}dB/top1={top1:.0%} "
          f"YENI-GORUNUM={npsnr:.2f}dB/top1={ntop1:.0%} "
          f"mse={mse:.4f} vs ortalama-baseline={mmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
