"""Denetim cozunurlugu x triplane cozunurlugu MALIYET matrisi.

Soru: "res 512 yapsak ne olur, neden yapmiyoruz?"
Bu betik cevabi OLCER (tahmin etmez): her konfigurasyon icin gercek egitim
adiminin suresini, VRAM'ini ve temsil-uyumunu basar.

Uc sutun kritik:
  ms/adim   : gercek maliyet (warmup disi, medyan)
  VRAM      : tepe ayrilan bellek
  hucre_px  : BIR triplane hucresinin denetim goruntusundeki piksel boyu.
              >2 ise denetim, temsilin tasiyabildiginden ince detay istiyor
              => bosa maliyet + izgara artefakti (last_V3'te olculdu: kor 0.007)
"""
import argparse, json, time, statistics, sys, os
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm import cameras, defaults
from lrm.dataset import LRMDataset
from lrm.nerf import TriplaneNeRF
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render_chunked
from lrm.losses import LRMLoss

DEV = "cuda"


def nyquist_n(tp_res):
    """Isin adimi hucreden kucuk olmali: (FAR-NEAR)/N < 2*BOUND/tp_res."""
    import math
    return int(math.ceil((defaults.FAR - defaults.NEAR) * tp_res / (2 * defaults.BOUND)))


def bir_kol(ds, res, tp_res, tp_ch, n_samples, n_sup, ray_chunk, isinma, olcum):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ds.render_res = res
    n = 2
    tp = nn.Parameter(torch.randn(n, 3, tp_ch, tp_res, tp_res, device=DEV) * 0.1)
    nerf = TriplaneNeRF(in_dim=3 * tp_ch, hidden=64, layers=4).to(DEV)
    opt = torch.optim.Adam([tp] + list(nerf.parameters()), lr=1e-2)
    loss_fn = LRMLoss(use_lpips=True, w_lpips=0.25).to(DEV)

    items = []
    for i in range(n):
        it = ds[i]
        items.append({k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in it.items()})

    def adim(idx):
        it = items[idx]
        c2w, K = it["sup_c2w"][:n_sup], it["sup_K"][:n_sup]
        gt, gm = it["sup_rgb"][:n_sup], it["sup_alpha"][:n_sup]
        os_, ds_ = [], []
        for v in range(c2w.shape[0]):
            o, d = cameras.rays_from_camera(c2w[v], K[v], res, res)
            os_.append(o); ds_.append(d)
        o, d = torch.cat(os_).to(DEV), torch.cat(ds_).to(DEV)
        bg = torch.ones(3, device=DEV)

        def q(pts):
            den, rgb = nerf(sample_triplane(tp[idx], pts, bound=defaults.BOUND), pts)
            ins = (pts.abs().amax(-1, keepdim=True) <= defaults.BOUND).to(den.dtype)
            return den * ins, rgb
        rgb, acc = volume_render_chunked(o, d, defaults.NEAR, defaults.FAR,
                                         n_samples, q, bg_color=bg, chunk=ray_chunk)
        V = c2w.shape[0]
        pr = rgb.reshape(V, res, res, 3).permute(0, 3, 1, 2)
        pa = acc.reshape(V, res, res, 1).permute(0, 3, 1, 2)
        out = loss_fn(pr, pa, gt, gm)
        l = out[0] if isinstance(out, tuple) else out
        opt.zero_grad(set_to_none=True); l.backward(); opt.step()

    for i in range(isinma):
        adim(i % n)
    torch.cuda.synchronize()
    sur = []
    for i in range(olcum):
        t0 = time.perf_counter(); adim(i % n); torch.cuda.synchronize()
        sur.append((time.perf_counter() - t0) * 1000)
    vram = torch.cuda.max_memory_allocated() / 1e9

    # hucre_px: bir triplane hucresinin denetim goruntusundeki piksel boyu
    fx = float(items[0]["sup_K"][0][0, 0])
    z = float(items[0]["sup_c2w"][0][:3, 3].norm())
    hucre_px = (2 * defaults.BOUND / tp_res) * fx / z

    del tp, nerf, opt, loss_fn, items
    torch.cuda.empty_cache()
    return statistics.median(sur), vram, hucre_px


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--n_sup", type=int, default=1)
    ap.add_argument("--ray_chunk", type=int, default=32768)
    ap.add_argument("--isinma", type=int, default=3)
    ap.add_argument("--olcum", type=int, default=8)
    ap.add_argument("--out", default="dataset/lrm_logs/res_maliyet.json")
    a = ap.parse_args()

    kollar = [
        ("URETIM  res64  tp64",   64,  64,  32,  96),
        ("        res128 tp64",  128,  64,  32,  96),
        ("        res256 tp64",  256,  64,  32,  96),
        ("ONERI   res256 tp128", 256, 128,  64, 192),
        ("        res512 tp64",  512,  64,  32,  96),
        ("        res512 tp128", 512, 128,  64, 192),
        ("TAM     res512 tp256", 512, 256,  64, 320),
    ]
    ds = LRMDataset(a.train_list, a.renders_dir, split="train", render_res=64,
                    n_sup=4, augment=False, deterministic=True)
    print(f"{'kol':22s} {'ms/adim':>9s} {'VRAM GB':>8s} {'hucre_px':>9s} "
          f"{'x uretim':>9s}  not", flush=True)
    taban = None
    sonuc = []
    for ad, res, tpr, tpc, ns in kollar:
        try:
            ms, vram, hpx = bir_kol(ds, res, tpr, tpc, ns, a.n_sup, a.ray_chunk,
                                    a.isinma, a.olcum)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{ad:22s} {'OOM':>9s}", flush=True)
            sonuc.append({"kol": ad, "res": res, "tp_res": tpr, "oom": True})
            continue
        if taban is None:
            taban = ms
        not_ = "denetim temsilden ince (bosa)" if hpx > 2.0 else "uyumlu"
        print(f"{ad:22s} {ms:9.0f} {vram:8.2f} {hpx:9.2f} {ms/taban:8.1f}x  {not_}",
              flush=True)
        sonuc.append({"kol": ad, "res": res, "tp_res": tpr, "tp_ch": tpc,
                      "n_samples": ns, "ms": ms, "vram_gb": vram,
                      "hucre_px": hpx, "x_uretim": ms / taban})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"nyquist": {tp: nyquist_n(tp) for tp in (64, 128, 256)},
                   "kollar": sonuc}, f, indent=2, ensure_ascii=False)
    print(f"\nNyquist gereken n_samples: tp64>={nyquist_n(64)} "
          f"tp128>={nyquist_n(128)} tp256>={nyquist_n(256)}")
    print(f"yazildi: {a.out}")
