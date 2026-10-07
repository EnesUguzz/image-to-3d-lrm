"""Hiz tezgahi: obje basina egitim adimi kac ms?

NEDEN VAR:
Hiz bu projenin en pahali kisiti (13k obje x 16k adim = 7 saat, ~10 epoch;
referans OpenLRM 60 epoch yapiyor). Her hiz iddiasi TEK KOMUTLA dogrulanabilmeli,
yoksa "sanirim hizlandi" ile ilerlenir.

OLCULEN TABAN (2026-08-26, RTX 5080, 2 girdi gorunumu):
    fp32, 4 gorunum x 128^2 ....... 192.3 ms/obje   <- bugunku egitim
    bf16, 4 gorunum x 128^2 ....... 107.5 ms/obje   1.79x
    bf16, 4 gorunum x  64^2 .......  66.9 ms/obje   2.87x
    bf16, 3 gorunum x  64^2 .......  63.5 ms/obje   3.03x  <- OpenLRM tarifesi
    veri yukleme (senkron, CPU) ...  28.5 ms/obje   (worker'la gizlenir)
FAYDASI OLCULUP ELENENLER: torch.compile (%2), obje-batch'leme (fayda yok,
B=1 zaten compute-bound), isin-AABB kirpma (orneklerin %69.8'i zaten kutu icinde).

Kullanim:
    python scripts/bench_speed.py --tag taban
    python scripts/bench_speed.py --tag hedef --amp --render_res 64 --n_sup 3
    python scripts/bench_speed.py --tag veri --data      # gercek dataloader dahil
"""
import argparse
import contextlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm.model import LRM
from lrm.losses import LRMLoss
from lrm import runstamp
from lrm import defaults

DEV = "cuda"
OUT = "dataset/lrm_bench"


def _timeit(fn, n=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


def _fake_batch(n_input, n_sup, render_res, input_res=None):
    # 2026-08-29: sabit 224 idi -> defaults.INPUT_RES degistiginde olcum SESSIZCE
    # eski cozunurlugu olcuyordu (336 ile 224 birebir ayni sure cikti).
    input_res = defaults.INPUT_RES if input_res is None else input_res
    """Sentetik girdi: hiz olcumu icin veri icerigi onemsiz, SEKIL onemli."""
    ii = torch.rand(n_input, 3, input_res, input_res, device=DEV)
    ic = torch.eye(4, device=DEV)[None].repeat(n_input, 1, 1)
    ic[:, 2, 3] = 1.4866
    f_in = input_res * 1.09375                      # lens 35 / sensor 32
    ik = torch.tensor([[f_in, 0, input_res / 2], [0, f_in, input_res / 2],
                       [0, 0, 1.0]], device=DEV)[None].repeat(n_input, 1, 1)
    sc = ic[:1].repeat(n_sup, 1, 1)
    f_r = render_res * 1.09375
    sk = torch.tensor([[f_r, 0, render_res / 2], [0, f_r, render_res / 2],
                       [0, 0, 1.0]], device=DEV)[None].repeat(n_sup, 1, 1)
    tgt = torch.rand(n_sup, 3, render_res, render_res, device=DEV)
    al = torch.rand(n_sup, 1, render_res, render_res, device=DEV)
    return ii, ic, ik, sc, sk, tgt, al


def data_ms(train_list, renders_dir, render_res, n_sup, n=30):
    """Gercek dataloader'in ornek basina CPU maliyeti (senkron olcum)."""
    from lrm.dataset import LRMDataset
    ds = LRMDataset(train_list, renders_dir, split="train", render_res=render_res,
                    n_sup=n_sup, augment=True, deterministic=False)
    _ = ds[0]
    t0 = time.time()
    for i in range(n):
        _ = ds[(i * 37) % len(ds)]
    return (time.time() - t0) / n * 1000.0, len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="speed")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (loss fp32'de)")
    ap.add_argument("--render_res", type=int, default=128)
    ap.add_argument("--n_sup", type=int, default=4)
    ap.add_argument("--n_input", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=defaults.N_SAMPLES)
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--data", action="store_true",
                    help="gercek dataloader maliyetini de olc")
    ap.add_argument("--train_list", default="dataset/train_list_opp_score3.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = LRM(n_samples=a.n_samples).to(DEV)
    model.train()
    loss_fn = LRMLoss(use_lpips=True, w_lpips=a.w_lpips).to(DEV)
    ii, ic, ik, sc, sk, tgt, al = _fake_batch(a.n_input, a.n_sup, a.render_res)
    ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.amp
           else (lambda: contextlib.nullcontext()))

    def full_step():
        with ctx():
            rgb, acc = model(ii, ic, ik, sc, sk, (a.render_res, a.render_res),
                             bg_color=torch.zeros(3, device=DEV))
        # loss DAIMA fp32'de: bf16'da genis toplamlarin hassasiyeti dusuyor
        total, _ = loss_fn(rgb.float(), acc.float(), tgt, al)
        total.backward()

    def fwd_only():
        with torch.no_grad(), ctx():
            model(ii, ic, ik, sc, sk, (a.render_res, a.render_res),
                  bg_color=torch.zeros(3, device=DEV))

    def triplane_only():
        with ctx():
            model.make_triplane(ii, ic, ik)

    def encoder_only():
        with ctx():
            model.encoder(ii)

    torch.cuda.reset_peak_memory_stats()
    res = {
        "encoder_fwd": _timeit(encoder_only, a.iters),
        "make_triplane_fwd": _timeit(triplane_only, a.iters),
        "full_fwd_nograd": _timeit(fwd_only, a.iters),
        "full_step": _timeit(full_step, a.iters),
    }
    res["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["obj_per_s_gpu"] = 1000.0 / res["full_step"]

    if a.data:
        d_ms, n_ds = data_ms(a.train_list, a.renders_dir, a.render_res, a.n_sup)
        res["data_ms_senkron"] = d_ms
        res["obj_per_s_senkron"] = 1000.0 / (res["full_step"] + d_ms)
        res["dataset_n"] = n_ds

    cfg = vars(a)
    os.makedirs(OUT, exist_ok=True)
    payload = {"cfg": cfg, "stamp": runstamp.run_stamp(cfg), "ms": res}
    with open(os.path.join(OUT, f"speed_{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    print(f"\n[{a.tag}] amp={int(a.amp)} girdi={a.n_input} sup={a.n_sup} "
          f"res={a.render_res} ornek={a.n_samples}")
    print(f"  encoder ileri        : {res['encoder_fwd']:8.1f} ms")
    print(f"  triplane ileri       : {res['make_triplane_fwd']:8.1f} ms")
    print(f"  tam ileri (nograd)   : {res['full_fwd_nograd']:8.1f} ms")
    print(f"  TAM ADIM (ileri+geri): {res['full_step']:8.1f} ms/obje"
          f"   => {res['obj_per_s_gpu']:.1f} obje/s (sadece GPU)")
    if a.data:
        print(f"  veri yukleme (senkron): {res['data_ms_senkron']:7.1f} ms/obje"
              f"   => {res['obj_per_s_senkron']:.1f} obje/s (worker YOK)")
    print(f"  VRAM tepe            : {res['vram_gb']:8.2f} GB")
    print(f"  -> {OUT}/speed_{a.tag}.json")


if __name__ == "__main__":
    main()
