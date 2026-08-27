"""ASAMA 2: LRM'i ogretmen triplane'lerine SAF distilasyonla egit (render YOK).

Neden render yok: distilasyon triplane uzayinda calisir, hacim render'ina ihtiyaci
yoktur. Render ornek basina ~786 bin sorgu demek ve maruziyet oranini o belirliyor.
Olculdu (train_lrm icinde birlikte kosarken): obje basina 2000 adimda sadece 6
maruziyet, 11250'de 34. Dogrulanmis T2 deneyinde 150'ydi ve orada yakinsamisti.
Render'i cikarinca ayni surede kat kat fazla maruziyet.

Zincir: fit_teacher.py (asama 1) -> distill_lrm.py (asama 2) -> train_lrm.py (asama 3).
E1 deneyi asama 3'un calistigini gosterdi: distile agirliklardan render kaybiyla
devam edilince 24.10 -> 24.99 dB.
"""
import argparse, contextlib, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from lrm.dataset import LRMDataset
from lrm.model import LRM

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="dataset/lrm_ckpts/teacher_init.pt")
    ap.add_argument("--train_list", default="dataset/train_list.json")
    ap.add_argument("--renders_dir", default="dataset/renders")
    ap.add_argument("--n_obj", type=int, default=0,
                    help="olcek egrisi: bankanin ilk N objesi (0=hepsi)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--out", default="dataset/lrm_ckpts/distilled.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast; burada hacim render yok (saf triplane "
                         "regresyonu) => K2 cokus riski yok")
    a = ap.parse_args()
    # DURAKLAT kancasi: zincirin 2. asamasi otomatik baslamasin diye (ornegin
    # makineyi yeniden baslatmak icin). Dosyayi silince normale doner.
    if os.path.isfile("dataset/DURAKLAT"):
        print("dataset/DURAKLAT var -> distilasyon BASLATILMADI. "
              "Devam etmek icin bu dosyayi sil.", flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)

    tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
    target = tk["triplanes"].to(DEV)
    # OLCEK EGRISI icin bankanin ILK n_obj objesi (bench uid'leri ic ice oldugu
    # gibi burada da alt kume ust kumenin ONEKI olsun -- tek degisken: obje sayisi).
    if a.n_obj and a.n_obj < target.shape[0]:
        target = target[:a.n_obj].contiguous()
        tk["uids"] = tk["uids"][:a.n_obj]
    n = target.shape[0]
    tvar = target.var().item()
    # HEDEFIN GURULTU PAYI (2026-08-26'da bulundu):
    # TV'siz oturtulan ogretmen triplane'lerinin varyansinin %40.5'i yuksek
    # frekansti ve render kalitesine sadece 0.34 dB katki veriyordu -- yani
    # ogrenilemez VE degersiz. Bu, rel = ((tp-hedef)^2)/hedef.var() metrigine
    # ~0.40'lik bir TABAN koyuyor ve "kapasite siniri" kapisini GECERSIZ kiliyor.
    # Artik taban olculup raporlaniyor; ayrica YUMUSATILMIS hedefe gore
    # rel_smooth hesaplaniyor (ogrenilebilir bileseni olcer).
    import torch.nn.functional as _F
    _C = target.shape[2]
    _x = target.reshape(-1, _C, target.shape[3], target.shape[4])
    _k = torch.ones(_C, 1, 3, 3, device=DEV) / 9.0
    target_sm = _F.conv2d(_F.pad(_x, (1, 1, 1, 1), mode="replicate"), _k,
                          groups=_C).reshape_as(target)
    noise_frac = ((target - target_sm).var() / target.var()).item()
    svar = target_sm.var().item()
    print(f"hedef gurultu payi: {noise_frac:.1%}  => rel icin ogrenilemez taban ~{noise_frac:.2f}",
          flush=True)
    if noise_frac > 0.20:
        print("  !! UYARI: taban esiklere yakin, kapi guvenilmez "
              "(ogretmeni --w_tv ile yeniden oturt)", flush=True)
    ds = LRMDataset(a.train_list, a.renders_dir, split="train", render_res=64,
                    n_sup=1, augment=False, deterministic=False)
    assert ds.uids[:n] == tk["uids"], "uid sirasi ogretmen dosyasiyla uyusmuyor"

    model = LRM(n_samples=48).to(DEV)
    model.nerf.load_state_dict(tk["nerf"])      # asama 3 icin tutarli kalsin
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / a.warmup if s < a.warmup else
        0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))
    start = 0
    if a.resume and os.path.isfile(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start = ck["step"]
        print(f"resume: step {start}", flush=True)

    exposure = a.steps * a.batch / n
    print(f"distilasyon: {n} obje | {a.steps} adim x batch {a.batch} "
          f"=> obje basina ~{exposure:.0f} maruziyet (T2'de 150 yeterliydi) "
          f"| hedef var={tvar:.4f}", flush=True)

    rng = np.random.default_rng(0); t0 = time.time()
    model.train()
    for step in range(start, a.steps):
        opt.zero_grad(); agg = 0.0
        for _ in range(a.batch):
            i = int(rng.integers(n))
            it = ds[i]
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if a.amp
                   else contextlib.nullcontext())
            with ctx:
                tp = model.make_triplane(it["input_imgs"].to(DEV), it["input_c2w"].to(DEV),
                                         it["input_K"].to(DEV))
            loss = ((tp.float() - target[i]) ** 2).mean() / tvar   # kayip fp32'de
            (loss / a.batch).backward(); agg += float(loss.detach()) / a.batch
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                model.eval()
                rel, rel_s, std = [], [], []
                for i in range(0, min(n, 64), 8):
                    it = ds[i]
                    _c = (torch.autocast("cuda", dtype=torch.bfloat16) if a.amp
                          else contextlib.nullcontext())
                    with _c:
                        tp = model.make_triplane(it["input_imgs"].to(DEV),
                                                 it["input_c2w"].to(DEV), it["input_K"].to(DEV))
                    tp = tp.float()
                    rel.append(float(((tp - target[i]) ** 2).mean() / tvar))
                    rel_s.append(float(((tp - target_sm[i]) ** 2).mean() / svar))
                    std.append(float(tp.std()))
                model.train()
            el = time.time() - t0
            done = step - start + 1
            eta = (a.steps - step - 1) / max(done / el, 1e-6) / 3600
            print(f"  step {step:6d}/{a.steps} egitim_rel={agg:.4f} val_rel={np.mean(rel):.4f} "
                  f"rel_smooth={np.mean(rel_s):.4f} "
                  f"ogrenci_std={np.mean(std):.4f} (hedef {target.std():.4f}) "
                  f"[{done/el:.2f} it/s, kalan ~{eta:.1f} sa]", flush=True)
        if step > start and step % 1000 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step}, a.out)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": a.steps}, a.out)
    print(f"kaydedildi: {a.out}", flush=True)


if __name__ == "__main__":
    main()
