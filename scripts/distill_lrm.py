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
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from lrm import defaults, imutil
from lrm.dataset import LRMDataset
from lrm.model import LRM

DEV = "cuda"
EVAL_EPOCH = 10_000_003   # egitimde asla kullanilmayan sabit epoch (olcum tutarli kalsin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="dataset/lrm_ckpts/teacher_init.pt")
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--head", default="k2s2", choices=["k2s2", "k4s2"],
                    help="TriplaneHead tabani. k2s2 (varsayilan) ORTUSMESIZ: "
                         "her token bagimsiz 2x2 blok uretir. k4s2 ortusmeli "
                         "(k=4,s=2,p=1). CPU tezgahinda olculdu: gercek "
                         "calisma noktasinda blok_sinir 8.67 -> 1.01 ve rel "
                         "0.351 -> 0.315 (yani bedel YOK).")
    ap.add_argument("--seed", type=int, default=0,
                    help="TEK KOLLU A/B YASAK: kontrolun kendi ic yayilimini "
                         "bilmeden fark okunamaz. Kontrolu iki tohumla kos.")
    ap.add_argument("--holdout", type=int, default=32,
                    help="Bankanin SON N objesi distilasyon egitiminden "
                         "CIKARILIR ve yalnizca olcumde kullanilir. "
                         "NEDEN (2026-09-02 bagimsiz denetim): `val_rel` "
                         "adini tasiyan sayi EGITIM objelerinde "
                         "hesaplaniyordu (8 obje, sadece farkli kamera) "
                         "=> in-sample. Deney 2nin tum okumasi buna "
                         "dayaniyordu ve genelleme hakkinda hicbir sey "
                         "soylemiyordu. 0 = kapali (eski davranis).")
    ap.add_argument("--n_obj", type=int, default=0,
                    help="olcek egrisi: bankanin ilk N objesi (0=hepsi)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--out", default="dataset/lrm_ckpts/distilled.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--oracle", type=int, default=0,
                    help="ORACLE KOSULLANDIRMA. 0 = normal (DINOv2). >0 ise "
                         "encoder ciktisi yerine obje basina SERBEST ogrenilebilir "
                         "N token konur (N tam kare olmali: 256, 1024...). "
                         "Bu, 'bu token butcesindeki HERHANGI bir encoder'in "
                         "verebilecegi en iyi sinyal' = KOSULLANDIRMANIN TAVANI. "
                         "rel duserse darbogaz encoder, dusmezse transformer+head.")
    ap.add_argument("--tok_lr", type=float, default=1e-2,
                    help="serbest token'larin lr'i. Model agirliklarindan AYRI ve "
                         "yuksek olmali: her token obje basina yalnizca "
                         "steps*batch/n kez guncelleniyor (fit_teacher'da "
                         "triplane'ler icin ayni gerekce, lr 1e-2).")
    ap.add_argument("--force_n_input", type=int, default=1,
                    help="girdi gorunum sayisini sabitler. Oracle kollarinda 1 "
                         "SART (serbest token (1,N,384) sekilli); taban kol da "
                         "ayni olmali yoksa kollar kiyaslanamaz.")
    ap.add_argument("--unfreeze_last", type=int, default=0,
                    help="DINOv2 son N blogunu ac (0 = tamamen donuk = "
                         "bugune kadarki davranis). Distilasyon bunu HIC "
                         "yapmiyordu; train_lrm.py:527 yapiyor. Referans "
                         "uygulamalarin ucu de encoder'i egitir.")
    ap.add_argument("--enc_lr_scale", type=float, default=0.1,
                    help="encoder param gruplarinin lr carpani "
                         "(train_lrm ile ayni varsayilan).")
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
    torch.manual_seed(a.seed)
    # ORACLE + HELD-OUT BIRLIKTE ANLAMSIZ (ve IndexError verir):
    # serbest token'lar OBJE BASINA ogrenilen parametreler; egitimden cikarilan
    # bir objenin token'i hic guncellenmez, yani olculen sey "egitilmemis
    # rastgele token"in hatasi olur. Ayrica `serbest_tok` egitim objesi kadar
    # satira sahip => held-out indisi tasar.
    if a.oracle and a.holdout:
        print(f"  !! --oracle {a.oracle} verildi => --holdout {a.holdout} "
              f"YOK SAYILIYOR (serbest token obje basina ogreniliyor; "
              f"held-out objenin token'i egitilmemis olurdu).", flush=True)
        a.holdout = 0

    tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
    target = tk["triplanes"].to(DEV)
    # OLCEK EGRISI icin bankanin ILK n_obj objesi (bench uid'leri ic ice oldugu
    # gibi burada da alt kume ust kumenin ONEKI olsun -- tek degisken: obje sayisi).
    if a.n_obj and a.n_obj < target.shape[0]:
        target = target[:a.n_obj].contiguous()
        tk["uids"] = tk["uids"][:a.n_obj]
    # HELD-OUT AYRIMI: bankanin son `holdout` objesi egitimden cikarilir.
    # Ogretmen triplane'i YALNIZ bankadaki objeler icin var, o yuzden
    # genelleme ancak bankanin bir dilimini saklayarak olculebilir.
    n_tum = target.shape[0]
    n_ho = min(max(a.holdout, 0), max(n_tum - 8, 0))
    n = n_tum - n_ho
    ho_idx = list(range(n, n_tum))
    tvar = target.var().item()
    if n_ho:
        print(f"HELD-OUT: {n_ho} obje egitimden cikarildi (egitim {n}, "
              f"olcum {n_ho}) -- `heldout_rel` GENELLEME sayisidir; "
              f"`insample_rel` ile arasindaki fark ACIKtir.", flush=True)
    else:
        print("  !! holdout=0: `val_rel` IN-SAMPLE (egitim objeleri, farkli "
              "kamera). Genelleme hakkinda BILGI TASIMAZ.", flush=True)
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
                    n_sup=1, augment=False, deterministic=False,
                    force_n_input=a.force_n_input)
    # DIKKAT: `n` artik EGITIM obje sayisi (held-out haric). Hizalama
    # kontrolu TUM banka uzerinden yapilmali, yoksa holdout>0 iken bu
    # assert dogru veride bile patlar (duman testinde yasandi).
    assert ds.uids[:n_tum] == tk["uids"], "uid sirasi ogretmen dosyasiyla uyusmuyor"

    model = LRM(n_samples=defaults.N_SAMPLES, head_tip=a.head).to(DEV)
    model.nerf.load_state_dict(tk["nerf"])      # asama 3 icin tutarli kalsin

    # ENCODER ACMA (2026-09-04). Distilasyonda encoder HER ZAMAN tamamen
    # donuktu: `encoder.py:15` requires_grad_(False) + `:27` @torch.no_grad.
    # `train_lrm.py:527` gerektiginde sarmalayiciyi aciyor, ama distilasyon --
    # goruntu->triplane haritasinin FIILEN ogrenildigi asama -- hic acmiyordu.
    #
    # GEREKCE OLCULDU (encoder duvari A/B, 2026-09-03, uc kol):
    #   girdi 224 -> 448  held-out'ta HICBIR SEY yapmadi (d = -0.0011,
    #   kontrolun tohum gurultusu 0.0050) AMA in-sample'da yardim etti
    #   (-0.0163 = 2.8x gurultu). Yani daha cok piksel giriyor ve DONUK
    #   ozellikler onu tasimiyor. Referans uygulamalarin ucu de
    #   (LRM / OpenLRM / TripoSR) encoder'i EGITIR.
    _enc_ids = set()
    if a.unfreeze_last > 0:
        for blk in model.encoder.model.blocks[-a.unfreeze_last:]:
            for p_ in blk.parameters():
                p_.requires_grad_(True)
        for p_ in model.encoder.model.norm.parameters():
            p_.requires_grad_(True)
        # forward @torch.no_grad ile sarili; gradyan aksin diye sarmalayiciyi ac
        model.encoder.forward = model.encoder.forward.__wrapped__.__get__(
            model.encoder)
        _enc_ids = {id(p_) for p_ in model.encoder.parameters()
                    if p_.requires_grad}
        _m = sum(p_.numel() for p_ in model.encoder.parameters()
                 if p_.requires_grad) / 1e6
        print(f"ENCODER: son {a.unfreeze_last} blok ACIK | {_m:.1f}M parametre "
              f"@ lr x {a.enc_lr_scale} = {a.lr * a.enc_lr_scale:.2e}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]

    # ORACLE: encoder ciktisi yerine obje basina serbest token'lar.
    serbest_tok = None
    if a.oracle:
        yan = int(round(a.oracle ** 0.5))
        assert yan * yan == a.oracle, (
            f"--oracle {a.oracle} tam kare degil; Plucker haritasi "
            f"sqrt(P)xsqrt(P) izgara kuruyor (model.py make_triplane).")
        _d = model.encoder.embed_dim
        serbest_tok = nn.Parameter(
            torch.randn(n, a.oracle, _d, device=DEV) * 0.02)
        print(f"ORACLE: encoder DEVRE DISI. Obje basina {a.oracle} serbest token "
              f"x {_d} boyut = {a.oracle * _d} sayi/obje "
              f"(hedef triplane {target[0].numel()} sayi/obje, "
              f"oran {a.oracle * _d / target[0].numel():.2f}x) | "
              f"bellek {serbest_tok.numel() * 4 / 1e9:.2f} GB", flush=True)
    # wd: norm/bias haric (bkz. train_lrm.py ayni duzeltme, 2026-08-29)
    _d = [q for q in params if q.ndim > 1 and id(q) not in _enc_ids]
    _n = [q for q in params if q.ndim <= 1 and id(q) not in _enc_ids]
    # ENCODER AYRI LR: onceden egitilmis agirliklar, model lr'iyle (4e-4)
    # bozulur. train_lrm ayni gerekceyle `--enc_lr_scale 0.1` kullaniyor.
    # LambdaLR grubun KENDI initial_lr'ini olcekler, o yuzden burada mutlak
    # deger vermek dogru.
    _ed = [q for q in params if q.ndim > 1 and id(q) in _enc_ids]
    _en = [q for q in params if q.ndim <= 1 and id(q) in _enc_ids]
    _elr = a.lr * a.enc_lr_scale
    _gruplar = [g for g in ({'params': _d, 'weight_decay': 0.05},
                            {'params': _n, 'weight_decay': 0.0},
                            {'params': _ed, 'weight_decay': 0.05, 'lr': _elr},
                            {'params': _en, 'weight_decay': 0.0, 'lr': _elr})
                if g['params']]
    if serbest_tok is not None:
        # AYRI LR: obje basina guncelleme sayisi cok dusuk (steps*batch/n).
        # Model lr'iyle (4e-4) 94 guncellemede hicbir yere gitmez ve kol
        # "kapasite yetmedi" gibi okunur -- oysa optimizasyon yetmemistir.
        _gruplar.append({'params': [serbest_tok], 'weight_decay': 0.0,
                         'lr': a.tok_lr})
    opt = torch.optim.AdamW(_gruplar, lr=a.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / a.warmup if s < a.warmup else
        0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))
    start = 0
    if a.resume and os.path.isfile(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start = ck["step"]
        print(f"resume: step {start}", flush=True)

    # RENDER KUNYESI (2026-09-02 kod incelemesi B2): bu asama kunye YAZMIYORDU
    # (dogrulandi: distilled_v3.pt yalnizca {model,opt,sched,step} tasiyor).
    # Sonuc: `train_lrm --init_from` uyusmazlik kapisi kunye bulamayinca
    # degerleri VARSAYIYOR (train_lrm.py:461) => kapi sessizce geciyor.
    # Distilasyon render yapmaz, ama urettigi triplane BELLI bir uzayda oturur
    # (bound) ve 3. asama onu BELLI bir decoder'la (density_bias) render eder.
    _render_cfg = {"density_bias": float(model.nerf.density_bias),
                   "noise_std": float(model.nerf.noise_std),
                   "bound": float(model.bound),
                   "n_samples": int(model.n_samples),
                   "input_res": defaults.INPUT_RES,
                   "nerf_layers": len([m for m in model.nerf.backbone
                                       if isinstance(m, torch.nn.Linear)]),
                   "teacher": os.path.basename(a.teacher),
                   "oracle": a.oracle, "force_n_input": a.force_n_input,
                   "olcek_yontemi": imutil._yontem(),
                   "head_tip": a.head, "seed": a.seed,
                   "unfreeze_last": a.unfreeze_last,
                   "enc_lr_scale": a.enc_lr_scale,
                   "holdout": a.holdout}

    exposure = a.steps * a.batch / n
    print(f"distilasyon{' [ORACLE %d token]' % a.oracle if a.oracle else ''}: "
          f"{n} obje | {a.steps} adim x batch {a.batch} "
          f"=> obje basina ~{exposure:.0f} maruziyet (T2'de 150 yeterliydi) "
          f"| hedef var={tvar:.4f}", flush=True)

    rng = np.random.default_rng(0); t0 = time.time()
    model.train()
    for step in range(start, a.steps):
        # 2026-08-29 DUZELTME: set_epoch() cagrilmiyordu -> LRMDataset RNG'si
        # donuktu (_epoch daima 0) ve her obje 16 goruumun SABIT 4'unden
        # ogreniliyordu; 12 render oluydu. Ayni hata train_lrm.py'de
        # duzeltilmis ama bu iki asamaya tasinmamisti.
        ds.set_epoch(step)
        opt.zero_grad(); agg = 0.0
        for _ in range(a.batch):
            i = int(rng.integers(n))
            it = ds[i]
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if a.amp
                   else contextlib.nullcontext())
            _tok = None if serbest_tok is None else serbest_tok[i][None]
            with ctx:
                tp = model.make_triplane(it["input_imgs"].to(DEV), it["input_c2w"].to(DEV),
                                         it["input_K"].to(DEV), tok=_tok)
            loss = ((tp.float() - target[i]) ** 2).mean() / tvar   # kayip fp32'de
            (loss / a.batch).backward(); agg += float(loss.detach()) / a.batch
        torch.nn.utils.clip_grad_norm_(
            params if serbest_tok is None else params + [serbest_tok], 1.0)
        opt.step(); sched.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            ds.set_epoch(EVAL_EPOCH)      # egri adimlar arasi karsilastirilabilir
            with torch.no_grad():
                model.eval()
                def _olc(idx_listesi):
                    r, rs, sd = [], [], []
                    for i_ in idx_listesi:
                        it = ds[i_]
                        _c = (torch.autocast("cuda", dtype=torch.bfloat16)
                              if a.amp else contextlib.nullcontext())
                        # ORACLE kolunda serbest token OBJE BASINA ogreniliyor;
                        # held-out objenin token'i EGITILMEMISTIR. O yuzden
                        # oracle + holdout birlikte anlamsiz -> asagida engellendi.
                        _tk = None if serbest_tok is None else serbest_tok[i_][None]
                        with _c:
                            tp = model.make_triplane(it["input_imgs"].to(DEV),
                                                     it["input_c2w"].to(DEV),
                                                     it["input_K"].to(DEV), tok=_tk)
                        tp = tp.float()
                        r.append(float(((tp - target[i_]) ** 2).mean() / tvar))
                        rs.append(float(((tp - target_sm[i_]) ** 2).mean() / svar))
                        sd.append(float(tp.std()))
                    return r, rs, sd
                rel, rel_s, std = _olc(range(0, min(n, 64), 8))
                ho_rel = _olc(ho_idx)[0] if ho_idx else []
                model.train()
            ds.set_epoch(step)            # egitim epoch'una don
            el = time.time() - t0
            done = step - start + 1
            eta = (a.steps - step - 1) / max(done / el, 1e-6) / 3600
            # `val_rel` ADI KORUNDU (eski loglarla kiyaslanabilsin) ama artik
            # ne oldugu ACIKCA yaziyor: in-sample. Karar `heldout_rel`den verilir.
            _ho = (f"heldout_rel={np.mean(ho_rel):.4f} "
                   f"ACIK={np.mean(ho_rel)-np.mean(rel):+.4f} " if ho_rel
                   else "heldout_rel=YOK(holdout=0) ")
            print(f"  step {step:6d}/{a.steps} egitim_rel={agg:.4f} "
                  f"insample_rel={np.mean(rel):.4f} {_ho}"
                  f"rel_smooth={np.mean(rel_s):.4f} "
                  f"ogrenci_std={np.mean(std):.4f} (hedef {target.std():.4f}) "
                  f"[{done/el:.2f} it/s, kalan ~{eta:.1f} sa]", flush=True)
        if step > start and step % 1000 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "render_cfg": _render_cfg}, a.out)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": a.steps, "render_cfg": _render_cfg}, a.out)
    print(f"kaydedildi: {a.out}", flush=True)


if __name__ == "__main__":
    main()
