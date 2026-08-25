"""Tam LRM egitim dongusu: mikro-batch + gradient accumulation, bf16,
checkpoint/resume, val preview, zaman damgali log."""
import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import torch.nn as nn

from lrm import cameras
from lrm.dataset import LRMDataset, _load_rgba
from lrm.model import LRM
from lrm.losses import LRMLoss
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render

CKPT_DIR = "dataset/lrm_ckpts"
PREVIEW_DIR = "dataset/lrm_val_previews"
LOG_DIR = "dataset/lrm_logs"


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("train_lrm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{ts}.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    for h in (fh, ch):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def save_checkpoint(path, model, opt, scheduler, step, teacher=None, opt_t=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ck = {"model": model.state_dict(), "opt": opt.state_dict(),
          "sched": scheduler.state_dict(), "step": step}
    if teacher is not None:   # resume'da ogretmen sifirdan baslamasin
        ck["teacher"] = teacher.detach().cpu()
        ck["opt_t"] = opt_t.state_dict()
    torch.save(ck, path)


def load_checkpoint(path, model, opt, scheduler):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scheduler.load_state_dict(ck["sched"])
    return ck["step"], ck.get("teacher"), ck.get("opt_t")


def build_val_probe(renders_dir, uids, render_res, input_res=224, device="cuda"):
    """SABIT val problemi: her obje icin girdi = kanonik[0] (on, az 0 el 20),
    hedef = kanonik[1] ve [2] (yan + arka). Kanonik acilar TUM objelerde ayni
    oldugu icin metrik objeler arasi karsilastirilabilir; supervision gorunumleri
    uid-seed'li rastgele oldugu icin onlarla ayni sey yapilamazdi.
    Bu objeler egitimde HIC gorulmez (val split)."""
    probe = []
    for uid in uids:
        with open(os.path.join(renders_dir, uid, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        canon = list(meta["canonical_indices"])
        if len(canon) < 3:
            continue
        views = meta["views"]
        master = meta.get("resolution", 512)

        def K_for(i, dst):
            K = torch.tensor(views[i]["intrinsic"], dtype=torch.float32)
            return cameras.scale_intrinsics(K, master, dst)

        def c2w(i):
            return torch.linalg.inv(torch.tensor(views[i]["extrinsic"], dtype=torch.float32))

        i_in = canon[0]
        rgba = _load_rgba(os.path.join(renders_dir, uid, views[i_in]["file"]), input_res)
        img = rgba[:3] * rgba[3:4] + (1.0 - rgba[3:4])          # temiz beyaz bg
        tg_rgb, tg_c2w, tg_K = [], [], []
        for i in canon[1:3]:
            r = _load_rgba(os.path.join(renders_dir, uid, views[i]["file"]), render_res)
            tg_rgb.append(r[:3] * r[3:4] + (1.0 - r[3:4]))      # beyaz kompozit hedef
            tg_c2w.append(c2w(i)); tg_K.append(K_for(i, render_res))
        probe.append(dict(
            uid=uid,
            ii=img[None].to(device), ic=c2w(i_in)[None].to(device),
            ik=K_for(i_in, input_res)[None].to(device),
            tc=torch.stack(tg_c2w).to(device), tk=torch.stack(tg_K).to(device),
            gt=torch.stack(tg_rgb).to(device)))
    return probe


@torch.no_grad()
def val_metrics(model, probe, render_res, device="cuda"):
    """Genelleme olcumu -- TEK fotodan, gorulmemis objelerde.
      psnr   : rekonstruksiyon kalitesi (dB)
      top1   : tahmin kendi objesinin GT'sine mi en yakin (conditioning testi;
               sans = 1/N). Dusukse model girdiyi kullanmiyor demektir.
      oran   : mse / ortalama-obje-baseline. 1.0 = 'herkese ayni blob' cokusu,
               kucuk = gercekten objeye ozgu cikti.
    """
    if not probe:
        return None
    model.eval()
    preds, gts = [], []
    for d in probe:
        rgb, _ = model(d["ii"], d["ic"], d["ik"], d["tc"], d["tk"],
                       (render_res, render_res), bg_color=torch.ones(3, device=device))
        preds.append(rgb); gts.append(d["gt"])
    P, G = torch.stack(preds), torch.stack(gts)        # (N,2,3,H,W)
    mse_i = ((P - G) ** 2).mean((1, 2, 3, 4))
    psnr = (-10 * torch.log10(mse_i.clamp_min(1e-9))).mean().item()
    D = ((P[:, None] - G[None]) ** 2).mean((2, 3, 4, 5))
    top1 = (D.argmin(1) == torch.arange(len(probe), device=device)).float().mean().item()
    mean_mse = ((P.mean(0, keepdim=True) - G) ** 2).mean().item()
    model.train()
    mse = mse_i.mean().item()
    return dict(psnr=psnr, top1=top1, mse=mse, mean_mse=mean_mse,
                ratio=mse / max(mean_mse, 1e-9), n=len(probe))


def save_val_grid(model, dataset, uids, device, path, render_res):
    model.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    with torch.no_grad():
        for uid in uids:
            idx = dataset.uids.index(uid)
            it = dataset[idx]
            to = lambda t: t.to(device)
            rgb, _ = model(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]),
                           to(it["sup_c2w"][:1]), to(it["sup_K"][:1]), (render_res, render_res))
            pr = (rgb[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            gt = (it["sup_rgb"][0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            rows.append(np.concatenate([gt, pr], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    model.train()


def train(train_list, renders_dir, steps=40000, micro_batch=2, grad_accum=4,
          render_res=128, n_sup=4, lr=4e-4, warmup=500, ckpt_every=2000,
          val_every=1000, resume=False, device="cuda", amp=True, grad_ckpt=False,
          n_samples=48, w_lpips=0.25, w_tv=5e-4, low_res=64, coarse_frac=0.5,
          cross_attn=False, augment=True, teacher_subset=512, w_distill=1.0,
          teacher_off=0.5, teacher_lr=1e-2, teacher_init="", freeze_nerf=True,
          train_encoder=False, unfreeze_last=0, val_n=64, max_input=None):
    logger = setup_logging()
    # TF32: Blackwell'de matmul'u hizlandirir, cikti fp32 kalir (bf16'nin renk-oldurme
    # sorunu YOK). cudnn.benchmark: sabit boyutlu render icin en hizli kerneli sec.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    train_ds = LRMDataset(train_list, renders_dir, split="train",
                          render_res=render_res, n_sup=n_sup, augment=augment,
                          max_input=max_input,
                          # gorunum rastgeleligi augment'ten BAGIMSIZ olmali:
                          # --no_augment ile deterministik olunca her obje hep ayni
                          # 4 supervision gorunumunu goruyordu (16 render'in 12'si olu)
                          deterministic=False)
    val_ds = LRMDataset(train_list, renders_dir, split="val",
                        render_res=render_res, n_sup=n_sup, augment=False)
    val_uids = val_ds.uids[:6]
    # Sayisal val: sabit N obje, sabit kanonik kameralar, tek girdi.
    val_probe = build_val_probe(renders_dir, val_ds.uids[:val_n], render_res,
                                device=device)
    val_jsonl = os.path.join(LOG_DIR, "val_metrics.jsonl")
    logger.info(f"egitim: {len(train_ds)} train, {len(val_ds)} val | render {render_res} "
                f"| micro_batch {micro_batch} x grad_accum {grad_accum} | lr {lr} "
                f"| amp {amp} | grad_ckpt {grad_ckpt} | cross_attn {cross_attn} "
                f"| augment {augment}")

    model = LRM(n_samples=n_samples, cross_attn=cross_attn).to(device)
    if grad_ckpt:  # VRAM bol oldugu icin varsayilan kapali (hiz icin)
        model.transformer.enable_checkpointing()
    # LPIPS agirligi 0.25 (OLCULDU, 32-obje tezgahi, ayni 2000 adimlik program):
    #   w_lpips 2.0 -> PSNR 17.06, top1 %25, mse/ortalama-baseline 0.832
    #   w_lpips 0.25 -> PSNR 18.83, top1 %62, oran 0.556   <-- optimum
    #   w_lpips 0.0  -> PSNR 18.09, top1 %66, oran 0.629 (gurultulu cikti)
    # LRM'in 2.0'i 730k obje + batch 1024 rejimi icin; bizim rejimde LPIPS baskin
    # olunca model 'makul genel doku' yani ortalama-obje havzasinda kaliyor.
    # Encoder'i cozmek: LRM/OpenLRM/TripoSR ucu de encoder'i EGITIR. Bizde donuk
    # DINOv2 + 13k obje rejiminde tavan yaptigi olculmustu; --unfreeze_last N ile
    # son N blok acilir (tam cozme 21M parametre daha ekler, VRAM/hiz maliyeti var).
    if train_encoder:
        for p_ in model.encoder.model.parameters():
            p_.requires_grad_(True)
    elif unfreeze_last > 0:
        for blk in model.encoder.model.blocks[-unfreeze_last:]:
            for p_ in blk.parameters():
                p_.requires_grad_(True)
        for p_ in model.encoder.model.norm.parameters():
            p_.requires_grad_(True)
    if train_encoder or unfreeze_last > 0:
        # encoder.forward @torch.no_grad ile sarili; gradyan aksin diye ac
        model.encoder.forward = model.encoder.forward.__wrapped__.__get__(model.encoder)

    loss_fn = LRMLoss(use_lpips=True, w_lpips=w_lpips).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"egitilebilir parametre: {sum(p.numel() for p in params)/1e6:.1f}M "
                f"| enc_train={train_encoder or unfreeze_last} | val_probe={len(val_probe)}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        prog = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_step = 0
    _resume_teacher = _resume_opt_t = None
    last_ckpt = os.path.join(CKPT_DIR, "last.pt")
    if resume and os.path.isfile(last_ckpt):
        start_step, _resume_teacher, _resume_opt_t = load_checkpoint(
            last_ckpt, model, opt, scheduler)
        logger.info(f"resume: step {start_step}")

    # OGRETMEN TRIPLANE (docs/lrm-blob-diagnosis.md bolum 9):
    # render kaybinin hedefi dogru ama model rastgele baslangictan 'ortalama obje'
    # yerel minimumuna dusuyor. Obje basina SERBEST triplane ayni kayipla hizla
    # dogru cozume gidiyor (araya transformer girmiyor); LRM ona distile olunca
    # dogru havzada basliyor ve ogretmen kapandiktan sonra kendi basina iyilesiyor.
    # 32 objede: 17.06 dB/top1 %25 -> 26.68 dB/top1 %100.
    # Bellek: obje basina 1.57 MB => tum 2635 sigmaz; ilk teacher_subset objede kosulur
    # (mekanizma orada ogrenilir, geri kalanina genellenir).
    # ASAMA 1 ciktisi varsa oradan basla (scripts/fit_teacher.py). Olculdu: egitim
    # dongusu icinde sifirdan yetisen ogretmen obje basina sadece ~34 guncelleme
    # aliyor ve ogrenciden KOTU kaliyor (15.02 vs 16.38 dB) => onyukleme tersine
    # calisiyordu. Onceden oturtulmus ogretmen bu sorunu kokten cozer.
    init_tp = None
    if teacher_init:
        _ck_t = torch.load(teacher_init, map_location="cpu")
        init_tp = _ck_t["triplanes"]
        # NeRF'i SADECE sifirdan baslarken yukle. Resume'da checkpoint'teki egitilmis
        # NeRF gecerlidir; burada uzerine yazmak saatlerce ilerlemeyi sessizce siler.
        if start_step == 0:
            model.nerf.load_state_dict(_ck_t["nerf"])  # triplane'ler bu NeRF'e gore fit
        else:
            logger.info("resume: asama-1 NeRF'i YUKLENMEDI (checkpoint'teki gecerli)")
        teacher_subset = (min(teacher_subset, init_tp.shape[0]) if teacher_subset
                          else init_tp.shape[0])
        logger.info(f"ogretmen onyuklendi: {teacher_init} "
                    f"({init_tp.shape[0]} obje), NeRF de yuklendi")
    n_teacher = min(teacher_subset, len(train_ds))
    teacher = opt_t = None
    if n_teacher > 0 and w_distill > 0:
        teacher = nn.Parameter(torch.randn(n_teacher, 3, 32, 64, 64, device=device) * 0.1)
        if init_tp is not None:
            with torch.no_grad():
                teacher.copy_(init_tp[:n_teacher].to(device))
        # teacher_lr<=0 => ogretmen DONUK. Onceden oturtulmus ogretmeni (asama 1)
        # Adam'i sifirdan baslatarak guncellemek onu yerinden oynatir; dogrulanmis
        # zincirde (T2->E1) distilasyon hedefi SABITTI.
        opt_t = torch.optim.Adam([teacher], lr=teacher_lr) if teacher_lr > 0 else None
        if opt_t is None:
            teacher.requires_grad_(False)   # bos gradyan biriktirmesin
        if opt_t is not None and _resume_teacher is not None and _resume_teacher.shape == teacher.shape:
            with torch.no_grad():
                teacher.copy_(_resume_teacher.to(device))
            opt_t.load_state_dict(_resume_opt_t)
            logger.info("ogretmen checkpoint'ten geri yuklendi")
        logger.info(f"ogretmen triplane: {n_teacher} obje "
                    f"({teacher.numel() * 4 / 1e9:.2f} GB), {int(teacher_off * steps)}. "
                    f"adimda kapanir, w_distill={w_distill}, "
                    f"{'DONUK' if opt_t is None else f'lr={teacher_lr}'}, "
                    f"NeRF {'ogretmen fazinda donuk' if freeze_nerf else 'egitiliyor'}")

    def render_teacher(tp, c2w, K, res, bg):
        os_, ds_ = [], []
        for v in range(c2w.shape[0]):
            o, d = cameras.rays_from_camera(c2w[v], K[v], res, res)
            os_.append(o); ds_.append(d)
        o, d = torch.cat(os_).to(device), torch.cat(ds_).to(device)

        def q(pts):
            den, rgbq = model.nerf(sample_triplane(tp, pts, bound=model.bound))
            ins = (pts.abs().amax(-1, keepdim=True) <= model.bound).to(den.dtype)
            return den * ins, rgbq
        rgb, acc = volume_render(o, d, model.near, model.far, n_samples, q, bg_color=bg)
        V = c2w.shape[0]
        return (rgb.reshape(V, res, res, 3).permute(0, 3, 1, 2),
                acc.reshape(V, res, res, 1).permute(0, 3, 1, 2))

    rng = np.random.default_rng(0)
    model.train()
    t0 = time.time()
    import contextlib
    coarse_until = int(coarse_frac * steps)
    for step in range(start_step, steps):
        opt.zero_grad()
        if opt_t is not None:
            opt_t.zero_grad()
        use_teacher = teacher is not None and step < int(teacher_off * steps)
        # OLCULDU (2026-08-21): paylasilan NeRF, ogrencinin erken 'bos uret'
        # cozumune uyum saglayip yogunlugu sifira cokertiyor (acc=0.000) ve ayni
        # NeRF ogretmenin IYI triplane'lerini de bos render ediyor (28.9 dB -> 17.5).
        # Cozum: asama-1'den gelen saglam NeRF ogretmen fazinda DONUK kalir; ogrenci
        # triplane'ini ogretmenin uzayina tasimak zorunda kalir. Ogretmen kapaninca acilir.
        if freeze_nerf and teacher is not None:
            for _p in model.nerf.parameters():
                _p.requires_grad_(not use_teacher)
        agg = {"total": 0.0, "mse": 0.0, "mask": 0.0, "lpips": 0.0, "tv": 0.0}
        # coarse-to-fine: erken adimlar dusuk cozunurluk (render maliyeti ~res^2 => cok
        # daha hizli kaba geometri), sonra tam cozunurlukte rafine (LRM/OpenLRM recetesi)
        cur_res = low_res if step < coarse_until else render_res
        for _ in range(grad_accum):
            for _ in range(micro_batch):
                ds_idx = int(rng.integers(len(train_ds)))
                it = train_ds[ds_idx]
                to = lambda t: t.to(device)
                ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if amp else contextlib.nullcontext())
                # her ornek RASTGELE arka plan rengine kompozitlenir => model
                # girdiyi yok sayip sabit bg basma kisayolunu kullanamaz (renk-cokme fix)
                alpha = to(it["sup_alpha"])
                prem = to(it["sup_premult"])
                if cur_res != render_res:  # hedefleri render cozunurluguene indir
                    alpha = F.interpolate(alpha, size=(cur_res, cur_res), mode="bilinear",
                                          align_corners=False)
                    prem = F.interpolate(prem, size=(cur_res, cur_res), mode="bilinear",
                                         align_corners=False)
                c = torch.rand(3, device=device)
                target = prem + (1.0 - alpha) * c[None, :, None, None]
                with ctx:
                    rgb, acc = model(to(it["input_imgs"]), to(it["input_c2w"]),
                                     to(it["input_K"]), to(it["sup_c2w"]), to(it["sup_K"]),
                                     (cur_res, cur_res), bg_color=c)
                    total, parts = loss_fn(rgb, acc, target, alpha)
                    if w_tv > 0:  # triplane duzgunluk regülarizasyonu (anti-sis)
                        tv = model.tv_loss(model._last_triplane)
                        total = total + w_tv * tv
                        parts["tv"] = tv.detach()
                    if use_teacher and ds_idx < n_teacher:
                        tp_t = teacher[ds_idx]
                        rgb_t, acc_t = render_teacher(tp_t, to(it["sup_c2w"]),
                                                      to(it["sup_K"]), cur_res, c)
                        lt, _ = loss_fn(rgb_t, acc_t, target, alpha)
                        # distilasyon ogretmenin varyansina normalize (olcekten bagimsiz)
                        td = tp_t.detach()
                        total = total + lt + w_distill * (
                            (model._last_triplane - td) ** 2).mean() / td.var().clamp_min(1e-6)
                    parts["total"] = total.detach()
                (total / (grad_accum * micro_batch)).backward()
                for k in agg:
                    if k in parts:
                        agg[k] += parts[k].item() / (grad_accum * micro_batch)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if use_teacher and opt_t is not None:
            opt_t.step()
        scheduler.step()

        if step % 20 == 0:
            speed = (step - start_step + 1) / (time.time() - t0)
            vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            logger.info(f"step {step}/{steps} loss={agg['total']:.4f} "
                        f"mse={agg['mse']:.4f} mask={agg['mask']:.4f} lpips={agg['lpips']:.4f} "
                        f"tv={agg['tv']:.4f} res={cur_res} "
                        f"{'ogr' if use_teacher else '---'} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} {speed:.2f}it/s vram={vram:.1f}GB")
        if step > 0 and step % val_every == 0:
            save_val_grid(model, val_ds, val_uids, device,
                          os.path.join(PREVIEW_DIR, f"val_{step:06d}.png"), render_res)
            vm = val_metrics(model, val_probe, render_res, device)
            if vm:
                logger.info(f"[VAL step {step}] psnr={vm['psnr']:.2f}dB "
                            f"top1={vm['top1']:.1%} (sans {1/vm['n']:.1%}) "
                            f"mse={vm['mse']:.4f} ortalama-baseline={vm['mean_mse']:.4f} "
                            f"oran={vm['ratio']:.3f}")
                with open(val_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(step=step, **vm)) + chr(10))
        if step > 0 and step % ckpt_every == 0:
            save_checkpoint(last_ckpt, model, opt, scheduler, step,
                            teacher if opt_t is not None else None, opt_t)
            logger.info(f"checkpoint kaydedildi: {last_ckpt}")
    save_checkpoint(last_ckpt, model, opt, scheduler, steps)
    logger.info("egitim bitti")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list.json")
    ap.add_argument("--renders_dir", default="dataset/renders")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--render_res", type=int, default=128)
    ap.add_argument("--n_sup", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    # DIKKAT: bf16 renk gradyanini olduruyor (sadece geometri ogreniliyor, renk siyah
    # kaliyor). Bu yuzden VARSAYILAN fp32. Hiz icin bilerek bf16 istersen --amp.
    ap.add_argument("--amp", action="store_true",
                    help="bf16 mixed precision (DIKKAT: rengi ogrenmiyor, sadece hiz denemesi icin)")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="gradient checkpointing (VRAM darsa; varsayilan kapali)")
    ap.add_argument("--n_samples", type=int, default=48,
                    help=" isin basi ornek (hiz/kalite; overfit'te 48 kusursuz)")
    ap.add_argument("--w_lpips", type=float, default=0.25,
                    help="LPIPS agirligi (olculdu: 0.25 optimum; 2.0 conditioning'i olduruyor)")
    ap.add_argument("--w_tv", type=float, default=5e-4,
                    help="triplane TV regülarizasyonu (anti-sis; 0=kapat)")
    ap.add_argument("--low_res", type=int, default=64,
                    help="coarse-to-fine dusuk cozunurluk (erken adimlar; hiz)")
    ap.add_argument("--coarse_frac", type=float, default=0.5,
                    help="egitimin ilk bu orani low_res'te (0=kapat, hep render_res)")
    ap.add_argument("--cross_attn", action="store_true",
                    help="LRM-tarzi cross-attention + kamera modLN govdesi. OLCULDU: "
                         "32-obje tezgahinda joint self-attention ile AYNI sonuc "
                         "(17.05 vs 17.06 dB) ama %80 daha fazla parametre. "
                         "Varsayilan kapali; kok neden mimaride degildi "
                         "(bkz. docs/lrm-blob-diagnosis.md).")
    ap.add_argument("--teacher_subset", type=int, default=512,
                    help="ogretmen triplane'i ilk N objede kosar (0=kapat). Bellek: "
                         "obje basina 1.57 MB")
    ap.add_argument("--w_distill", type=float, default=1.0,
                    help="ogrenci->ogretmen distilasyon agirligi (0=ogretmeni kapat)")
    ap.add_argument("--teacher_off", type=float, default=0.5,
                    help="egitimin bu oranindan sonra ogretmen kapanir")
    ap.add_argument("--teacher_lr", type=float, default=1e-2)
    ap.add_argument("--no_freeze_nerf", action="store_true",
                    help="ogretmen fazinda NeRF'i dondurma (varsayilan: dondur)")
    ap.add_argument("--teacher_init", default="",
                    help="ASAMA 1 ciktisi (scripts/fit_teacher.py). Onceden oturtulmus "
                         "ogretmen triplane'leri + NeRF. Olculdu: dongu icinde sifirdan "
                         "yetisen ogretmen ogrenciden kotu kaliyor.")
    ap.add_argument("--train_encoder", action="store_true",
                    help="DINOv2'nin TAMAMINI egit")
    ap.add_argument("--unfreeze_last", type=int, default=0,
                    help="DINOv2'nin son N blogunu egit (0=donuk)")
    ap.add_argument("--val_n", type=int, default=64,
                    help="sayisal val'de kullanilacak gorulmemis obje sayisi")
    ap.add_argument("--max_input", type=int, default=None,
                    help="girdi gorunum tavani (None=meta'daki kanonik sayisi)")
    ap.add_argument("--no_augment", action="store_true",
                    help="girdi augmentation'i kapat (sistem kanitlama asamasinda "
                         "temiz girdi; robustluk sonra)")
    a = ap.parse_args()
    train(a.train_list, a.renders_dir, a.steps, a.micro_batch, a.grad_accum,
          a.render_res, a.n_sup, a.lr, a.warmup, a.ckpt_every, a.val_every,
          resume=a.resume, amp=a.amp, grad_ckpt=a.grad_ckpt, n_samples=a.n_samples,
          w_lpips=a.w_lpips, w_tv=a.w_tv, low_res=a.low_res, coarse_frac=a.coarse_frac,
          cross_attn=a.cross_attn, augment=not a.no_augment,
          teacher_subset=a.teacher_subset, w_distill=a.w_distill,
          teacher_off=a.teacher_off, teacher_lr=a.teacher_lr,
          teacher_init=a.teacher_init, freeze_nerf=not a.no_freeze_nerf,
          train_encoder=a.train_encoder, unfreeze_last=a.unfreeze_last,
          val_n=a.val_n, max_input=a.max_input)
