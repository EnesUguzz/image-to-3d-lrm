"""Tam LRM egitim dongusu: mikro-batch + gradient accumulation, bf16,
checkpoint/resume, val preview, zaman damgali log."""
import argparse
import logging
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.losses import LRMLoss

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


def save_checkpoint(path, model, opt, scheduler, step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": scheduler.state_dict(), "step": step}, path)


def load_checkpoint(path, model, opt, scheduler):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scheduler.load_state_dict(ck["sched"])
    return ck["step"]


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
          val_every=1000, resume=False, device="cuda", amp=True, grad_ckpt=False):
    logger = setup_logging()
    train_ds = LRMDataset(train_list, renders_dir, split="train",
                          render_res=render_res, n_sup=n_sup, augment=True)
    val_ds = LRMDataset(train_list, renders_dir, split="val",
                        render_res=render_res, n_sup=n_sup, augment=False)
    val_uids = val_ds.uids[:6]
    logger.info(f"egitim: {len(train_ds)} train, {len(val_ds)} val | render {render_res} "
                f"| micro_batch {micro_batch} x grad_accum {grad_accum} | lr {lr} "
                f"| amp {amp} | grad_ckpt {grad_ckpt}")

    model = LRM(n_samples=64).to(device)
    if grad_ckpt:  # VRAM bol oldugu icin varsayilan kapali (hiz icin)
        model.transformer.enable_checkpointing()
    loss_fn = LRMLoss(use_lpips=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        prog = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_step = 0
    last_ckpt = os.path.join(CKPT_DIR, "last.pt")
    if resume and os.path.isfile(last_ckpt):
        start_step = load_checkpoint(last_ckpt, model, opt, scheduler)
        logger.info(f"resume: step {start_step}")

    rng = np.random.default_rng(0)
    model.train()
    t0 = time.time()
    import contextlib
    for step in range(start_step, steps):
        opt.zero_grad()
        agg = {"total": 0.0, "mse": 0.0, "mask": 0.0, "lpips": 0.0}
        for _ in range(grad_accum):
            for _ in range(micro_batch):
                it = train_ds[int(rng.integers(len(train_ds)))]
                to = lambda t: t.to(device)
                ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if amp else contextlib.nullcontext())
                with ctx:
                    rgb, acc = model(to(it["input_imgs"]), to(it["input_c2w"]),
                                     to(it["input_K"]), to(it["sup_c2w"]), to(it["sup_K"]),
                                     (render_res, render_res))
                    total, parts = loss_fn(rgb, acc, to(it["sup_rgb"]), to(it["sup_alpha"]))
                (total / (grad_accum * micro_batch)).backward()
                for k in agg:
                    if k in parts:
                        agg[k] += parts[k].item() / (grad_accum * micro_batch)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        scheduler.step()

        if step % 20 == 0:
            speed = (step - start_step + 1) / (time.time() - t0)
            vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            logger.info(f"step {step}/{steps} loss={agg['total']:.4f} "
                        f"mse={agg['mse']:.4f} mask={agg['mask']:.4f} lpips={agg['lpips']:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} {speed:.2f}it/s vram={vram:.1f}GB")
        if step > 0 and step % val_every == 0:
            save_val_grid(model, val_ds, val_uids, device,
                          os.path.join(PREVIEW_DIR, f"val_{step:06d}.png"), render_res)
        if step > 0 and step % ckpt_every == 0:
            save_checkpoint(last_ckpt, model, opt, scheduler, step)
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
    a = ap.parse_args()
    train(a.train_list, a.renders_dir, a.steps, a.micro_batch, a.grad_accum,
          a.render_res, a.n_sup, a.lr, a.warmup, a.ckpt_every, a.val_every,
          resume=a.resume, amp=a.amp, grad_ckpt=a.grad_ckpt)
