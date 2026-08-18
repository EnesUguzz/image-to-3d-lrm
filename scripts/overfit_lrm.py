"""Overfit saglik testi: 1-2 objede bilerek ezberlet. Loss ~0'a inip render
GT'ye oturmali. Tam egitimden ONCE pipeline'i dogrular."""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.losses import LRMLoss

PREVIEW_DIR = "dataset/lrm_val_previews"


def _save_preview(pred_rgb, gt_rgb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pr = (pred_rgb[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    gt = (gt_rgb[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    grid = np.concatenate([gt, pr], axis=1)
    Image.fromarray(grid).save(path)


def overfit(train_list, renders_dir, n_obj=2, steps=500, render_res=64,
            device="cuda", lr=4e-4, use_lpips=True):
    ds = LRMDataset(train_list, renders_dir, split="train", input_res=224,
                    render_res=render_res, n_sup=4, augment=False)
    ds.uids = ds.uids[:n_obj]
    model = LRM(n_samples=48).to(device)
    model.transformer.enable_checkpointing()
    loss_fn = LRMLoss(use_lpips=use_lpips).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        it = ds[step % len(ds)]
        to = lambda t: t.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rgb, acc = model(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]),
                             to(it["sup_c2w"]), to(it["sup_K"]), (render_res, render_res))
            total, parts = loss_fn(rgb, acc, to(it["sup_rgb"]), to(it["sup_alpha"]))
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())
        if step % 50 == 0:
            print(f"step {step}: loss={total.item():.4f} "
                  f"mse={parts['mse'].item():.4f} mask={parts['mask'].item():.4f}")
            _save_preview(rgb, to(it["sup_rgb"]),
                          os.path.join(PREVIEW_DIR, f"overfit_{it['uid'][:8]}.png"))
    return losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list.json")
    ap.add_argument("--renders_dir", default="dataset/renders")
    ap.add_argument("--n_obj", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--render_res", type=int, default=64)
    ap.add_argument("--no_lpips", action="store_true")
    a = ap.parse_args()
    losses = overfit(a.train_list, a.renders_dir, a.n_obj, a.steps, a.render_res,
                     use_lpips=not a.no_lpips)
    print(f"ilk loss={losses[0]:.4f}  son loss={losses[-1]:.4f}")
    print("BASARILI: pipeline ogreniyor" if losses[-1] < losses[0] * 0.3
          else "DIKKAT: loss yeterince dusmedi - bug arastir")
