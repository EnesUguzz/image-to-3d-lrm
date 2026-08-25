"""Teshis: mevcut checkpoint train setindeki (GORULEN) objeleri keskin
render ediyor mu? Train keskin + val blob => genelleme acigi (veri az).
Train de blob => conditioning bug (kod)."""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lrm.dataset import LRMDataset
from lrm.model import LRM

CKPT = "dataset/lrm_ckpts/last.pt"
OUT = "dataset/lrm_val_previews/diag_train_vs_val.png"
RENDER_RES = 128
DEVICE = "cuda"


def render_rows(model, ds, uids, device):
    rows = []
    with torch.no_grad():
        for uid in uids:
            it = ds[ds.uids.index(uid)]
            to = lambda t: t.to(device)
            rgb, _ = model(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]),
                           to(it["sup_c2w"][:1]), to(it["sup_K"][:1]), (RENDER_RES, RENDER_RES))
            pr = (rgb[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            gt = (it["sup_rgb"][0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            rows.append(np.concatenate([gt, pr], axis=1))
    return rows


def main():
    train_ds = LRMDataset("dataset/train_list.json", "dataset/renders", split="train",
                          render_res=RENDER_RES, n_sup=4, augment=False)
    val_ds = LRMDataset("dataset/train_list.json", "dataset/renders", split="val",
                        render_res=RENDER_RES, n_sup=4, augment=False)

    model = LRM(n_samples=64).to(DEVICE)
    ck = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint step: {ck['step']}")

    train_rows = render_rows(model, train_ds, train_ds.uids[:6], DEVICE)
    val_rows = render_rows(model, val_ds, val_ds.uids[:6], DEVICE)

    # sol blok = TRAIN (gorulen), sag blok = VAL (gorulmemis); her blok [GT|tahmin]
    gap = np.full((train_rows[0].shape[0], 8, 3), 128, dtype=np.uint8)
    combined = [np.concatenate([tr, gap, vl], axis=1) for tr, vl in zip(train_rows, val_rows)]
    Image.fromarray(np.concatenate(combined, axis=0)).save(OUT)
    print(f"kaydedildi: {OUT}  (SOL=train[gt|tahmin] | SAG=val[gt|tahmin])")


if __name__ == "__main__":
    main()
