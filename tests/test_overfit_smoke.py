import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.losses import LRMLoss


class FakeEncoder(nn.Module):
    """Deterministik sahte encoder: gercek DINOv2 gibi ayni girdiye ayni token.
    (Her adim rastgele olsaydi model ezberleyecek sinyal bulamazdi.)"""
    embed_dim = 384
    patch = 14

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("fixed", torch.randn(4, 256, 384, generator=g))

    def forward(self, imgs):
        return self.fixed[:imgs.shape[0]]


def _tiny_dataset(tmp_path):
    rroot = tmp_path / "renders"
    rroot.mkdir()
    uid = "u0"
    d = rroot / uid
    d.mkdir()
    meta = {"intrinsic": [[400, 0, 256], [0, 400, 256], [0, 0, 1]],
            "canonical_indices": [0, 1, 2, 3], "views": []}
    for i in range(16):
        arr = np.zeros((64, 64, 4), np.uint8)
        arr[20:44, 20:44, :3] = 200
        arr[20:44, 20:44, 3] = 255
        Image.fromarray(arr, "RGBA").save(d / f"{i:03d}.png")
        c2w = np.eye(4)
        c2w[2, 3] = 1.5
        meta["views"].append({"extrinsic": c2w.tolist()})
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    tl = tmp_path / "tl.json"
    with open(tl, "w", encoding="utf-8") as f:
        json.dump({"train": [uid], "val": []}, f)
    return str(tl), str(rroot)


def test_overfit_loss_decreases(tmp_path):
    tl, rroot = _tiny_dataset(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", input_res=224, render_res=32,
                    n_sup=2, augment=False)
    model = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
                nerf_hidden=16, encoder=FakeEncoder(), n_samples=16)
    loss_fn = LRMLoss(use_lpips=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for step in range(200):
        it = ds[0]
        rgb, acc = model(it["input_imgs"], it["input_c2w"], it["input_K"],
                         it["sup_c2w"], it["sup_K"], (32, 32))
        total, _ = loss_fn(rgb, acc, it["sup_rgb"], it["sup_alpha"])
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())
    # deterministik girdi + sabit hedef => model ezberlemeli (loss ~0'a iner)
    assert losses[-1] < losses[0] * 0.1
