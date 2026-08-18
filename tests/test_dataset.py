import json
import os

import numpy as np
import torch
from PIL import Image

from lrm.dataset import LRMDataset, lrm_collate


def _make_obj(root, uid, n_views=16, res=64):
    d = os.path.join(root, uid)
    os.makedirs(d, exist_ok=True)
    metas = {"intrinsic": [[400, 0, 256], [0, 400, 256], [0, 0, 1]],
             "canonical_indices": [0, 1, 2, 3], "views": []}
    for i in range(n_views):
        arr = np.zeros((res, res, 4), dtype=np.uint8)
        arr[16:48, 16:48, :3] = 180
        arr[16:48, 16:48, 3] = 255
        Image.fromarray(arr, "RGBA").save(os.path.join(d, f"{i:03d}.png"))
        c2w = np.eye(4)
        c2w[2, 3] = 1.5  # extrinsic = world->camera
        metas["views"].append({"extrinsic": c2w.tolist()})
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f)


def _setup(tmp_path):
    rroot = tmp_path / "renders"
    rroot.mkdir()
    uids = [f"u{i}" for i in range(3)]
    for u in uids:
        _make_obj(str(rroot), u)
    tl = tmp_path / "train_list.json"
    with open(tl, "w", encoding="utf-8") as f:
        json.dump({"train": uids, "val": []}, f)
    return str(tl), str(rroot)


def test_item_shapes_and_input_count_in_range(tmp_path):
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", input_res=224, render_res=128, n_sup=4)
    item = ds[0]
    k = item["input_imgs"].shape[0]
    assert 1 <= k <= 4
    assert item["input_imgs"].shape[1:] == (3, 224, 224)
    assert item["input_c2w"].shape == (k, 4, 4)
    assert item["sup_rgb"].shape == (4, 3, 128, 128)
    assert item["sup_alpha"].shape == (4, 1, 128, 128)


def test_supervision_disjoint_from_input(tmp_path):
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", n_sup=4, seed=0)
    it = ds[0]
    assert set(it["input_view_idx"]).isdisjoint(set(it["sup_view_idx"]))


def test_collate_returns_list(tmp_path):
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train")
    batch = lrm_collate([ds[0], ds[1]])
    assert isinstance(batch, list) and len(batch) == 2
