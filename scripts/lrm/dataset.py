"""LRMDataset: render + poz okur; her ornekte 1-4 girdi + n_sup supervision secer.
Girdiye augmentation uygulanir, supervision temiz kalir."""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lrm import cameras
from lrm.augment import augment_input

RENDER_MASTER_RES = 512  # meta intrinsic bu cozunurluge gore


def _load_rgba(path, res):
    im = Image.open(path).convert("RGBA")
    arr = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0  # (4,H,W)
    if arr.shape[-1] != res:
        arr = F.interpolate(arr[None], size=(res, res), mode="bilinear",
                            align_corners=False)[0]
    return arr


class LRMDataset(torch.utils.data.Dataset):
    def __init__(self, train_list_path, renders_dir, split="train",
                 input_res=224, render_res=128, n_sup=4, augment=True, seed=0):
        with open(train_list_path, encoding="utf-8") as f:
            self.uids = json.load(f)[split]
        self.renders_dir = renders_dir
        self.input_res = input_res
        self.render_res = render_res
        self.n_sup = n_sup
        self.augment = augment
        self.base_seed = seed

    def __len__(self):
        return len(self.uids)

    def _meta(self, uid):
        with open(os.path.join(self.renders_dir, uid, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        meta = self._meta(uid)
        rng = random.Random(self.base_seed * 1_000_003 + idx)
        canon = list(meta["canonical_indices"])
        views = meta["views"]
        n_views = len(views)
        master_res = meta.get("resolution", RENDER_MASTER_RES)

        k = rng.randint(1, min(4, len(canon)))
        input_idx = rng.sample(canon, k)
        remaining = [i for i in range(n_views) if i not in input_idx]
        sup_idx = rng.sample(remaining, min(self.n_sup, len(remaining)))

        def K_for(i, dst_res):
            # intrinsic Faz A'da her view'de saklanir (master cozunurluk icin)
            K = torch.tensor(views[i]["intrinsic"], dtype=torch.float32)
            return cameras.scale_intrinsics(K, master_res, dst_res)

        def c2w(i):
            ext = torch.tensor(views[i]["extrinsic"], dtype=torch.float32)
            return torch.linalg.inv(ext)  # extrinsic = world->camera

        def path(i):
            return os.path.join(self.renders_dir, uid, views[i]["file"])

        input_imgs, input_c2w, input_K = [], [], []
        for i in input_idx:
            rgba = _load_rgba(path(i), self.input_res)
            if self.augment:
                img = augment_input(rgba, random.Random(rng.random() * 1e9))
            else:
                img = rgba[:3] * rgba[3:4] + (1.0 - rgba[3:4])  # temiz: BEYAZ bg
            input_imgs.append(img)
            input_c2w.append(c2w(i))
            input_K.append(K_for(i, self.input_res))

        sup_rgb, sup_alpha, sup_c2w, sup_K = [], [], [], []
        for i in sup_idx:
            rgba = _load_rgba(path(i), self.render_res)
            # hedef BEYAZ bg uzerine (model white_bg render eder; rgb siyaha cokmez)
            sup_rgb.append(rgba[:3] * rgba[3:4] + (1.0 - rgba[3:4]))
            sup_alpha.append(rgba[3:4])
            sup_c2w.append(c2w(i))
            sup_K.append(K_for(i, self.render_res))

        return {
            "uid": uid,
            "input_imgs": torch.stack(input_imgs),
            "input_c2w": torch.stack(input_c2w),
            "input_K": torch.stack(input_K),
            "input_view_idx": input_idx,
            "sup_rgb": torch.stack(sup_rgb),
            "sup_alpha": torch.stack(sup_alpha),
            "sup_c2w": torch.stack(sup_c2w),
            "sup_K": torch.stack(sup_K),
            "sup_view_idx": sup_idx,
        }


def lrm_collate(batch):
    """Degisken girdi sayisi (1-4) => padding yerine liste dondur.
    Model tek obje isler; egitim dongusu listeyi gezer."""
    return list(batch)
