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
from lrm import crop as cropmod
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
                 input_res=224, render_res=128, n_sup=4, augment=True, seed=0,
                 normalize_cams=False, deterministic=None, max_input=None,
                 force_n_input=0, region=0, render_low=64, render_high=192,
                 fg_bias=0.75):
        with open(train_list_path, encoding="utf-8") as f:
            self.uids = json.load(f)[split]
        self.renders_dir = renders_dir
        self.input_res = input_res
        self.render_res = render_res
        self.n_sup = n_sup
        self.augment = augment
        self.base_seed = seed
        # HATA (duzeltildi): seed idx'e sabitlenince her epoch AYNI girdi gorunumu,
        # AYNI supervision gorunumleri ve AYNI augmentation cikiyordu => 16 render'in
        # 12'si hic kullanilmiyor, augmentation rastgele degil (her obje kalici olarak
        # tek bir arka plan rengine yapisik). Egitimde her erisim taze rastgelelik ister;
        # val/onizleme sabit kalsin diye augment kapaliyken deterministik.
        self.deterministic = (not augment) if deterministic is None else deterministic
        # kamera normalizasyonu (LRM +3.7 PSNR): giris kamerasini kanonik poza sabitle.
        # ANCAK ezber kisayolunu kapatir => kucuk veride (2635, hatta 50K) yakinsama
        # cok yavaslar (izole test: mask 400 adimda hala 0.5). 1M+ olcekte faydali.
        # Bu yuzden bizim rejimde VARSAYILAN KAPALI; buyuk veride tekrar denenecek.
        self.normalize_cams = normalize_cams
        # Girdi gorunum sayisi tavani. None = meta'daki kanonik sayisi kadar
        # (ring12 semasinda 4, sphere20 semasinda 6 -> ust/alt de girdi olabilir).
        self.max_input = max_input
        # force_n_input>0: k'yi sabitler. Val metriginde sart -- yoksa her objenin
        # girdi sayisi farkli olur ve "tek fotodan rekonstruksiyon" olcumu karisir.
        self.force_n_input = force_n_input
        # BOLGE KIRPMA (OpenLRM/TripoSR). region>0 ise supervision hedefi, U[low,high]
        # cozunurlukte render edilmis kareden kirpilan region x region bir yamadir.
        # Isin butcesi region^2'de SABIT kalir ama yamadaki obje orani cok yukselir
        # (objelerimiz tam karenin sadece ~%9'unu kapliyor). Bkz. lrm/crop.py.
        self.region = region
        self.render_low = render_low
        self.render_high = render_high
        self.fg_bias = fg_bias

    def __len__(self):
        return len(self.uids)

    def _meta(self, uid):
        with open(os.path.join(self.renders_dir, uid, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        meta = self._meta(uid)
        rng = (random.Random(self.base_seed * 1_000_003 + idx)
               if self.deterministic else random.Random())
        canon = list(meta["canonical_indices"])
        views = meta["views"]
        n_views = len(views)
        master_res = meta.get("resolution", RENDER_MASTER_RES)

        hi = len(canon) if self.max_input is None else min(self.max_input, len(canon))
        k = self.force_n_input if self.force_n_input else rng.randint(1, hi)
        k = max(1, min(k, len(canon)))
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

        sup_rgb, sup_premult, sup_alpha, sup_c2w, sup_K = [], [], [], [], []
        for i in sup_idx:
            if self.region:
                # her gorunum kendi cozunurlugunu ve kendi capasini alir (OpenLRM)
                r = cropmod.sample_render_res(rng, self.render_low,
                                              self.render_high, self.region)
                rgba = _load_rgba(path(i), r)
                ax, ay = cropmod.sample_anchor(rgba[3:4], r, self.region, rng,
                                               self.fg_bias)
                rgba = cropmod.crop_image(rgba, ax, ay, self.region)
                K = cropmod.scale_and_crop_K(
                    torch.tensor(views[i]["intrinsic"], dtype=torch.float32),
                    master_res, r, ax, ay)
            else:
                rgba = _load_rgba(path(i), self.render_res)
                K = K_for(i, self.render_res)
            # premultiplied obje rengi (obj*alpha): egitimde her adim RASTGELE bg
            # rengine kompozitlenir => model sabit ciktiyla arka plani tutturamaz,
            # objeyi gercekten kurar (sabit-bg 'renk cokmesi' tuzagini kapatir).
            sup_premult.append(rgba[:3] * rgba[3:4])
            # sup_rgb: beyaz-kompozit (yalnizca gorsel onizleme/val icin)
            sup_rgb.append(rgba[:3] * rgba[3:4] + (1.0 - rgba[3:4]))
            sup_alpha.append(rgba[3:4])
            sup_c2w.append(c2w(i))
            sup_K.append(K)

        in_c2w = torch.stack(input_c2w)
        su_c2w = torch.stack(sup_c2w)
        if self.normalize_cams:
            # referans = ilk giris kamerasi; ayni rotasyonu giris+supervision'a uygula
            ref = in_c2w[0]
            in_c2w = cameras.canonicalize(ref, in_c2w)
            su_c2w = cameras.canonicalize(ref, su_c2w)

        return {
            "uid": uid,
            "idx": idx,
            "input_imgs": torch.stack(input_imgs),
            "input_c2w": in_c2w,
            "input_K": torch.stack(input_K),
            "input_view_idx": input_idx,
            "sup_rgb": torch.stack(sup_rgb),
            "sup_premult": torch.stack(sup_premult),
            "sup_alpha": torch.stack(sup_alpha),
            "sup_c2w": su_c2w,
            "sup_K": torch.stack(sup_K),
            "sup_view_idx": sup_idx,
            "sup_res": self.region or self.render_res,
        }


def lrm_collate(batch):
    """Degisken girdi sayisi (1-4) => padding yerine liste dondur.
    Model tek obje isler; egitim dongusu listeyi gezer."""
    return list(batch)
