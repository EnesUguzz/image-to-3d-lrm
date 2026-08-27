"""Girdi siki kirpma (input_crop) regresyon testleri.

NEDEN (2026-08-27 bagimsiz veri denetimi): `fit` normalizasyonu 16 kameradaki
EN KOTU bbox kosesini cerceveye oturttugu icin obje kanonik goruumun sadece
~%9'unu kapliyor; DINOv2'nin 256 patch'inden ~19'u obje (p05: 4 patch).
Siluet bbox'ina kirpinca ~77 patch (4x). Faz C zaten bunu yapmak zorunda.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from lrm import cameras
from lrm.dataset import LRMDataset, _crop_input

KOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "dataset", "renders_opp_score3")
LISTE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "dataset", "train_list_v2.json")
gercek_veri = pytest.mark.skipif(not os.path.isdir(KOK), reason="render dizini yok")


def _ilk_uid():
    with open(LISTE, encoding="utf-8") as f:
        return json.load(f)["train"][0]


@gercek_veri
def test_kirpma_kaplamayi_belirgin_artiriyor():
    """Olculdu (60 obje x 4 kanonik): kaplama 0.0920 -> 0.2499 = 2.72x,
    DINOv2 patch 24 -> 64. Tek objede degisken oldugu icin ORTALAMA test edilir."""
    from lrm.dataset import _load_rgba
    with open(LISTE, encoding="utf-8") as f:
        uids = json.load(f)["train"][:12]
    ham, krp = [], []
    for uid in uids:
        with open(os.path.join(KOK, uid, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        for vi in meta["canonical_indices"]:
            v = meta["views"][vi]
            p = os.path.join(KOK, uid, v["file"])
            ham.append(float((_load_rgba(p, 224)[3] > 0.05).float().mean()))
            c, _ = _crop_input(p, 224, meta["resolution"], v["intrinsic"], pad=1.15)
            krp.append(float((c[3] > 0.05).float().mean()))
    oran = float(np.mean(krp)) / float(np.mean(ham))
    assert oran > 2.0, "beklenen ~2.7x kazanc, olculen %.2fx" % oran


@gercek_veri
def test_kirpilmis_K_ile_isin_denkligi():
    """KRITIK: kirpilmis intrinsic, TAM KAREDEKI ayni fiziksel isini uretmeli.

    Yanlissa model dogru goruntuyu YANLIS kameradan gordugunu sanir; sessizce
    bozuk conditioning. (Ayni test bolge kirpmasi icin test_region_crop.py'de var.)
    """
    uid = _ilk_uid()
    with open(os.path.join(KOK, uid, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    v = meta["views"][meta["canonical_indices"][0]]
    p = os.path.join(KOK, uid, v["file"])
    R = 224
    _, Kc = _crop_input(p, R, meta["resolution"], v["intrinsic"], pad=1.15)
    c2w = torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float32))

    # Kirpilmis goruntunun MERKEZ isini, tam karede bbox merkezinden gecen
    # isinla ayni olmali. Tam kare isinlarini master cozunurlukte uret.
    Kf = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                  meta["resolution"], meta["resolution"])
    o_c, d_c = cameras.rays_from_camera(c2w, Kc, R, R)
    o_f, d_f = cameras.rays_from_camera(c2w, Kf, meta["resolution"], meta["resolution"])

    from PIL import Image
    a = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32)[..., 3] / 255.0
    ys, xs = np.nonzero(a > 0.05)
    cx = int(round(0.5 * (xs.min() + xs.max())))
    cy = int(round(0.5 * (ys.min() + ys.max())))

    merkez_kirpma = d_c.reshape(R, R, 3)[R // 2, R // 2]
    merkez_tam = d_f.reshape(meta["resolution"], meta["resolution"], 3)[cy, cx]
    cos = float((merkez_kirpma * merkez_tam).sum())
    assert cos > 0.999, "kirpma merkezi isini kaymis (cos=%.5f)" % cos


@gercek_veri
def test_dataset_bayragi_uctan_uca():
    ds_ham = LRMDataset(LISTE, KOK, split="train", input_res=224, render_res=64,
                        n_sup=2, augment=False, input_crop=0.0)
    ds_krp = LRMDataset(LISTE, KOK, split="train", input_res=224, render_res=64,
                        n_sup=2, augment=False, input_crop=1.15)
    a, b = ds_ham[0], ds_krp[0]
    assert a["input_imgs"].shape == b["input_imgs"].shape
    assert a["input_K"].shape == b["input_K"].shape
    assert not torch.allclose(a["input_K"], b["input_K"]), "kirpma K'yi degistirmeli"


def test_bos_render_cokmuyor(tmp_path):
    """Tamamen seffaf girdi: kirpma kodu bolme hatasi vermemeli."""
    from PIL import Image
    p = tmp_path / "bos.png"
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(p)
    K = [[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]]
    arr, Kc = _crop_input(str(p), 32, 64, K, pad=1.15)
    assert arr.shape == (4, 32, 32)
    assert torch.isfinite(Kc).all()
