import json
import os

import numpy as np
import torch
from PIL import Image

from lrm.dataset import LRMDataset, lrm_collate


def _make_obj(root, uid, n_views=16, res=64):
    """Faz A meta.json semasini taklit eder: resolution + per-view intrinsic/file/extrinsic."""
    d = os.path.join(root, uid)
    os.makedirs(d, exist_ok=True)
    metas = {"uid": uid, "resolution": 512, "num_views": n_views,
             "canonical_indices": [0, 1, 2, 3], "views": []}
    for i in range(n_views):
        arr = np.zeros((res, res, 4), dtype=np.uint8)
        arr[16:48, 16:48, :3] = 180
        arr[16:48, 16:48, 3] = 255
        fname = f"{i:03d}.png"
        Image.fromarray(arr, "RGBA").save(os.path.join(d, fname))
        ext = np.eye(4)
        ext[2, 3] = -1.5  # world->camera: c2w kamerayi +z'ye koyar, orijine bakar
        metas["views"].append({
            "index": i, "role": "canonical" if i < 4 else "supervision",
            "file": fname, "extrinsic": ext.tolist(),
            "intrinsic": [[400, 0, 256], [0, 400, 256], [0, 0, 1]],
        })
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
    assert item["sup_premult"].shape == (4, 3, 128, 128)
    assert item["sup_alpha"].shape == (4, 1, 128, 128)
    # premult + (1-alpha)*beyaz == beyaz-kompozit sup_rgb (random-bg egitim tutarliligi)
    recon = item["sup_premult"] + (1.0 - item["sup_alpha"])
    assert torch.allclose(recon, item["sup_rgb"], atol=1e-5)


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


def test_egitimde_her_epoch_farkli_gorunum_ve_augment(tmp_path):
    """Regresyon: seed idx'e sabitlenince her epoch ayni girdi/sup gorunumu ve
    ayni augmentation ciktisi geliyordu (16 render'in 12'si olu, augment sahte)."""
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", render_res=32, augment=True)
    assert ds.deterministic is False
    goruldu = {tuple(ds[0]["sup_view_idx"]) for _ in range(12)}
    assert len(goruldu) > 1, "supervision gorunumleri epoch'lar arasi degismiyor"
    imgs = [ds[0]["input_imgs"] for _ in range(6)]
    assert any((imgs[0].shape != i.shape) or (imgs[0] - i).abs().mean() > 1e-6
               for i in imgs[1:]), "augmentation rastgele degil"


def test_val_deterministik_kalir(tmp_path):
    """Onizleme/val kararli olsun: augment kapaliyken ayni idx ayni ornegi verir."""
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", render_res=32, augment=False)
    assert ds.deterministic is True
    assert ds[0]["sup_view_idx"] == ds[0]["sup_view_idx"]


def test_bolge_kirpma_sekilleri_ve_on_plan_orani(tmp_path):
    """region>0: hedefler region x region gelir ve on-plan orani TAM KAREYE gore
    belirgin sekilde artar (kaybin %85'i arka plana gitmesin diye)."""
    tl, rroot = _setup(tmp_path)
    full = LRMDataset(tl, rroot, split="train", render_res=64, n_sup=4,
                      augment=False, deterministic=False)
    reg = LRMDataset(tl, rroot, split="train", render_res=64, n_sup=4,
                     augment=False, deterministic=False,
                     region=32, render_low=48, render_high=96, fg_bias=1.0)
    it_f, it_r = full[0], reg[0]
    assert it_f["sup_alpha"].shape[-1] == 64 and it_f["sup_res"] == 64
    assert it_r["sup_alpha"].shape[-1] == 32 and it_r["sup_res"] == 32
    assert it_r["sup_premult"].shape[-1] == 32
    assert it_r["sup_K"].shape == (4, 3, 3)
    # fg_bias=1.0 => her kirpma obje icermeli
    assert float(it_r["sup_alpha"].amax()) > 0


def test_bolge_kirpma_on_plan_yogunlugunu_ARTIRIR(tmp_path):
    tl, rroot = _setup(tmp_path)
    full = LRMDataset(tl, rroot, split="train", render_res=64, n_sup=4,
                      augment=False, deterministic=False)
    reg = LRMDataset(tl, rroot, split="train", render_res=64, n_sup=4,
                     augment=False, deterministic=False,
                     region=24, render_low=64, render_high=64, fg_bias=1.0)
    cov_f = float(torch.stack([full[i]["sup_alpha"] for i in range(3)]).mean())
    cov_r = float(torch.stack([reg[i]["sup_alpha"] for i in range(3)]).mean())
    assert cov_r > cov_f, f"kirpma on-plani artirmadi: {cov_r:.3f} vs {cov_f:.3f}"


def test_kirpma_kapaliyken_davranis_DEGISMEZ(tmp_path):
    """Geriye donuk uyumluluk: region=0 eski yolu birebir korumali."""
    tl, rroot = _setup(tmp_path)
    a = LRMDataset(tl, rroot, split="train", render_res=64, n_sup=4,
                   augment=False, deterministic=True, seed=7)
    b = LRMDataset(tl, rroot, split="train", render_res=64, n_sup=4,
                   augment=False, deterministic=True, seed=7, region=0)
    assert torch.equal(a[0]["sup_premult"], b[0]["sup_premult"])
    assert torch.equal(a[0]["sup_K"], b[0]["sup_K"])
