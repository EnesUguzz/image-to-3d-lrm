import json
import numpy as np
from PIL import Image
import filter_dataset as fd


def _obj(tmp_path, uid, coverage_pixels, size=64, border=False):
    """Belirli sayıda opak piksel içeren 4 kanonik + 12 dolgu png üretir."""
    d = tmp_path / uid
    d.mkdir()
    for i in range(16):
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        if border:
            arr[:, :, 3] = 255                      # tüm kare opak (kenara değer)
        else:
            side = int(coverage_pixels ** 0.5)
            arr[10:10 + side, 10:10 + side, 3] = 255  # ortada blok
        Image.fromarray(arr, "RGBA").save(d / f"{i:03d}.png")
    json.dump({"uid": uid, "num_views": 16}, open(d / "meta.json", "w"))
    return str(tmp_path)


def test_alpha_coverage_fraction(tmp_path):
    root = _obj(tmp_path, "u1", coverage_pixels=400, size=64)   # 20x20 blok = 400/4096
    cov = fd.alpha_coverage(f"{root}/u1/000.png")
    assert abs(cov - 400 / 4096) < 0.01


def test_touches_border_true_when_full(tmp_path):
    root = _obj(tmp_path, "ub", coverage_pixels=0, size=64, border=True)
    assert fd.touches_border(f"{root}/ub/000.png") is True


def test_score_object_low_coverage_rejected(tmp_path):
    root = _obj(tmp_path, "tiny", coverage_pixels=16, size=64)  # ~0.4% doluluk
    passed, reason = fd.passes(root, "tiny", min_cov=0.010, max_cov=0.90)
    assert passed is False and "coverage" in reason


def test_score_object_good_passes(tmp_path):
    root = _obj(tmp_path, "good", coverage_pixels=1600, size=64)  # ~39% doluluk
    passed, reason = fd.passes(root, "good", min_cov=0.010, max_cov=0.90)
    assert passed is True and reason == "ok"


def test_thin_object_passes_if_big_from_some_view(tmp_path):
    """İnce obje: kanonik açılarda az, bazı açılarda çok dolu → GEÇMELİ
    (16 açının maksimumuna bakılır, sadece kanoniklere değil)."""
    d = tmp_path / "thin"
    d.mkdir()
    for i in range(16):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        px = 1600 if i >= 8 else 25          # yarısı büyük, yarısı ince
        side = int(px ** 0.5)
        arr[10:10 + side, 10:10 + side, 3] = 255
        Image.fromarray(arr, "RGBA").save(d / f"{i:03d}.png")
    import json as _j
    _j.dump({"uid": "thin", "num_views": 16}, open(d / "meta.json", "w"))
    passed, reason = fd.passes(str(tmp_path), "thin", min_cov=0.010, max_cov=0.90)
    assert passed is True and reason == "ok"


def test_filter_dataset_splits(tmp_path):
    root = _obj(tmp_path, "good", coverage_pixels=1600, size=64)
    _obj(tmp_path, "tiny", coverage_pixels=16, size=64)
    out = tmp_path / "train_list.json"
    kept, rejected = fd.filter_dataset(root, ["good", "tiny"], str(out),
                                       min_cov=0.010, max_cov=0.90, val_frac=0.0)
    assert kept == ["good"] and "tiny" in dict(rejected)
    data = json.load(open(out))
    assert data["train"] == ["good"]
