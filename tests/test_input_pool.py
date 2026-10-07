"""Girdi gorunum havuzu -- S1 duzeltmesi (2026-08-29 alan taramasi).

Eski davranis: girdi DAIMA 4 kanonikten, hepsi elevation +20 => model baska
hicbir girdi acisi gormuyordu. Taranan hicbir referans (LRM, OpenLRM,
InstantMesh, Hunyuan3D, TRELLIS) girdi elevation'ini sabitlemiyor.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm.dataset import LRMDataset

LISTE = "dataset/train_list_v2.json"
RENDERS = "dataset/renders_opp_score3"
VAR = os.path.isfile(LISTE) and os.path.isdir(RENDERS)
import pytest
pytestmark = pytest.mark.skipif(not VAR, reason="veri seti yok")


def _elevations(ds, n=40):
    """Cekilen girdi gorunumlerinin elevation'lari."""
    out = []
    for i in range(n):
        it = ds[i % len(ds)]
        for c in it["input_c2w"]:
            pos = c[:3, 3].numpy()
            r = np.linalg.norm(pos)
            out.append(np.degrees(np.arcsin(pos[2] / max(r, 1e-9))))
    return np.array(out)


def test_canon_havuzu_daima_20_derece():
    ds = LRMDataset(LISTE, RENDERS, split="train", input_pool="canon", augment=False)
    el = _elevations(ds)
    assert np.allclose(el, 20.0, atol=0.5), f"kanonik havuz +20 disina cikti: {el[:5]}"


def test_all_havuzu_cesitlilik_getiriyor():
    ds = LRMDataset(LISTE, RENDERS, split="train", input_pool="all", augment=False)
    el = _elevations(ds)
    assert el.std() > 5.0, f"cesitlilik yok (std={el.std():.2f})"
    assert (np.abs(el - 20.0) > 5).mean() > 0.4, "cogu hala +20 civarinda"


def test_mixed_ikisinin_arasinda():
    ds = LRMDataset(LISTE, RENDERS, split="train", input_pool="mixed",
                    mixed_p=0.5, augment=False)
    el = _elevations(ds, n=80)
    yirmi = np.isclose(el, 20.0, atol=0.5).mean()
    assert 0.15 < yirmi < 0.95, f"kanonik orani beklenen bantta degil: {yirmi:.2f}"


def test_gecersiz_havuz_reddediliyor():
    import pytest as _p
    with _p.raises(AssertionError):
        LRMDataset(LISTE, RENDERS, split="train", input_pool="hatali")
