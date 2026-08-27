import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm import guards


def _healthy_preds(n=8, res=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, res, res, generator=g)


def test_sabit_cikti_MEAN_COLLAPSE_isaretlenir():
    """M_base/M_enc4 vakasi: tum objeler icin BIREBIR ayni cikti."""
    preds = torch.ones(8, 3, 16, 16)
    r = guards.collapse_flags(preds=preds)
    assert guards.MEAN_COLLAPSE in r["flags"]
    assert r["degenerate"] is True
    assert r["inter_std"] == pytest.approx(0.0, abs=1e-6)


def test_saglikli_cikti_temiz_gecer():
    r = guards.collapse_flags(preds=_healthy_preds(), acc_mean=0.12,
                              psnr_history=[15.1, 16.4, 17.0])
    assert r["flags"] == []
    assert r["degenerate"] is False


def test_bos_sahne_EMPTY_COLLAPSE_isaretlenir():
    r = guards.collapse_flags(preds=_healthy_preds(), acc_mean=0.001)
    assert guards.EMPTY_COLLAPSE in r["flags"]
    assert r["degenerate"] is True


def test_donmus_psnr_FROZEN_OUTPUT_isaretlenir():
    """M_base: 17.80 dB adim 500/1000/1500/1999'da birebir ayni."""
    r = guards.collapse_flags(preds=_healthy_preds(),
                              psnr_history=[17.80, 17.80, 17.80])
    assert guards.FROZEN_OUTPUT in r["flags"]


def test_psnr_gecmisi_kisaysa_donma_arastirilmaz():
    r = guards.collapse_flags(preds=_healthy_preds(), psnr_history=[17.8, 17.8])
    assert guards.FROZEN_OUTPUT not in r["flags"]


def test_psnr_degisiyorsa_donma_yok():
    r = guards.collapse_flags(preds=_healthy_preds(),
                              psnr_history=[17.80, 17.95, 18.20])
    assert guards.FROZEN_OUTPUT not in r["flags"]


def test_olculen_gercek_degerler_dogru_siniflanir():
    """dataset/lrm_bench/*.png uzerinden olculen inter-obje std degerleri.
    cokmus: 0.0000 / 0.0001   saglam: 0.0382 .. 0.0881"""
    for v in (0.0000, 0.0001):
        assert guards.is_degenerate_std(v) is True
    for v in (0.0382, 0.0452, 0.0520, 0.0881):
        assert guards.is_degenerate_std(v) is False


def test_numpy_girdi_de_kabul_edilir():
    np = pytest.importorskip("numpy")
    r = guards.collapse_flags(preds=np.ones((8, 3, 16, 16), dtype="float32"))
    assert guards.MEAN_COLLAPSE in r["flags"]


def test_esikler_disaridan_gecilebilir():
    preds = _healthy_preds()
    r = guards.collapse_flags(preds=preds, inter_std_min=1.0)
    assert guards.MEAN_COLLAPSE in r["flags"]


def test_ozet_metni_dejenerasyonu_gorunur_kilar():
    r = guards.collapse_flags(preds=torch.ones(8, 3, 8, 8), acc_mean=0.0)
    s = guards.format_flags(r)
    assert "DEJENERE" in s
    assert guards.MEAN_COLLAPSE in s
    r_ok = guards.collapse_flags(preds=_healthy_preds(), acc_mean=0.1)
    assert "DEJENERE" not in guards.format_flags(r_ok)


def test_preds_verilmezse_inter_std_none_kalir():
    r = guards.collapse_flags(acc_mean=0.2)
    assert r["inter_std"] is None
    assert r["degenerate"] is False


def test_tek_obje_ile_inter_std_hesaplanmaz():
    """N=1'de objeler arasi std tanimsiz; yanlislikla DEJENERE demesin."""
    r = guards.collapse_flags(preds=torch.rand(1, 3, 8, 8), acc_mean=0.2)
    assert r["inter_std"] is None
    assert r["degenerate"] is False
