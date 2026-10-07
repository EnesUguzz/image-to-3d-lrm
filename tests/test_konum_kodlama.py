"""Konum kodlamasi: kapaliyken davranis DEGISMEMELI, acikken alt-hucre
yapisi uretebilmeli.

Asil risk sessiz calismak: pts verilmezse kodlama devre disi kalip kosu
"bayrak acik" sanilarak raporlanirdi. O yuzden hata firlatmasi test ediliyor.
"""
import os, sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm.nerf import TriplaneNeRF


def test_kapaliyken_pts_gerekmiyor_ve_boyut_ayni():
    n = TriplaneNeRF(in_dim=96, hidden=32, layers=2)
    d, c = n(torch.randn(10, 96))
    assert d.shape == (10, 1) and c.shape == (10, 3)
    assert n.backbone[0].in_features == 96


def test_acikken_girdi_boyutu_buyuyor():
    n = TriplaneNeRF(in_dim=96, hidden=32, layers=2, pos_enc=6)
    assert n.backbone[0].in_features == 96 + 3 * 2 * 6


def test_acikken_pts_yoksa_HATA_verir_sessizce_gecmez():
    n = TriplaneNeRF(in_dim=96, hidden=32, layers=2, pos_enc=6)
    with pytest.raises(ValueError, match="pts"):
        n(torch.randn(4, 96))


def test_ayni_ozellik_farkli_konum_farkli_cikti():
    """Alt-hucre yapinin TEK kaynagi bu: ayni interpolasyonlu ozellik,
    farkli konum => farkli yogunluk/renk uretebilmeli."""
    torch.manual_seed(0)
    n = TriplaneNeRF(in_dim=8, hidden=32, layers=2, pos_enc=6).eval()
    f = torch.randn(1, 8).expand(2, 8)
    p = torch.tensor([[0.10, 0.0, 0.0], [0.11, 0.0, 0.0]])
    with torch.no_grad():
        d, c = n(f, p)
    assert not torch.allclose(d[0], d[1], atol=1e-7)


def test_kapaliyken_ayni_ozellik_farkli_konum_AYNI_cikti():
    """Mevcut mimarinin sinirinin kanidi: kodlama yokken konum hic etkisiz."""
    torch.manual_seed(0)
    n = TriplaneNeRF(in_dim=8, hidden=32, layers=2).eval()
    f = torch.randn(1, 8).expand(2, 8)
    with torch.no_grad():
        d, _ = n(f)
    assert torch.allclose(d[0], d[1])
