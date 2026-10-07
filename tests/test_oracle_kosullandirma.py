"""ORACLE KOSULLANDIRMA yolu -- `make_triplane(tok=...)` enjeksiyonu.

NEDEN VAR (2026-09-02):
Ogrenci ogretmenin triplane'ini uretemiyor (`rel` ~0.26-0.30'da plato). Sebep
iki adaydan hangisi, bugun BILINMIYOR:
  H1 koşullandırma : girdi yolu yeterli bilgi tasimiyor (donuk DINOv2 @224,
                     objede 8.6 patch)
  H2 kapasite      : bilgi tam olsa bile transformer+head o hedefi uretemiyor
Planin yarisi H1'e (girdi cozunurlugu), yarisi H2'ye (token izgarasi) yatirim
yapiyor => biri kesinlikle bosa gidecek.

Oracle kolu encoder ciktisini obje basina SERBEST ogrenilebilir token'larla
degistirir: "bu token butcesindeki HERHANGI bir encoder'in verebilecegi en iyi
sinyal". Kritik nokta, enjeksiyonun encoder'i GERCEKTEN devre disi birakmasi;
sessizce encoder'i kullanmaya devam eden bir kol, "oracle da duzelmedi" diye
okunur ve H1 haksiz yere elenir.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOK, "scripts"))

from lrm.model import LRM   # noqa: E402


class SahteEncoder(nn.Module):
    """DINOv2 indirmeden LRM kurmak icin. Cagrildigini SAYAR."""

    def __init__(self, embed_dim=384, patch=14, n_patch=256):
        super().__init__()
        self.embed_dim, self.patch, self.n_patch = embed_dim, patch, n_patch
        self.cagri = 0

    def forward(self, imgs):
        self.cagri += 1
        return torch.zeros(imgs.shape[0], self.n_patch, self.embed_dim)


def _kur(dim=64, depth=1, heads=2, tp_res=8, tp_ch=4):
    enc = SahteEncoder()
    m = LRM(dim=dim, depth=depth, heads=heads, triplane_res=tp_res,
            triplane_ch=tp_ch, encoder=enc)
    return m.eval(), enc


def _girdi(v=1, res=None):
    res = LRM.INPUT_RES if res is None else res
    imgs = torch.zeros(v, 3, res, res)
    c2w = torch.eye(4).repeat(v, 1, 1)
    c2w[:, 2, 3] = 1.5
    K = torch.tensor([[100.0, 0, res / 2], [0, 100.0, res / 2], [0, 0, 1]]).repeat(v, 1, 1)
    return imgs, c2w, K


def test_tok_verilince_encoder_HIC_cagrilmaz():
    """Oracle kolunun tek isi encoder'i devreden cikarmak. Cagrilirsa deney
    anlamsizlasir ve bunu hicbir sayi ele vermez."""
    m, enc = _kur()
    imgs, c2w, K = _girdi()
    with torch.no_grad():
        m.make_triplane(imgs, c2w, K)
    assert enc.cagri == 1, "normal yolda encoder cagrilmali"
    with torch.no_grad():
        m.make_triplane(imgs, c2w, K, tok=torch.randn(1, 256, 384))
    assert enc.cagri == 1, "tok verildiginde encoder BIR DAHA cagrilmamali"


def test_varsayilan_davranis_degismedi():
    """tok=None normal yol -- oracle eklentisi mevcut kosulari etkilememeli."""
    m, _ = _kur()
    imgs, c2w, K = _girdi()
    torch.manual_seed(0)
    with torch.no_grad():
        a = m.make_triplane(imgs, c2w, K)
        b = m.make_triplane(imgs, c2w, K)
    assert torch.allclose(a, b)
    assert a.shape == (3, 4, 16, 16)      # tp_ch 4, tp_res 8 -> deconv x2


@pytest.mark.parametrize("n_tok", [256, 1024])
def test_serbest_token_sayisi_degisebilir(n_tok):
    """Token butcesini buyutmek (256 -> 1024) 'darbogaz token sayisi mi' sorusunu
    ayirir. Plucker izgarasi sqrt(N) oldugu icin N tam kare olmali."""
    m, _ = _kur()
    imgs, c2w, K = _girdi()
    with torch.no_grad():
        tp = m.make_triplane(imgs, c2w, K, tok=torch.randn(1, n_tok, 384))
    assert tp.shape == (3, 4, 16, 16)


def test_gradyan_serbest_tokenlara_akiyor():
    """Serbest token'lar ogrenilebilir olmali; akmiyorsa kol sabit gurultu olur."""
    m, _ = _kur()
    imgs, c2w, K = _girdi()
    tok = torch.randn(1, 256, 384, requires_grad=True)
    m.make_triplane(imgs, c2w, K, tok=tok).sum().backward()
    assert tok.grad is not None and float(tok.grad.abs().sum()) > 0


def test_tam_kare_olmayan_token_sayisi_yakalaniyor():
    """distill_lrm assert'i: sqrt(N) tam sayi degilse Plucker izgarasi sessizce
    yanlis kurulur (side = int(P**0.5) asagi yuvarlar)."""
    import ast
    with open(os.path.join(KOK, "scripts", "distill_lrm.py"), encoding="utf-8") as f:
        src = f.read()
    assert "tam kare degil" in src, "--oracle icin tam-kare assert'i kaybolmus"
    ast.parse(src)
