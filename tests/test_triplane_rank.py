"""TriplaneHead rutbe darbogazi ve NeRF derinligi -- referans sirasina uyum.

2026-08-29 bagimsiz denetimi olctu: eski sira (Linear(dim->C) sonra ortusmesiz
ConvTranspose2d(C->C, k=2, s=2)) 2x2 blogun 4*C sayisini C boyutlu bir alt
uzaya hapsediyordu. Bu testler o darbogazin geri gelmesini engeller.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm.triplane import TriplaneHead
from lrm.nerf import TriplaneNeRF


def test_head_linear_darbogazi_yok():
    h = TriplaneHead(dim=512, out_channels=32)
    assert not hasattr(h, "proj"), "Linear darbogazi geri geldi"
    assert h.up.in_channels == 512, "deconv dogrudan token boyutunu almali"
    assert h.up.out_channels == 32


def test_head_cikti_sekli_korunuyor():
    h = TriplaneHead(dim=512, out_channels=32, upsample=2)
    out = h(torch.randn(3, 32, 32, 512))          # (3, r, r, dim)
    assert out.shape == (3, 32, 64, 64)


def test_head_2x2_blok_rutbesi_C_ile_sinirli_degil():
    """Ayni hucreden uretilen 2x2 blogun 4 vektoru dogrusal bagimsiz olabilmeli.

    Eski sirada bu 4 vektorun HEPSI ayni C boyutlu izdusumun afin
    fonksiyonuydu; simdi her biri dim boyutlu token'in AYRI izdusumu.
    """
    h = TriplaneHead(dim=512, out_channels=32, upsample=2)
    W = h.up.weight                                # (dim, C, 2, 2)
    dorт = torch.stack([W[:, :, i, j].reshape(-1)
                        for i in range(2) for j in range(2)])
    assert torch.linalg.matrix_rank(dorт.float()) == 4, \
        "2x2 blogun 4 izdusumu bagimsiz olmali"


def test_nerf_derinligi_referansla_ayni():
    n = TriplaneNeRF(in_dim=96)
    lineer = [m for m in n.backbone if isinstance(m, torch.nn.Linear)]
    assert len(lineer) == 4, "referans (OpenLRM) 4 katman kullaniyor"


def test_nerf_derinligi_parametrik():
    assert len([m for m in TriplaneNeRF(in_dim=96, layers=2).backbone
                if isinstance(m, torch.nn.Linear)]) == 2


def test_head_baslangic_olcegi_ogretmen_dagilimina_yakin():
    """Baslangic cikti std'si ogretmen triplane dagilimina (0.2416) yakin olmali.

    Varsayilan ConvTranspose2d init'i fan_in'i out_channels'tan hesapliyor;
    duzeltilmezse cikti std 1.16 (6.5x buyuk) -> NeRF'te softplus doygunlugu ->
    acc->0 cokusu. Bu cokus projede UC KEZ yasandi.
    """
    torch.manual_seed(0)
    h = TriplaneHead(dim=512, out_channels=32)
    out = h(torch.randn(3, 32, 32, 512))
    assert 0.12 < float(out.std()) < 0.45, \
        f"baslangic olcegi ogretmen bandinin disinda: {float(out.std()):.4f}"


def test_head_bias_sifir():
    h = TriplaneHead(dim=512, out_channels=32)
    assert torch.allclose(h.up.bias, torch.zeros_like(h.up.bias))
