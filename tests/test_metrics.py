"""lrm/metrics.py regresyon testleri.

Bu metrikler artik HEM kapi kararlarini HEM egitim ici izlemeyi besliyor.
Sessiz bir tanim hatasi (yanlis eksende ortalama, p10/p90 karisikligi, SSIM
penceresi) tum kapi kararlarini gecersiz kilar -- o yuzden test ediliyorlar.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from lrm import metrics as M


def _imgs(n=6, c=3, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, c, h, w, generator=g)


# ----------------------------------------------------------------- PSNR
def test_psnr_ayni_goruntude_yuksek():
    P = _imgs()
    psnr, mse = M.psnr_per_object(P, P.clone())
    assert torch.all(psnr > 80), "ayni goruntu icin PSNR tavana yakin olmali"
    assert torch.allclose(mse, torch.zeros_like(mse))


def test_psnr_obje_basina_dondurur():
    P, G = _imgs(5), _imgs(5, seed=1)
    psnr, _ = M.psnr_per_object(P, G)
    assert psnr.shape == (5,)


def test_psnr_bilinen_deger():
    # sabit fark 0.1 => mse = 0.01 => PSNR = 20 dB
    G = torch.full((2, 3, 8, 8), 0.5)
    P = G + 0.1
    psnr, _ = M.psnr_per_object(P, G)
    assert torch.allclose(psnr, torch.full((2,), 20.0), atol=1e-4)


# ----------------------------------------------------------------- SSIM
def test_ssim_ayni_goruntude_bir():
    P = _imgs()
    s = M.ssim_per_object(P, P.clone())
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)


def test_ssim_skimage_ile_uyumlu():
    """Kendi torch SSIM'imiz referans uygulamayla ayni sayiyi vermeli."""
    skimage = pytest.importorskip("skimage.metrics")
    P, G = _imgs(3, h=64, w=64), _imgs(3, h=64, w=64, seed=9)
    ours = M.ssim_per_object(P, G).numpy()
    ref = np.array([
        skimage.structural_similarity(
            G[i].permute(1, 2, 0).numpy(), P[i].permute(1, 2, 0).numpy(),
            channel_axis=2, data_range=1.0, gaussian_weights=True,
            sigma=1.5, use_sample_covariance=False)
        for i in range(len(P))])
    assert np.allclose(ours, ref, atol=0.02), f"ours={ours} ref={ref}"


def test_ssim_bozulmayla_dusuyor():
    G = _imgs(4)
    az = G + 0.02 * torch.randn_like(G)
    cok = G + 0.30 * torch.randn_like(G)
    assert M.ssim_per_object(az, G).mean() > M.ssim_per_object(cok, G).mean()


# ----------------------------------------------------------------- top-1
def test_top1_mukemmel_eslesme():
    G = _imgs(8)
    assert M.top1_retrieval(G.clone(), G) == pytest.approx(1.0)


def test_top1_kaydirilmis_eslesme_sifir():
    """Her tahmin BASKA objenin GT'sine esitse top-1 = 0."""
    G = _imgs(8)
    P = torch.roll(G, 1, dims=0)
    assert M.top1_retrieval(P, G) == pytest.approx(0.0)


def test_top1_ayni_ciktida_sansa_yakin():
    """Tum tahminler ayni (blob cokusu) => top-1 sansa duser."""
    G = _imgs(16)
    P = G.mean(0, keepdim=True).expand_as(G).contiguous()
    assert M.top1_retrieval(P, G) <= 1.0 / 16 + 1e-6


# ----------------------------------------------------------------- oran
def test_ortalama_baseline_orani_cokuste_bir():
    """Model ortalamayi basiyorsa oran tam 1.0 olmali (cokus imzasi)."""
    G = _imgs(10)
    P = G.mean(0, keepdim=True).expand_as(G).contiguous()
    assert M.mean_baseline_ratio(P, G) == pytest.approx(1.0, abs=1e-5)


def test_ortalama_baseline_orani_mukemmelde_sifir():
    G = _imgs(10)
    assert M.mean_baseline_ratio(G.clone(), G) < 1e-6


# ----------------------------------------------------------------- komsu
def test_komsu_kendini_secmez():
    X = _imgs(12)
    j = M.neighbor_indices(X)
    assert not torch.any(j == torch.arange(12)), "komsu baseline kendini secemez"


def test_komsu_gercekten_en_yakini_buluyor():
    """Obje 0 ile obje 3 neredeyse ayni => birbirlerini secmeliler."""
    X = _imgs(6)
    X[3] = X[0] + 1e-4
    j = M.neighbor_indices(X)
    assert int(j[0]) == 3 and int(j[3]) == 0


# ----------------------------------------------------------------- IoU
def test_maske_iou_tam_ortusme():
    A = (torch.rand(5, 1, 16, 16) > 0.5).float()
    assert torch.allclose(M.mask_iou_per_object(A, A), torch.ones(5))


def test_maske_iou_ayrik_maskeler_sifir():
    A = torch.zeros(2, 1, 8, 8); A[:, :, :4] = 1.0
    B = torch.zeros(2, 1, 8, 8); B[:, :, 4:] = 1.0
    assert torch.allclose(M.mask_iou_per_object(A, B), torch.zeros(2))


# ----------------------------------------------------------------- ozet
def test_ozet_yuksek_iyi_metrikte_p10():
    s = M.summarize({"psnr": torch.arange(100.0)})
    assert "p10" in s["psnr"] and "p90" not in s["psnr"]
    assert s["psnr"]["p10"] == pytest.approx(9.9, abs=0.5)


def test_ozet_dusuk_iyi_metrikte_p90():
    """LPIPS'te kotu uc YUKSEK taraftadir => p90 raporlanmali."""
    s = M.summarize({"lpips": torch.arange(100.0)})
    assert "p90" in s["lpips"] and "p10" not in s["lpips"]


def test_ozet_bos_ve_none_dayanikli():
    s = M.summarize({"a": None, "b": torch.tensor([])})
    assert s == {}


# ----------------------------------------------------------------- katman 1
def test_image_metrics_obje_basina_uzunluk():
    P, G = _imgs(7), _imgs(7, seed=2)
    out = M.image_metrics(P, G, use_lpips=False, use_clip=False)
    for k, v in out.items():
        assert v.shape == (7,), f"{k} obje basina olmali, {v.shape} geldi"


def test_image_metrics_cok_gorunum_katlaniyor():
    """(N,V,3,H,W) girdide sonuc yine obje basina (N,) olmali."""
    P = torch.rand(4, 3, 3, 16, 16)
    G = torch.rand(4, 3, 3, 16, 16)
    out = M.image_metrics(P, G, use_lpips=False, use_clip=False)
    for k, v in out.items():
        assert v.shape == (4,), f"{k}: cok gorunum obje eksenine katlanmali"


def test_image_metrics_maske_verilince_iou_var():
    P, G = _imgs(3), _imgs(3, seed=4)
    A = torch.rand(3, 1, 32, 32)
    out = M.image_metrics(P, G, A, A.clone(), use_lpips=False)
    assert "iou" in out and torch.allclose(out["iou"], torch.ones(3))
