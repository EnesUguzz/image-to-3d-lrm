"""mask_fg_weight regresyon testleri.

NEDEN (2026-08-27): mask kaybi RGB ile ayni fg_weight=5.0'i kullaniyordu.
fg_weight TAM KARE denetimi icin eklenmisti; bolge kirpmasiyla (fg_bias=0.75)
arka plan zaten azinlikta oldugundan ayni duzeltme cift sayilip 'arka plan bos
olsun' sinyalini ~%14'e dusuruyordu. Kol A 6000 adim boyunca bound kupunu
doldurup oturdu (acc~0.50, PSNR 8.8 dB, taban 17.36 dB).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from lrm.losses import LRMLoss


def _sahne(fg_orani=0.5, h=16, w=16):
    """Yarisi obje olan bir hedef + TAMAMEN DOLU bir tahmin (kol A'nin hatasi)."""
    gt_alpha = torch.zeros(1, 1, h, w)
    gt_alpha[..., : int(h * fg_orani), :] = 1.0
    gt_rgb = torch.full((1, 3, h, w), 0.5)
    pred_acc = torch.ones(1, 1, h, w)          # her yer dolu
    pred_rgb = torch.full((1, 3, h, w), 0.5)
    return pred_rgb, pred_acc, gt_rgb, gt_alpha


def test_varsayilan_eski_davranisla_ayni():
    """mask_fg_weight verilmezse davranis BIT-BAZINDA degismemeli."""
    a = LRMLoss(use_lpips=False)
    b = LRMLoss(use_lpips=False, mask_fg_weight=5.0)
    args = _sahne()
    assert a(*args)[1]["mask"].item() == pytest.approx(b(*args)[1]["mask"].item())


def test_tekduze_mask_dolu_kupu_daha_cok_cezalandirir():
    """Asil iddia: fg agirligi kalkinca 'her yer dolu' hatasi buyur."""
    agirlikli = LRMLoss(use_lpips=False, mask_fg_weight=5.0)
    tekduze = LRMLoss(use_lpips=False, mask_fg_weight=0.0)
    args = _sahne()
    m_ag = agirlikli(*args)[1]["mask"].item()
    m_td = tekduze(*args)[1]["mask"].item()
    assert m_td > m_ag, f"tekduze {m_td} <= agirlikli {m_ag}"
    # olculen oran ~3.5x (belgede iddia edilen sayi)
    assert m_td / m_ag == pytest.approx(3.5, rel=0.15)


def test_mukemmel_tahminde_mask_sifir_her_agirlikta():
    for mfw in (0.0, 1.0, 5.0):
        L = LRMLoss(use_lpips=False, mask_fg_weight=mfw)
        gt_alpha = (torch.rand(1, 1, 8, 8) > 0.5).float()
        gt_rgb = torch.rand(1, 3, 8, 8)
        assert L(gt_rgb, gt_alpha, gt_rgb, gt_alpha)[1]["mask"].item() == pytest.approx(0.0)


def test_rgb_agirligi_etkilenmiyor():
    """mask_fg_weight sadece mask'i degistirmeli, mse'ye dokunmamali."""
    args = (torch.rand(1, 3, 8, 8), torch.rand(1, 1, 8, 8),
            torch.rand(1, 3, 8, 8), (torch.rand(1, 1, 8, 8) > 0.5).float())
    m0 = LRMLoss(use_lpips=False, mask_fg_weight=0.0)(*args)[1]["mse"].item()
    m5 = LRMLoss(use_lpips=False, mask_fg_weight=5.0)(*args)[1]["mse"].item()
    assert m0 == pytest.approx(m5)


def test_bos_tahmin_hala_cezalaniyor():
    """Ters cokus (acc->0) tekduze agirlikta da cezasiz kalmamali."""
    L = LRMLoss(use_lpips=False, mask_fg_weight=0.0)
    _, _, gt_rgb, gt_alpha = _sahne()
    bos = torch.zeros(1, 1, 16, 16)
    assert L(gt_rgb, bos, gt_rgb, gt_alpha)[1]["mask"].item() > 0.4
