"""Checkpoint yaz-oku turu: kunye + render_cfg ile birlikte.

NEDEN (2026-08-27): PyTorch 2.6'dan beri torch.load varsayilani weights_only=True.
runstamp'in icindeki torch.torch_version.TorchVersion nesnesi bu moda takiliyor
=> KUNYELI bir checkpoint yazildiktan sonra --resume ve --init_from
UnpicklingError ile duser. 14 saatlik bir kosu yarida kesilseydi orada patlardi.
Ayni tuzak 17 cagri noktasindaydi.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import train_lrm as T
from lrm import runstamp


class _Kucuk(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)

    def forward(self, x):
        return self.l(x)


def _kurulum():
    m = _Kucuk()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    return m, opt, sch


def test_kunyeli_checkpoint_geri_okunuyor(tmp_path):
    """Asil regresyon: stamp iceren checkpoint torch.load'dan gecmeli."""
    m, opt, sch = _kurulum()
    p = str(tmp_path / "a.pt")
    stamp = runstamp.run_stamp({"a": 1})
    assert T.save_checkpoint(p, m, opt, sch, 100, stamp=stamp,
                             render_cfg={"density_bias": 0.0, "bound": 0.6})
    m2, opt2, sch2 = _kurulum()
    adim, _, _ = T.load_checkpoint(p, m2, opt2, sch2)
    assert adim == 100
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(a, b)


def test_render_cfg_checkpointte_tasiniyor(tmp_path):
    m, opt, sch = _kurulum()
    p = str(tmp_path / "b.pt")
    T.save_checkpoint(p, m, opt, sch, 10,
                      render_cfg={"density_bias": 1.0, "bound": 0.552})
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["render_cfg"] == {"density_bias": 1.0, "bound": 0.552}


def test_render_cfg_verilmezse_anahtar_yok(tmp_path):
    """Eski davranis korunmali; olmayan kunye 'bilinmiyor' demek."""
    m, opt, sch = _kurulum()
    p = str(tmp_path / "c.pt")
    T.save_checkpoint(p, m, opt, sch, 10)
    assert "render_cfg" not in torch.load(p, map_location="cpu", weights_only=False)


def test_kucuk_adim_buyugu_ezmiyor(tmp_path):
    """Blok 0'da yasandi: 40 adimlik duman testi 16.000 adimlik kaydi ezdi."""
    m, opt, sch = _kurulum()
    p = str(tmp_path / "d.pt")
    assert T.save_checkpoint(p, m, opt, sch, 16000)
    assert not T.save_checkpoint(p, m, opt, sch, 40)
    assert T.save_checkpoint(p, m, opt, sch, 40, force=True)


def test_kunye_alanlari_duz_tip():
    """Kok neden: torch.__version__ bir TorchVersion NESNESI. Kunyeye oldugu gibi
    konarsa checkpoint guvenli modda (weights_only=True) okunamaz."""
    s = runstamp.run_stamp({"a": 1})
    for k, v in s.items():
        assert isinstance(v, (str, bool, int, float, list, type(None))), \
            "kunye alani '%s' duz tip degil: %s" % (k, type(v).__name__)


def test_kunyeli_checkpoint_guvenli_modda_da_okunuyor(tmp_path):
    m, opt, sch = _kurulum()
    p = str(tmp_path / "e.pt")
    T.save_checkpoint(p, m, opt, sch, 5, stamp=runstamp.run_stamp({"a": 1}))
    ck = torch.load(p, map_location="cpu", weights_only=True)   # patlamamali
    assert ck["step"] == 5
