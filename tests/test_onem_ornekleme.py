"""Hiyerarsik onem ornekleme: dogru mu, gercekten yuzeye mi yogunlasiyor?

Sadece "cokmuyor" testi yetmez -- onem ornekleme yanlis kurulunca da makul
goruntu uretir, sadece faydasi olmaz. Bu yuzden asil test, orneklerin
YOGUNLUGUN OLDUGU yere kaydigini dogrulamak.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm.renderer import (volume_render, volume_render_importance,
                          render_with_t, _pdf_ornekle)


def _ince_kabuk(merkez=1.5, kalinlik=0.02):
    """t=merkez'de ince opak kabuk: tekduze ornekleme bunu kacirir."""
    def q(pts):
        r = pts.norm(dim=-1, keepdim=True)
        d = torch.where((r - merkez).abs() < kalinlik,
                        torch.full_like(r, 500.0), torch.zeros_like(r))
        return d, torch.full((pts.shape[0], 3), 0.5, device=pts.device)
    return q


def _isinlar(n=32):
    o = torch.zeros(n, 3)
    d = torch.zeros(n, 3); d[:, 2] = 1.0
    return o, d


def test_onem_ornekleme_ince_kabugu_tekduzeden_iyi_yakalar():
    o, d = _isinlar()
    q = _ince_kabuk()
    torch.manual_seed(0)
    _, acc_u = volume_render(o, d, 0.8, 2.2, 32, q, jitter=False)
    torch.manual_seed(0)
    _, acc_i = volume_render_importance(o, d, 0.8, 2.2, 32, 32, q, jitter=False)
    # ayni kaba butce + ince gecis => kabuk daha iyi yakalanmali
    assert acc_i.mean() >= acc_u.mean(), (float(acc_u.mean()), float(acc_i.mean()))


def test_pdf_ornekleri_agirlikli_bolgeye_dusuyor():
    R, S = 8, 16
    t = torch.linspace(0.8, 2.2, S).expand(R, S).clone()
    w = torch.zeros(R, S); w[:, 8] = 1.0          # tum agirlik tek yerde
    hedef = float(t[0, 8])
    orn = _pdf_ornekle(t, w, 64, det=True)
    assert (orn - hedef).abs().mean() < 0.15, float((orn - hedef).abs().mean())


def test_render_with_t_tekduze_t_ile_volume_render_ile_ayni():
    o, d = _isinlar(16)
    q = _ince_kabuk(kalinlik=0.5)
    t = torch.linspace(0.8, 2.2, 24).expand(16, 24).clone()
    a = volume_render(o, d, 0.8, 2.2, 24, q, jitter=False)
    b = render_with_t(o, d, t, q)
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert torch.allclose(a[1], b[1], atol=1e-6)


def test_gradyan_akiyor():
    o, d = _isinlar(16)
    w = torch.nn.Parameter(torch.randn(3, 4) * 0.1)

    def q(pts):
        h = pts @ w
        return torch.nn.functional.softplus(h[:, :1]), torch.sigmoid(h[:, 1:4])
    r, _ = volume_render_importance(o, d, 0.8, 2.2, 16, 16, q, jitter=False)
    r.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert w.grad.abs().sum() > 0
