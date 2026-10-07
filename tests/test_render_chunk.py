"""volume_render_chunked, volume_render ile AYNI sonucu ve AYNI gradyani vermeli.

Neden test: checkpoint yeniden hesap yaparken jitter'i farkli ornekleyebilir
(volume_render'in jitter varsayilani torch.is_grad_enabled()'a bagli). O
durumda kayip dogru gorunur ama gradyan sessizce yanlis olur -- egitimi
bozan, hicbir assert'in yakalamayacagi bir hata sinifi.
"""
import os, sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm.renderer import volume_render, volume_render_chunked


def _kur(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.zeros(n, 3)
    d = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    w = torch.nn.Parameter(torch.randn(3, 8, generator=g) * 0.1)

    def q(pts):
        h = pts @ w                      # (N,8)
        return torch.nn.functional.softplus(h[:, :1]), torch.sigmoid(h[:, 1:4])
    return o, d, w, q


def test_chunked_ileri_gecis_ayni():
    o, d, w, q = _kur()
    with torch.no_grad():
        a = volume_render(o, d, 0.8, 2.2, 32, q, jitter=False)
        b = volume_render_chunked(o, d, 0.8, 2.2, 32, q, chunk=16, jitter=False)
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert torch.allclose(a[1], b[1], atol=1e-6)


def test_chunked_gradyan_ayni():
    o, d, w, q = _kur()
    r, _ = volume_render(o, d, 0.8, 2.2, 32, q, jitter=False)
    r.sum().backward()
    g1 = w.grad.clone(); w.grad = None
    r2, _ = volume_render_chunked(o, d, 0.8, 2.2, 32, q, chunk=16, jitter=False)
    r2.sum().backward()
    assert torch.allclose(g1, w.grad, atol=1e-6), (g1 - w.grad).abs().max()


def test_chunk_0_veya_buyuk_ise_bolmez():
    o, d, w, q = _kur(n=8)
    with torch.no_grad():
        a = volume_render(o, d, 0.8, 2.2, 16, q, jitter=False)
        for c in (0, 8, 999):
            b = volume_render_chunked(o, d, 0.8, 2.2, 16, q, chunk=c, jitter=False)
            assert torch.allclose(a[0], b[0], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda yok")
def test_chunked_jitterli_gradyan_ayni_cuda():
    """jitter ACIK iken: checkpoint RNG'yi korumazsa gradyan bozulur."""
    torch.manual_seed(0)
    o, d, w, q = _kur(n=128)
    o, d, w = o.cuda(), d.cuda(), torch.nn.Parameter(w.detach().cuda())

    def q2(pts):
        h = pts @ w
        return torch.nn.functional.softplus(h[:, :1]), torch.sigmoid(h[:, 1:4])
    torch.manual_seed(7)
    r, _ = volume_render_chunked(o, d, 0.8, 2.2, 32, q2, chunk=32, jitter=True)
    r.sum().backward()
    g = w.grad.clone()
    assert torch.isfinite(g).all() and g.abs().sum() > 0
