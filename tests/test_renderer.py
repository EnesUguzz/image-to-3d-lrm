import torch

from lrm.renderer import volume_render


def test_empty_scene_zero_alpha():
    def q(pts):
        n = pts.shape[0]
        return torch.zeros(n, 1), torch.zeros(n, 3)
    o = torch.zeros(4, 3)
    d = torch.tensor([[0., 0., -1.]]).expand(4, 3).contiguous()
    rgb, acc = volume_render(o, d, 0.5, 2.0, 16, q)
    assert rgb.shape == (4, 3) and acc.shape == (4, 1)
    assert acc.max() < 1e-3


def test_partial_density_not_saturated():
    # DUSUK sabit yogunluk => acc kismi olmali (~0.5), 1'e SATURE OLMAMALI.
    # (son delta=1e10 hatasi burada acc'yi zorla 1 yapardi => seffaf arka plan ogrenilemez)
    def q(pts):
        n = pts.shape[0]
        return torch.full((n, 1), 0.5), torch.full((n, 3), 0.5)
    o = torch.zeros(2, 3)
    d = torch.tensor([[0., 0., -1.]]).expand(2, 3).contiguous()
    rgb, acc = volume_render(o, d, 0.8, 2.2, 64, q, jitter=False)
    assert 0.3 < acc.min() and acc.max() < 0.7  # kismi opaklik, sature degil


def test_bg_color_composites_empty_scene():
    # bos sahne (yogunluk 0) => render TAM OLARAK bg_color olmali.
    # (random-bg egitim fix'i: model bg rengini dogru kompozitlemeli)
    def q(pts):
        n = pts.shape[0]
        return torch.zeros(n, 1), torch.zeros(n, 3)
    o = torch.zeros(2, 3)
    d = torch.tensor([[0., 0., -1.]]).expand(2, 3).contiguous()
    c = torch.tensor([0.2, 0.7, 0.4])
    rgb, acc = volume_render(o, d, 0.5, 2.0, 16, q, bg_color=c)
    assert torch.allclose(rgb, c.expand(2, 3), atol=1e-3)


def test_bg_color_changes_output_partial_scene():
    # yari-saydam sahne: farkli bg rengi => farkli render (sabit cikti olamaz).
    def q(pts):
        n = pts.shape[0]
        return torch.full((n, 1), 0.5), torch.full((n, 3), 0.5)
    o = torch.zeros(2, 3)
    d = torch.tensor([[0., 0., -1.]]).expand(2, 3).contiguous()
    r1, _ = volume_render(o, d, 0.8, 2.2, 64, q, jitter=False,
                          bg_color=torch.tensor([1., 0., 0.]))
    r2, _ = volume_render(o, d, 0.8, 2.2, 64, q, jitter=False,
                          bg_color=torch.tensor([0., 0., 1.]))
    assert (r1 - r2).abs().max() > 0.1  # bg farki cikisa yansimali


def test_dense_wall_high_alpha():
    # her yerde cok yuksek yogunluk, kirmizi => acc ~1, rgb ~ kirmizi
    def q(pts):
        n = pts.shape[0]
        dens = torch.full((n, 1), 1e3)
        col = torch.tensor([1., 0., 0.]).expand(n, 3)
        return dens, col
    o = torch.zeros(3, 3)
    d = torch.tensor([[0., 0., -1.]]).expand(3, 3).contiguous()
    rgb, acc = volume_render(o, d, 0.5, 2.0, 32, q)
    assert acc.min() > 0.9
    assert rgb[:, 0].min() > 0.8 and rgb[:, 1].max() < 0.2
