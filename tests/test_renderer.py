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
