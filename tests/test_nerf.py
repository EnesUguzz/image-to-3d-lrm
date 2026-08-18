import torch

from lrm.nerf import TriplaneNeRF


def test_nerf_outputs_ranges():
    net = TriplaneNeRF(in_dim=48, hidden=32)
    d, c = net(torch.randn(50, 48))
    assert d.shape == (50, 1) and c.shape == (50, 3)
    assert (d >= 0).all()
    assert (c >= 0).all() and (c <= 1).all()
