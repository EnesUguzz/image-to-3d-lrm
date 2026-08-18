import torch

from lrm.triplane import TriplaneHead, sample_triplane


def test_head_upsamples():
    head = TriplaneHead(dim=64, out_channels=16, upsample=2)
    out = head(torch.randn(3, 8, 8, 64))
    assert out.shape == (3, 16, 16, 16)


def test_sample_shape():
    tri = torch.randn(3, 16, 32, 32)
    pts = torch.randn(100, 3) * 0.3
    feats = sample_triplane(tri, pts, bound=0.6)
    assert feats.shape == (100, 48)  # 3*16


def test_sample_center_matches_grid_center():
    # sabit dolu triplane => her nokta ayni ozellik
    tri = torch.ones(3, 4, 8, 8)
    feats = sample_triplane(tri, torch.zeros(5, 3), bound=0.6)
    assert torch.allclose(feats, torch.ones(5, 12), atol=1e-5)
