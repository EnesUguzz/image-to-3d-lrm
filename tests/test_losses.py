import torch

from lrm.losses import LRMLoss


def test_zero_loss_on_identical():
    loss = LRMLoss(use_lpips=False)
    rgb = torch.rand(2, 3, 16, 16)
    alpha = torch.rand(2, 1, 16, 16)
    total, parts = loss(rgb, alpha, rgb, alpha)
    assert total.item() < 1e-6
    assert "mse" in parts and "mask" in parts


def test_mask_loss_increases_with_silhouette_diff():
    loss = LRMLoss(use_lpips=False, w_mse=0.0, w_mask=1.0)
    rgb = torch.zeros(1, 3, 8, 8)
    a1 = torch.zeros(1, 1, 8, 8)
    a2 = torch.ones(1, 1, 8, 8)
    same, _ = loss(rgb, a1, rgb, a1)
    diff, _ = loss(rgb, a1, rgb, a2)
    assert diff.item() > same.item()
