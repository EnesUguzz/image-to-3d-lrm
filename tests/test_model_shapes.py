import torch
import torch.nn as nn

from lrm.model import LRM


class FakeEncoder(nn.Module):
    embed_dim = 384
    patch = 14

    def forward(self, imgs):  # (V,3,224,224) -> (V,256,384)
        v = imgs.shape[0]
        return torch.randn(v, 256, 384)


def test_forward_output_shapes():
    model = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
                nerf_hidden=16, encoder=FakeEncoder(), n_samples=8)
    Vi, Vr, H, W = 2, 3, 16, 16
    imgs = torch.rand(Vi, 3, 224, 224)
    ic2w = torch.eye(4)[None].expand(Vi, 4, 4).contiguous()
    iK = torch.tensor([[100., 0, 8], [0, 100, 8], [0, 0, 1]])[None].expand(Vi, 3, 3).contiguous()
    rc2w = torch.eye(4)[None].expand(Vr, 4, 4).contiguous()
    rK = torch.tensor([[50., 0, 8], [0, 50, 8], [0, 0, 1]])[None].expand(Vr, 3, 3).contiguous()
    rgb, acc = model(imgs, ic2w, iK, rc2w, rK, (H, W))
    assert rgb.shape == (Vr, 3, H, W)
    assert acc.shape == (Vr, 1, H, W)


def test_forward_single_input_view():
    model = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
                nerf_hidden=16, encoder=FakeEncoder(), n_samples=8)
    imgs = torch.rand(1, 3, 224, 224)
    iK = torch.tensor([[100., 0, 8], [0, 100, 8], [0, 0, 1]])[None]
    ic2w = torch.eye(4)[None]
    rc2w = torch.eye(4)[None]
    rK = iK.clone()
    rgb, acc = model(imgs, ic2w, iK, rc2w, rK, (8, 8))
    assert rgb.shape == (1, 3, 8, 8)
