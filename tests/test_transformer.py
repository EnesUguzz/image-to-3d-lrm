import torch

from lrm.transformer import LRMTransformer


def test_transformer_output_shape():
    net = LRMTransformer(dim=64, depth=2, heads=4, triplane_res=8, img_dim=384)
    img_tokens = torch.randn(256, 384)   # ornegin 1 gorunum 16x16
    plucker = torch.randn(256, 6)
    out = net(img_tokens, plucker)
    assert out.shape == (3, 8, 8, 64)


def test_transformer_handles_multiview_token_count():
    net = LRMTransformer(dim=64, depth=2, heads=4, triplane_res=8, img_dim=384)
    # 3 gorunum => 3*256 token
    out = net(torch.randn(768, 384), torch.randn(768, 6))
    assert out.shape == (3, 8, 8, 64)
