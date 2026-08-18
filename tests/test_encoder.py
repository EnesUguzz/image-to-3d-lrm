import pytest
import torch

from lrm.encoder import DinoEncoder


@pytest.mark.slow
def test_dino_patch_tokens_shape():
    enc = DinoEncoder()
    imgs = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        tok = enc(imgs)
    assert tok.shape == (2, 256, 384)


@pytest.mark.slow
def test_dino_is_frozen():
    enc = DinoEncoder()
    assert all(not p.requires_grad for p in enc.parameters())
