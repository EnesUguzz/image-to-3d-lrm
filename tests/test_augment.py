import random

import torch

from lrm import augment


def _rgba(H=32, W=32, alpha_val=1.0):
    x = torch.zeros(4, H, W)
    x[:3, 8:24, 8:24] = 0.7       # ortada obje
    x[3, 8:24, 8:24] = alpha_val  # alpha
    return x


def test_composite_fills_transparent_background():
    rgba = _rgba()
    rng = random.Random(0)
    out = augment.composite_background(rgba, rng)
    assert out.shape == (3, 32, 32)
    # seffaf kose artik bir arka plan rengiyle dolu
    corner = out[:, 0, 0]
    assert corner.abs().sum() > 0.0


def test_augment_output_shape_and_range():
    rng = random.Random(1)
    out = augment.augment_input(_rgba(), rng)
    assert out.shape == (3, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_is_deterministic_with_seed():
    a = augment.augment_input(_rgba(), random.Random(5))
    b = augment.augment_input(_rgba(), random.Random(5))
    assert torch.allclose(a, b)
