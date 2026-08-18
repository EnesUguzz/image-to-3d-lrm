import torch
from lrm import cameras


def _identity_c2w(dist=1.5):
    c2w = torch.eye(4)
    c2w[2, 3] = dist  # kamera +z'de, -z'ye bakar
    return c2w


def _K(res=128, f=200.0):
    K = torch.tensor([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]], dtype=torch.float32)
    return K


def test_scale_intrinsics_halves():
    K = _K(res=512, f=800.0)
    Ks = cameras.scale_intrinsics(K, 512, 256)
    assert torch.allclose(Ks[0, 0], torch.tensor(400.0))
    assert torch.allclose(Ks[0, 2], torch.tensor(128.0))


def test_center_ray_points_forward():
    c2w = _identity_c2w(dist=1.5)
    K = _K(res=128)
    o, d = cameras.rays_from_camera(c2w, K, 128, 128)
    center = (64 * 128 + 64)  # yaklasik merkez piksel
    # merkez isin dunya -z yonunde olmali (kamera +z'den -z'ye bakar)
    assert torch.allclose(d[center], torch.tensor([0.0, 0.0, -1.0]), atol=1e-2)
    assert torch.allclose(o[center], torch.tensor([0.0, 0.0, 1.5]), atol=1e-5)


def test_plucker_shape_and_center_moment_zero():
    c2w = _identity_c2w(dist=1.5)
    K = _K(res=16)
    pl = cameras.plucker_map(c2w, K, 16, 16)
    assert pl.shape == (16, 16, 6)
    # tam merkeze yakin pikselde d=-z, o=+1.5z => o x d ~ 0
    m = pl[8, 8, 3:]
    assert m.norm() < 0.2
