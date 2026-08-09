import math

import camera_poses as cp


def test_canonical_is_four_fixed_views():
    assert cp.CANONICAL == [(0.0, 20.0), (90.0, 20.0), (180.0, 20.0), (270.0, 20.0)]


def test_supervision_is_deterministic_per_uid():
    a = cp.sample_supervision_views("abc123")
    b = cp.sample_supervision_views("abc123")
    assert a == b
    assert cp.sample_supervision_views("different") != a


def test_supervision_count_and_ranges():
    views = cp.sample_supervision_views("uid-x", n=12)
    assert len(views) == 12
    for az, el in views:
        assert 0.0 <= az < 360.0
        assert -10.0 <= el <= 80.0


def test_build_view_list_shape():
    views = cp.build_view_list("uid-y")
    assert len(views) == 16
    assert [v["role"] for v in views[:4]] == ["canonical"] * 4
    assert all(v["role"] == "supervision" for v in views[4:])
    assert [v["index"] for v in views] == list(range(16))
    assert (views[0]["azimuth_deg"], views[0]["elevation_deg"]) == (0.0, 20.0)


def test_camera_location_axes():
    r = 1.5
    x, y, z = cp.camera_location(0.0, 0.0, r)
    assert math.isclose(x, r, abs_tol=1e-9) and abs(y) < 1e-9 and abs(z) < 1e-9
    x, y, z = cp.camera_location(90.0, 0.0, r)
    assert abs(x) < 1e-9 and math.isclose(y, r, abs_tol=1e-9)
    x, y, z = cp.camera_location(0.0, 90.0, r)
    assert math.isclose(z, r, abs_tol=1e-9)


def test_camera_distance_matches_formula():
    d = cp.camera_distance()
    assert math.isclose(d, 1.487, abs_tol=0.01)


def test_intrinsic_matrix():
    K = cp.intrinsic_matrix(35.0, 32.0, 512)
    assert math.isclose(K[0][0], 560.0, abs_tol=1e-6)   # f = 35/32*512
    assert math.isclose(K[1][1], 560.0, abs_tol=1e-6)
    assert K[0][2] == 256.0 and K[1][2] == 256.0 and K[2][2] == 1.0
