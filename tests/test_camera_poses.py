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
