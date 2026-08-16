import json
import build_subset as bs


def _fake_root(tmp_path):
    for sub, uid in [("000-000", "aaa"), ("000-000", "bbb"), ("000-001", "ccc")]:
        d = tmp_path / sub
        d.mkdir(exist_ok=True)
        (d / f"{uid}.glb").write_bytes(b"x")
    return str(tmp_path)


def test_find_glbs(tmp_path):
    pairs = bs.find_glbs(_fake_root(tmp_path))
    assert {u for u, _ in pairs} == {"aaa", "bbb", "ccc"}


def test_select_subset_deterministic(tmp_path):
    pairs = bs.find_glbs(_fake_root(tmp_path))
    a = bs.select_subset(pairs, 2, seed=42)
    b = bs.select_subset(pairs, 2, seed=42)
    assert a == b and len(a) == 2


def test_write_subset(tmp_path):
    pairs = [("aaa", "/p/aaa.glb"), ("bbb", "/p/bbb.glb")]
    out = tmp_path / "subset.json"
    bs.write_subset(pairs, str(out))
    data = json.load(open(out))
    assert data == {"aaa": "/p/aaa.glb", "bbb": "/p/bbb.glb"}


def test_select_subset_curated_filters(tmp_path):
    pairs = bs.find_glbs(_fake_root(tmp_path))          # aaa, bbb, ccc
    curated = {"aaa", "ccc"}
    sel = bs.select_subset(pairs, 10, seed=42, curated_uids=curated)
    got = {u for u, _ in sel}
    assert got == {"aaa", "ccc"}                        # bbb elenir


def test_load_curated_uids_reads_cache(tmp_path):
    cache = tmp_path / "lvis_uids.json"
    json.dump(["u1", "u2", "u3"], open(cache, "w"))
    uids = bs.load_curated_uids(str(cache))
    assert uids == {"u1", "u2", "u3"}
