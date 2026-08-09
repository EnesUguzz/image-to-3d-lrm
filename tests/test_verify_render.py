import json
import os
from PIL import Image
import verify_render as vr


def _make_render(tmp_path, uid, n=16):
    d = tmp_path / uid
    d.mkdir()
    for i in range(n):
        Image.new("RGBA", (32, 32), (i * 10 % 255, 0, 0, 255)).save(d / f"{i:03d}.png")
    json.dump({"uid": uid, "num_views": n, "views": [{"index": i} for i in range(n)]},
              open(d / "meta.json", "w"))
    return str(tmp_path)


def test_check_meta_ok(tmp_path):
    root = _make_render(tmp_path, "uid1")
    assert vr.check_meta(root, "uid1") == []


def test_check_meta_flags_missing_png(tmp_path):
    root = _make_render(tmp_path, "uid1")
    os.remove(os.path.join(root, "uid1", "003.png"))
    assert any("003" in p for p in vr.check_meta(root, "uid1"))


def test_contact_sheet_written(tmp_path):
    root = _make_render(tmp_path, "uid1")
    out = tmp_path / "sheet.png"
    vr.contact_sheet(root, "uid1", str(out))
    assert out.exists()
    assert Image.open(out).size[0] > 0
