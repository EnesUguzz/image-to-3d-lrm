import run_batch as rb


def test_is_done_false_when_missing(tmp_path):
    assert rb.is_done(str(tmp_path), "uid1") is False


def test_is_done_true_when_complete(tmp_path):
    d = tmp_path / "uid1"
    d.mkdir()
    (d / "meta.json").write_text("{}")
    for i in range(16):
        (d / f"{i:03d}.png").write_bytes(b"x")
    assert rb.is_done(str(tmp_path), "uid1") is True


def test_render_command_has_key_args():
    cmd = rb.render_command("blender.exe", "uid1", "a.glb", "out", 512)
    assert "blender.exe" in cmd and "-b" in cmd and "--object_path" in cmd
    assert "uid1" in cmd and "a.glb" in cmd
