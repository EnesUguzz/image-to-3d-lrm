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


def test_summarize_counts_timing_and_errors():
    records = [
        {"uid": "a", "status": "done", "seconds": 10.0, "error": None},
        {"uid": "b", "status": "done", "seconds": 20.0, "error": None},
        {"uid": "c", "status": "failed", "seconds": 5.0, "error": "timeout>120s"},
        {"uid": "d", "status": "failed", "seconds": 3.0, "error": "Traceback ... bpy error"},
    ]
    s = rb.summarize(records)
    assert s["total"] == 4 and s["done"] == 2 and s["failed"] == 2
    assert abs(s["total_seconds"] - 38.0) < 1e-6
    assert abs(s["avg_seconds"] - 9.5) < 1e-6
    assert s["slowest"][0]["uid"] == "b"          # en yavaş önce
    assert s["error_breakdown"]["timeout"] == 1
    assert s["error_breakdown"]["render_error"] == 1


def test_error_category():
    assert rb._error_category("timeout>120s") == "timeout"
    assert rb._error_category("some bpy traceback") == "render_error"
    assert rb._error_category(None) is None


def test_render_one_success(monkeypatch, tmp_path):
    class R:
        stdout = "blah RENDER_OK uid1 16"
        stderr = ""
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(rb, "is_done", lambda *a, **k: True)
    rec = rb._render_one("uid1", "a.glb", str(tmp_path), "blender", 256, 60)
    assert rec["status"] == "done" and "seconds" in rec and rec["uid"] == "uid1"


def test_render_one_failure(monkeypatch, tmp_path):
    class R:
        stdout = "no ok here"
        stderr = "bpy err"
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(rb, "is_done", lambda *a, **k: False)
    rec = rb._render_one("uid1", "a.glb", str(tmp_path), "blender", 256, 60)
    assert rec["status"] == "failed" and rec["error"]
