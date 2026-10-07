import json
import os
import subprocess
import pytest

BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
GLB = os.path.join(os.path.expanduser("~"), ".objaverse", "hf-objaverse-v1", "glbs",
                   "000-000", "001abb1a3f4c412fbd707239acb68cd6.glb")
UID = "001abb1a3f4c412fbd707239acb68cd6"


@pytest.mark.integration
def test_render_one_object(tmp_path):
    out = str(tmp_path)
    cmd = [BLENDER, "-b", "--factory-startup", "-P", "scripts/render_object.py", "--",
           "--object_path", GLB, "--output_dir", out, "--uid", UID, "--resolution", "256"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    assert "RENDER_OK" in r.stdout, r.stdout + r.stderr
    d = os.path.join(out, UID)
    pngs = [f for f in os.listdir(d) if f.endswith(".png")]
    assert len(pngs) == 16
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["num_views"] == 16
    assert len(meta["views"][0]["extrinsic"]) == 4
