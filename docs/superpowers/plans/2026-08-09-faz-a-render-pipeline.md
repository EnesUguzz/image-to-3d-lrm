# Faz A — Render Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Objaverse `.glb` objelerini sabit kanonik + seed'li rastgele açılardan render'layıp her obje için 16 görüntü + tam kamera pozları (`meta.json`) üreten, durup devam edebilen bir batch pipeline kurmak.

**Architecture:** Saf geometri/poz matematiği tek bir modülde (`camera_poses.py`) toplanır ve HEM sistem Python (testler) HEM Blender'ın gömülü Python'u (`render_object.py`) tarafından import edilir — böylece render'ın çekirdeği bpy olmadan birim-test edilir. Blender'a bağımlı render ise entegrasyon testiyle (gerçek Blender çağrısı) doğrulanır. Sürücü scriptleri (subset seçimi, batch, doğrulama) sistem Python'da çalışır.

**Tech Stack:** Python 3.10 (sürücü), Blender 4.4 gömülü Python (`bpy`), Cycles+OPTIX, numpy, Pillow, pytest.

## Global Constraints

- Render motoru: **CYCLES** (EEVEE değil). Cihaz: **GPU/OPTIX**, fallback CUDA.
- Çözünürlük: **512×512 RGBA PNG**, `film_transparent = True`.
- Obje başına **16 görünüm**: index 000–003 kanonik (azimuth 0/90/180/270°, elevation +20°), 004–015 supervision (uid-seed'li rastgele, azimuth [0,360°), elevation [−10°,+80°]).
- Normalizasyon: bbox merkezle → **bounding-sphere** yarıçapını `TARGET_RADIUS = 0.5`'e ölçekle.
- Kamera mesafesi FOV'dan: `mesafe = TARGET_RADIUS / sin(FILL_FACTOR · yarı_FOV)`, `FILL_FACTOR = 0.80`, `lens = 35mm`, `sensor = 32mm` → `radius ≈ 1.487`.
- Determinizm: uid→açı eşlemesi `hashlib.sha1` ile (asla built-in `hash()` — process başına tuzlanır).
- Çıktı yolları: `dataset/renders/<uid>/{000..015}.png` + `meta.json`; `dataset/manifest.jsonl`; `dataset/subset.json`.
- Blender exe: `C:\Program Files\Blender Foundation\Blender 4.4\blender.exe`.
- Örnek test objesi: `%USERPROFILE%\.objaverse\hf-objaverse-v1\glbs\000-000\001abb1a3f4c412fbd707239acb68cd6.glb`.
- Obje kaynak kökü: `%USERPROFILE%\.objaverse\hf-objaverse-v1\glbs`.

---

## File Structure

- `scripts/camera_poses.py` — Saf poz/geometri (numpy+stdlib). Blender'da ve testlerde import edilir.
- `scripts/render_object.py` — Blender içinde (`blender -b -P`) çalışır; tek objeyi render eder.
- `scripts/build_subset.py` — glb kökünden N uid seçer → `subset.json`.
- `scripts/run_batch.py` — subset'i gezer, Blender'ı subprocess ile çağırır, resume + timeout + manifest.
- `scripts/verify_render.py` — bir objenin 16 görünümünü contact-sheet PNG'ye dizer + meta bütünlüğü.
- `tests/test_camera_poses.py`, `tests/test_build_subset.py`, `tests/test_run_batch.py`, `tests/test_verify_render.py`, `tests/test_render_integration.py`
- `pytest.ini`, `requirements-dev.txt`

---

## Task 1: Proje env + poz örnekleme (sampling)

**Files:**
- Create: `requirements-dev.txt`, `pytest.ini`, `scripts/camera_poses.py`
- Test: `tests/test_camera_poses.py`

**Interfaces:**
- Produces:
  - `CANONICAL = [(0.0,20.0),(90.0,20.0),(180.0,20.0),(270.0,20.0)]` (azimuth_deg, elevation_deg)
  - `sample_supervision_views(uid: str, n: int = 12, elev_range=(-10.0, 80.0)) -> list[tuple[float,float]]`
  - `build_view_list(uid: str) -> list[dict]` — her dict: `{"index":int,"role":str,"azimuth_deg":float,"elevation_deg":float}`

- [ ] **Step 1: Env kur ve bağımlılıkları yaz**

`requirements-dev.txt`:
```
numpy>=1.26
Pillow>=10.0
pytest>=8.0
```

Run (PowerShell):
```
cd <proje-koku>
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
```
Expected: numpy, Pillow, pytest kurulur.

- [ ] **Step 2: pytest yapılandırması**

`pytest.ini`:
```ini
[pytest]
pythonpath = . scripts
testpaths = tests
```

- [ ] **Step 3: Failing test yaz**

`tests/test_camera_poses.py`:
```python
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
```

- [ ] **Step 4: Testi çalıştır, FAIL gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_camera_poses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'camera_poses'`.

- [ ] **Step 5: Minimal implementasyon**

`scripts/camera_poses.py`:
```python
"""Saf kamera poz/geometri matematiği. bpy'ye bağımlı DEĞİL —
hem sistem Python'da (testler) hem Blender gömülü Python'unda import edilir."""
import hashlib
import math
import random

CANONICAL = [(0.0, 20.0), (90.0, 20.0), (180.0, 20.0), (270.0, 20.0)]


def _seed_from_uid(uid: str) -> int:
    return int(hashlib.sha1(uid.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF


def sample_supervision_views(uid, n=12, elev_range=(-10.0, 80.0)):
    """uid-seed'li, alan-uniform (sin(elev) üzerinden) supervision açıları."""
    rng = random.Random(_seed_from_uid(uid))
    lo, hi = math.sin(math.radians(elev_range[0])), math.sin(math.radians(elev_range[1]))
    out = []
    for _ in range(n):
        az = rng.uniform(0.0, 360.0)
        el = math.degrees(math.asin(rng.uniform(lo, hi)))
        out.append((az, el))
    return out


def build_view_list(uid):
    views = []
    for i, (az, el) in enumerate(CANONICAL):
        views.append({"index": i, "role": "canonical",
                      "azimuth_deg": az, "elevation_deg": el})
    for j, (az, el) in enumerate(sample_supervision_views(uid)):
        views.append({"index": 4 + j, "role": "supervision",
                      "azimuth_deg": az, "elevation_deg": el})
    return views
```

- [ ] **Step 6: Testi çalıştır, PASS gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_camera_poses.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```
git add requirements-dev.txt pytest.ini scripts/camera_poses.py tests/test_camera_poses.py
git commit -m "Task1: poz ornekleme modulu + env"
```

---

## Task 2: Kamera geometrisi (konum, mesafe, intrinsic)

**Files:**
- Modify: `scripts/camera_poses.py`
- Test: `tests/test_camera_poses.py`

**Interfaces:**
- Produces:
  - `camera_location(azimuth_deg, elevation_deg, radius) -> (x, y, z)`
  - `camera_distance(target_radius=0.5, lens_mm=35.0, sensor_mm=32.0, fill_factor=0.80) -> float`
  - `intrinsic_matrix(lens_mm=35.0, sensor_mm=32.0, resolution=512) -> list[list[float]]` (3×3)

- [ ] **Step 1: Failing test yaz**

`tests/test_camera_poses.py`'ye ekle:
```python
import math


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
```

- [ ] **Step 2: Testi çalıştır, FAIL gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_camera_poses.py -k "location or distance or intrinsic" -v`
Expected: FAIL — `AttributeError: module 'camera_poses' has no attribute 'camera_location'`.

- [ ] **Step 3: Implementasyon ekle**

`scripts/camera_poses.py` sonuna ekle:
```python
def camera_location(azimuth_deg, elevation_deg, radius):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (radius * math.cos(el) * math.cos(az),
            radius * math.cos(el) * math.sin(az),
            radius * math.sin(el))


def camera_distance(target_radius=0.5, lens_mm=35.0, sensor_mm=32.0, fill_factor=0.80):
    half_fov = math.atan(sensor_mm / (2.0 * lens_mm))
    return target_radius / math.sin(fill_factor * half_fov)


def intrinsic_matrix(lens_mm=35.0, sensor_mm=32.0, resolution=512):
    f = lens_mm / sensor_mm * resolution
    c = resolution / 2.0
    return [[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]]
```

- [ ] **Step 4: Testi çalıştır, PASS gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_camera_poses.py -v`
Expected: tüm testler passed.

- [ ] **Step 5: Commit**

```
git add scripts/camera_poses.py tests/test_camera_poses.py
git commit -m "Task2: kamera konumu, mesafe (FOV), intrinsic"
```

---

## Task 3: Blender render scripti (tek obje)

**Files:**
- Create: `scripts/render_object.py`
- Test: `tests/test_render_integration.py`

**Interfaces:**
- Consumes: `camera_poses.build_view_list`, `camera_distance`, `camera_location`, `intrinsic_matrix`.
- Produces (dosya sözleşmesi): `<output_dir>/<uid>/000..015.png` + `<output_dir>/<uid>/meta.json`.
  `meta.json` anahtarları: `uid, resolution, num_views, camera{lens_mm,sensor_mm,radius,target_radius,fill_factor}, canonical_indices, views[]`; her view: `index, role, file, azimuth_deg, elevation_deg, extrinsic(4×4 world→camera), intrinsic(3×3)`.

- [ ] **Step 1: Render scriptini yaz**

`scripts/render_object.py`:
```python
"""Blender içinde çalışır: blender -b -P scripts/render_object.py -- --object_path X.glb ...
Tek objeyi 16 açıdan render eder ve meta.json yazar."""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

# Blender'ın gömülü Python'una scripts/ klasörünü ekle (camera_poses import için)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import camera_poses as cp

TARGET_RADIUS = 0.5
FILL_FACTOR = 0.80
LENS_MM = 35.0
SENSOR_MM = 32.0


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--object_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--uid", required=True)
    p.add_argument("--resolution", type=int, default=512)
    return p.parse_args(argv)


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA"):
        prefs.compute_device_type = dev_type
        prefs.refresh_devices()
        gpus = [d for d in prefs.devices if d.type == dev_type]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == dev_type
            return dev_type
    return None


def setup_render(resolution):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True


def reset_scene():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)


def load_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)


def _mesh_bbox():
    mn = Vector((math.inf,) * 3)
    mx = Vector((-math.inf,) * 3)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            mn = Vector(min(a, b) for a, b in zip(mn, w))
            mx = Vector(max(a, b) for a, b in zip(mx, w))
    return mn, mx


def normalize_scene():
    mn, mx = _mesh_bbox()
    center = (mn + mx) / 2.0
    radius_norm = (mx - center).length            # köşe mesafesi (güvenli üst sınır)
    scale = TARGET_RADIUS / radius_norm
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.location = (obj.location - center) * scale
            obj.scale = obj.scale * scale
    bpy.context.view_layer.update()


def setup_lighting():
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.scene.world = bpy.data.worlds.new("W") if not bpy.context.scene.world else bpy.context.scene.world
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[1].default_value = 1.0    # sabit ortam ışığı
    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.energy = 1000
    light_data.size = 5.0
    light = bpy.data.objects.new("Key", light_data)
    light.location = (0, 0, 4)
    bpy.context.scene.collection.objects.link(light)


def setup_camera():
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.lens = LENS_MM
    cam.data.sensor_width = SENSOR_MM
    cam.data.sensor_fit = "HORIZONTAL"
    empty = bpy.data.objects.new("Target", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = (0, 0, 0)
    con = cam.constraints.new(type="TRACK_TO")
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    con.target = empty
    return cam


def main():
    args = parse_args()
    enable_gpu()
    setup_render(args.resolution)
    reset_scene()
    load_glb(args.object_path)
    normalize_scene()
    setup_lighting()
    cam = setup_camera()

    radius = cp.camera_distance(TARGET_RADIUS, LENS_MM, SENSOR_MM, FILL_FACTOR)
    out_dir = os.path.join(args.output_dir, args.uid)
    os.makedirs(out_dir, exist_ok=True)
    views_meta = []
    for v in cp.build_view_list(args.uid):
        cam.location = cp.camera_location(v["azimuth_deg"], v["elevation_deg"], radius)
        bpy.context.view_layer.update()
        fname = f"{v['index']:03d}.png"
        bpy.context.scene.render.filepath = os.path.join(out_dir, fname)
        bpy.ops.render.render(write_still=True)
        extrinsic = [list(row) for row in cam.matrix_world.inverted()]  # world→camera
        views_meta.append({**v, "file": fname, "extrinsic": extrinsic,
                           "intrinsic": cp.intrinsic_matrix(LENS_MM, SENSOR_MM, args.resolution)})

    meta = {"uid": args.uid, "resolution": args.resolution, "num_views": len(views_meta),
            "camera": {"lens_mm": LENS_MM, "sensor_mm": SENSOR_MM, "radius": radius,
                       "target_radius": TARGET_RADIUS, "fill_factor": FILL_FACTOR},
            "canonical_indices": [0, 1, 2, 3], "views": views_meta}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("RENDER_OK", args.uid, len(views_meta))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Entegrasyon testi yaz**

`tests/test_render_integration.py`:
```python
import json
import os
import subprocess
import pytest

BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
GLB = r"%USERPROFILE%\.objaverse\hf-objaverse-v1\glbs\000-000\001abb1a3f4c412fbd707239acb68cd6.glb"
UID = "001abb1a3f4c412fbd707239acb68cd6"


@pytest.mark.integration
def test_render_one_object(tmp_path):
    out = str(tmp_path)
    cmd = [BLENDER, "-b", "--factory-startup", "-P", "scripts/render_object.py", "--",
           "--object_path", GLB, "--output_dir", out, "--uid", UID, "--resolution", "256"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert "RENDER_OK" in r.stdout, r.stdout + r.stderr
    d = os.path.join(out, UID)
    pngs = [f for f in os.listdir(d) if f.endswith(".png")]
    assert len(pngs) == 16
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["num_views"] == 16
    assert len(meta["views"][0]["extrinsic"]) == 4
```

Ekle `pytest.ini`'ye marker kaydı:
```ini
markers =
    integration: Blender gerektiren yavaş entegrasyon testleri
```

- [ ] **Step 3: Entegrasyon testini çalıştır (gerçek Blender)**

Run: `.\.venv\Scripts\python -m pytest tests/test_render_integration.py -m integration -v -s`
Expected: PASS — 16 PNG + meta.json, `RENDER_OK` çıktısı. (İlk çalıştırma OPTIX derlemesiyle 1-2 dk sürebilir.)
Eğer FAIL: stdout/stderr'deki bpy hatasına göre `systematic-debugging` skill'i ile düzelt (ör. gltf importer flag'i, GPU device adı).

- [ ] **Step 4: Commit**

```
git add scripts/render_object.py tests/test_render_integration.py pytest.ini
git commit -m "Task3: Blender render scripti + entegrasyon testi (16 gorunum + meta)"
```

---

## Task 4: Subset seçici

**Files:**
- Create: `scripts/build_subset.py`
- Test: `tests/test_build_subset.py`

**Interfaces:**
- Produces:
  - `find_glbs(root: str) -> list[tuple[str,str]]` — `(uid, abspath)` listesi.
  - `select_subset(pairs, n, seed=42) -> list[tuple[str,str]]` — deterministik.
  - `write_subset(pairs, out_path)` — `{uid: abspath}` JSON yazar.
  - CLI: `--root --n --out`.

- [ ] **Step 1: Failing test yaz**

`tests/test_build_subset.py`:
```python
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
```

- [ ] **Step 2: Testi çalıştır, FAIL gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_build_subset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_subset'`.

- [ ] **Step 3: Implementasyon**

`scripts/build_subset.py`:
```python
"""glb kökünden N obje seçip subset.json yazar (sistem Python)."""
import argparse
import glob
import json
import os
import random


def find_glbs(root):
    out = []
    for path in glob.glob(os.path.join(root, "**", "*.glb"), recursive=True):
        uid = os.path.splitext(os.path.basename(path))[0]
        out.append((uid, os.path.abspath(path)))
    return out


def select_subset(pairs, n, seed=42):
    pairs = sorted(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def write_subset(pairs, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({u: p for u, p in pairs}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="dataset/subset.json")
    a = ap.parse_args()
    pairs = select_subset(find_glbs(a.root), a.n)
    write_subset(pairs, a.out)
    print(f"subset yazildi: {len(pairs)} obje -> {a.out}")
```

- [ ] **Step 4: Testi çalıştır, PASS gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_build_subset.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add scripts/build_subset.py tests/test_build_subset.py
git commit -m "Task4: subset secici"
```

---

## Task 5: Batch sürücü (resume + timeout)

**Files:**
- Create: `scripts/run_batch.py`
- Test: `tests/test_run_batch.py`

**Interfaces:**
- Consumes: `subset.json` (`{uid: glb_path}`).
- Produces:
  - `is_done(output_dir, uid, num_views=16) -> bool` — `meta.json` + num_views png varsa True.
  - `render_command(blender, uid, glb_path, output_dir, resolution) -> list[str]`
  - `run_batch(subset, output_dir, blender, resolution, timeout)` — döngü + `manifest.jsonl`.

- [ ] **Step 1: Failing test yaz (resume + komut)**

`tests/test_run_batch.py`:
```python
import json
import os
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
```

- [ ] **Step 2: Testi çalıştır, FAIL gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_run_batch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_batch'`.

- [ ] **Step 3: Implementasyon**

`scripts/run_batch.py`:
```python
"""subset.json'u gezip her objeyi Blender ile render eder. Resume + timeout + manifest."""
import argparse
import json
import os
import subprocess
import time

BLENDER_DEFAULT = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"


def is_done(output_dir, uid, num_views=16):
    d = os.path.join(output_dir, uid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        return False
    pngs = [f for f in os.listdir(d) if f.endswith(".png")] if os.path.isdir(d) else []
    return len(pngs) >= num_views


def render_command(blender, uid, glb_path, output_dir, resolution):
    return [blender, "-b", "--factory-startup", "-P", "scripts/render_object.py", "--",
            "--object_path", glb_path, "--output_dir", output_dir,
            "--uid", uid, "--resolution", str(resolution)]


def run_batch(subset, output_dir, blender=BLENDER_DEFAULT, resolution=512, timeout=120):
    os.makedirs(output_dir, exist_ok=True)
    manifest = os.path.join(output_dir, "manifest.jsonl")
    total = len(subset)
    for i, (uid, glb) in enumerate(subset.items(), 1):
        if is_done(output_dir, uid):
            print(f"[{i}/{total}] atla (tamam): {uid}")
            continue
        t0 = time.time()
        rec = {"uid": uid, "status": "done", "error": None, "blender": "4.4"}
        try:
            r = subprocess.run(render_command(blender, uid, glb, output_dir, resolution),
                               capture_output=True, text=True, timeout=timeout)
            if "RENDER_OK" not in r.stdout or not is_done(output_dir, uid):
                rec["status"] = "failed"
                rec["error"] = (r.stdout + r.stderr)[-500:]
        except subprocess.TimeoutExpired:
            rec["status"] = "failed"
            rec["error"] = f"timeout>{timeout}s"
        rec["seconds"] = round(time.time() - t0, 1)
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i}/{total}] {rec['status']}: {uid} ({rec['seconds']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dataset/subset.json")
    ap.add_argument("--output_dir", default="dataset/renders")
    ap.add_argument("--blender", default=BLENDER_DEFAULT)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    with open(a.subset) as f:
        subset = json.load(f)
    run_batch(subset, a.output_dir, a.blender, a.resolution, a.timeout)
```

- [ ] **Step 4: Testi çalıştır, PASS gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_run_batch.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add scripts/run_batch.py tests/test_run_batch.py
git commit -m "Task5: batch surucu (resume + timeout + manifest)"
```

---

## Task 6: Doğrulama / contact sheet

**Files:**
- Create: `scripts/verify_render.py`
- Test: `tests/test_verify_render.py`

**Interfaces:**
- Produces:
  - `check_meta(render_dir, uid, num_views=16) -> list[str]` — sorun listesi (boşsa OK).
  - `contact_sheet(render_dir, uid, out_path, cols=4)` — 16 png'yi grid PNG'ye dizer.
  - CLI: `--render_dir --uid --out`.

- [ ] **Step 1: Failing test yaz**

`tests/test_verify_render.py`:
```python
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
```

- [ ] **Step 2: Testi çalıştır, FAIL gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_verify_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verify_render'`.

- [ ] **Step 3: Implementasyon**

`scripts/verify_render.py`:
```python
"""Render çıktısını doğrular ve contact-sheet üretir (sistem Python + Pillow)."""
import argparse
import json
import os
from PIL import Image


def check_meta(render_dir, uid, num_views=16):
    problems = []
    d = os.path.join(render_dir, uid)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.isfile(meta_path):
        return [f"{uid}: meta.json yok"]
    meta = json.load(open(meta_path))
    if meta.get("num_views") != num_views:
        problems.append(f"{uid}: num_views {meta.get('num_views')} != {num_views}")
    for i in range(num_views):
        if not os.path.isfile(os.path.join(d, f"{i:03d}.png")):
            problems.append(f"{uid}: {i:03d}.png eksik")
    return problems


def contact_sheet(render_dir, uid, out_path, cols=4):
    d = os.path.join(render_dir, uid)
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    imgs = [Image.open(os.path.join(d, f)).convert("RGBA") for f in files]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * w, rows * h), (30, 30, 30, 255))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h), im)
    sheet.convert("RGB").save(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", default="dataset/renders")
    ap.add_argument("--uid", required=True)
    ap.add_argument("--out", default="dataset/contact_sheet.png")
    a = ap.parse_args()
    probs = check_meta(a.render_dir, a.uid)
    print("SORUN YOK" if not probs else "\n".join(probs))
    contact_sheet(a.render_dir, a.uid, a.out)
    print("contact sheet:", a.out)
```

- [ ] **Step 4: Testi çalıştır, PASS gör**

Run: `.\.venv\Scripts\python -m pytest tests/test_verify_render.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add scripts/verify_render.py tests/test_verify_render.py
git commit -m "Task6: dogrulama + contact sheet"
```

---

## Task 7: Uçtan uca çalıştırma — birkaç obje ile kontrol

**Files:** (kod değişmez; pipeline'ı gerçek veride çalıştırıp gözle doğrula)

- [ ] **Step 1: Tüm birim testleri çalıştır**

Run: `.\.venv\Scripts\python -m pytest -v -m "not integration"`
Expected: tüm birim testler passed.

- [ ] **Step 2: Küçük subset üret (5 obje)**

Run:
```
.\.venv\Scripts\python scripts/build_subset.py --root "%USERPROFILE%\.objaverse\hf-objaverse-v1\glbs" --n 5 --out dataset/subset_smoke.json
```
Expected: `subset yazildi: 5 obje -> dataset/subset_smoke.json`.

- [ ] **Step 3: Batch render (512, 5 obje)**

Run:
```
.\.venv\Scripts\python scripts/run_batch.py --subset dataset/subset_smoke.json --output_dir dataset/renders --resolution 512 --timeout 180
```
Expected: her obje için `done: <uid> (~20s)`; `dataset/manifest.jsonl` 5 satır.

- [ ] **Step 4: Doğrula + contact sheet (her obje)**

Run (her uid için subset_smoke.json'daki uid'lerle):
```
.\.venv\Scripts\python scripts/verify_render.py --render_dir dataset/renders --uid <UID> --out dataset/sheet_<UID>.png
```
Expected: `SORUN YOK` + `dataset/sheet_<UID>.png`. **Contact sheet'i aç ve gözle kontrol et:**
obje ortalanmış mı, çerçeveyi ~%80 dolduruyor mu, arka plan şeffaf mı, 4 kanonik açı ön/arka/sol/sağ mı, supervision açıları çeşitli mi.

- [ ] **Step 5: Resume'u doğrula**

Run: Step 3'ü tekrar çalıştır.
Expected: hepsi `atla (tamam)` — hiçbir obje yeniden render edilmez.

- [ ] **Step 6: Bulguları not et**

Contact sheet'te sorun varsa (kırpılma, karanlık, yanlış açı) `CLAUDE.md`'ye not düş ve ilgili parametreyi (`FILL_FACTOR`, ışık `energy`, sample) ayarla. Sorun yoksa Faz A doğrulandı — subset N'i artırıp tam veri setine geçilebilir.

---

## Self-Review Notları

- **Spec kapsamı:** normalizasyon+framing (Task3 normalize_scene + camera_distance), 16 görünüm/kanonik+supervision (Task1), CYCLES/OPTIX/512/şeffaf (Task3 setup_render), meta.json şeması (Task3), resume/timeout/manifest (Task5), contact-sheet doğrulama (Task6), uçtan uca kontrol (Task7) — hepsi karşılandı.
- **Kapsam dışı** (depth/normal, çok-GPU, web app, eğitim) plana dahil edilmedi — doğru.
- **Tip tutarlılığı:** `build_view_list`/`camera_location`/`camera_distance`/`intrinsic_matrix` imzaları Task1-2'de tanımlanıp Task3'te aynen kullanıldı; `is_done`/`render_command` Task5 içinde tutarlı.
