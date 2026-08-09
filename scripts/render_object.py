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


def _world_extent(obj):
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector(min(p[i] for p in cs) for i in range(3))
    mx = Vector(max(p[i] for p in cs) for i in range(3))
    return mx - mn


def remove_floor_planes():
    """Objaverse glb'lerinde sık görülen dev zemin/plane mesh'lerini siler.
    Sezgi: 'düz' (min boyut ~0) VE içerikten (en büyük düz-olmayan mesh) çok
    daha büyük olan mesh bir zemindir. Tek mesh'li düz objelere dokunmaz."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(meshes) < 2:
        return
    ext = {o: _world_extent(o) for o in meshes}
    flat = [o for o in meshes if min(ext[o]) < 0.02 * max(max(ext[o]), 1e-9)]
    non_flat_max = max((max(ext[o]) for o in meshes if o not in flat), default=0.0)
    if non_flat_max <= 0:
        return
    for o in flat:
        if max(ext[o]) > 2.0 * non_flat_max:
            bpy.data.objects.remove(o, do_unlink=True)


def _scene_root_objects():
    return [o for o in bpy.context.scene.objects if o.parent is None]


def normalize_scene():
    # İki geçiş (objaverse-rendering ile aynı, hiyerarşiye dayanıklı):
    # 1) kök objeleri ölçekle, 2) güncelle+bbox'ı yeniden ölç, 3) dünya-uzayında merkezle.
    mn, mx = _mesh_bbox()
    radius_norm = (mx - mn).length / 2.0          # yarı-köşegen = bounding sphere üst sınırı
    if radius_norm == 0:
        return
    scale = TARGET_RADIUS / radius_norm
    for obj in _scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    mn, mx = _mesh_bbox()
    offset = -(mn + mx) / 2.0
    for obj in _scene_root_objects():
        obj.matrix_world.translation = obj.matrix_world.translation + offset
    bpy.context.view_layer.update()


def setup_lighting():
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("W")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[1].default_value = 1.5    # sabit ortam ışığı (her yönden dengeli)
    # Üst + alt yumuşak area ışıklar → hiçbir açı kapkaranlık kalmasın
    for name, z in (("KeyTop", 3.0), ("FillBottom", -3.0)):
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = 400
        ld.size = 8.0
        lo = bpy.data.objects.new(name, ld)
        lo.location = (0, 0, z)
        lo.rotation_euler = (0 if z > 0 else math.radians(180), 0, 0)
        bpy.context.scene.collection.objects.link(lo)


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
    remove_floor_planes()
    normalize_scene()
    setup_lighting()
    cam = setup_camera()

    radius = cp.camera_distance(TARGET_RADIUS, LENS_MM, SENSOR_MM, FILL_FACTOR)
    # Mutlak yol: Blender render.filepath'i göreli yolları .blend konumuna göre çözer;
    # blend dosyası olmadığı için göreli yol sürücü köküne düşer. abspath bunu engeller.
    out_dir = os.path.abspath(os.path.join(args.output_dir, args.uid))
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
