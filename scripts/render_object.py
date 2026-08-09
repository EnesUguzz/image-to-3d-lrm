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
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("W")
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
