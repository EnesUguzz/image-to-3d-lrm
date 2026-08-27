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
# Normalizasyon konvansiyonu:
#   "sphere" (eski): bbox yari-kosegeni = TARGET_RADIUS. Guvenli ama obje kucuk kalir;
#                    olculdu: alpha kaplama ort %9.4, objelerin %22.5'i karenin %5'inden az.
#   "bbox" (referans): objaverse-xl/blender_script.py ile ayni -> scale = 1/max(bbox kenari),
#                    yani en uzun kenar = 2*TARGET_RADIUS. LRM/OpenLRM/TripoSR bu duzende.
#                    Kup bir obje icin obje ~1.73 kat buyur (alan ~3 kat).
#   "fit" (varsayilan): iki gecisli. Once bbox normalize, sonra objenin 16 KAMERA
#   GORUNUMUNDEKI gercek izdusum genisligi olculup kareye tam oturacak sekilde
#   yeniden olceklenir. Kure sinirindan buyuk, bbox sinirindan guvenli.
#   OLCULDU (2026-08-24, 100 obje): "bbox" modunda render'larin %5'i kenardan
#   tasiyordu (duz panel/kutu objeler; en kotu 0.65) -> "fit" bunu kaynaginda cozer.
#   OLCULDU (2026-08-21, 6 obje x 16 gorunum): bbox modu alpha kaplamayi
#   0.0843 -> 0.1577 (1.87x) cikariyor, kenar tasmasi HER IKI modda da 0.000
#   (kirpilma riski yok), render suresi ayni (10.4 sn/obje).
NORM_MODE = os.environ.get("RENDER_NORM_MODE", "fit")
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


# OLCULDU (2026-08-24, 24 obje x 16 gorunum = 384 render cifti):
#   EEVEE_NEXT vs CYCLES -> alpha kaplama farki 0.00001 (geometri birebir),
#   obje RGB farki ort %2.4, PSNR ort 37.4 dB (en kotu 27.3 dB, metalik/yansimali obje).
#   Hiz: 3.68 -> 0.94 sn/obje (4x). Objaverse'in kendi blender_script.py'si de
#   BLENDER_EEVEE varsayilanini kullanir. EEVEE'nin en buyuk zaafi (cam/kirilma)
#   bizi ilgilendirmiyor: Objaverse++ filtresinde is_transparent objeler eleniyor.
ENGINE = os.environ.get("RENDER_ENGINE", "EEVEE").upper()
CYCLES_SAMPLES = int(os.environ.get("RENDER_SAMPLES", "48"))
EEVEE_SAMPLES = int(os.environ.get("RENDER_EEVEE_SAMPLES", "32"))


def setup_render(resolution):
    scene = bpy.context.scene
    if ENGINE.startswith("EEVEE"):
        # EEVEE_NEXT: rasterizer. Bizim sahnemiz duz ortam isigi + albedo;
        # Cycles'in isin izlemesine ihtiyac yok. Blender 4.4'te ad BLENDER_EEVEE_NEXT.
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = EEVEE_SAMPLES
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.device = "GPU"
        scene.cycles.samples = CYCLES_SAMPLES
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
    ext = mx - mn
    if NORM_MODE in ("bbox", "fit"):   # referans konvansiyon: en uzun kenar = 2*TARGET_RADIUS
        longest = max(ext.x, ext.y, ext.z)
        radius_norm = longest / 2.0
    else:                          # "sphere": yari-kosegen (eski, daha kucuk obje)
        radius_norm = ext.length / 2.0
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


def fit_scale_to_views(uid, radius, resolution):
    """Objenin 16 kamera gorunumundeki EN GENIS izdusumunu olcup kareye oturtur.

    Kure sinirina gore olceklemek (eski 'sphere' modu) fazla muhafazakar: obje
    karenin ~%9'unu kapliyordu. bbox sinirina gore olceklemek ise duz/panel
    objelerde tasmaya yol aciyordu (100-obje testi: render'larin %5'i, en kotu 0.65).
    Burada bbox kosleri her kameraya tasinip gercek acisal genislik olculur ve
    obje FILL_FACTOR'e tam oturacak sekilde kucultulur/buyutulur.
    Olcek degisince z de degistigi icin birkac kez yinelenir (hizla yakinsar).
    """
    import mathutils
    half_fov = math.atan(SENSOR_MM / (2.0 * LENS_MM))
    limit = math.tan(FILL_FACTOR * half_fov)
    views = cp.build_view_list(uid)
    cams = [mathutils.Vector(cp.camera_location(v["azimuth_deg"], v["elevation_deg"], radius))
            for v in views]

    for _ in range(5):
        # HER yinelemede once merkezle: obj.scale objenin KENDI orijinine gore
        # olcekler; kok objelerin konumu sifirda degilse olcekleme sahneyi
        # kaydirir ve izdusum olcumu sisip modeli gereksiz kucultur.
        mn, mx = _mesh_bbox()
        off = -(mn + mx) / 2.0
        for obj in _scene_root_objects():
            obj.matrix_world.translation = obj.matrix_world.translation + off
        bpy.context.view_layer.update()
        mn, mx = _mesh_bbox()
        corners = [mathutils.Vector((x, y, z))
                   for x in (mn.x, mx.x) for y in (mn.y, mx.y) for z in (mn.z, mx.z)]
        worst = 0.0
        for eye in cams:
            fwd = (-eye).normalized()                     # orijine bakiyor
            up0 = mathutils.Vector((0.0, 0.0, 1.0))
            right = fwd.cross(up0)
            if right.length < 1e-6:
                right = mathutils.Vector((1.0, 0.0, 0.0))
            right.normalize()
            up = right.cross(fwd).normalized()
            for c in corners:
                d = c - eye
                z = d.dot(fwd)
                if z <= 1e-6:
                    return                                 # kamera obje icinde: dokunma
                worst = max(worst, abs(d.dot(right)) / z, abs(d.dot(up)) / z)
        if worst <= 1e-9:
            return
        s = limit / worst
        if abs(s - 1.0) < 0.005:
            break
        for obj in _scene_root_objects():
            obj.scale = obj.scale * s
        bpy.context.view_layer.update()
    mn, mx = _mesh_bbox()                                  # yeniden merkezle
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
    radius = cp.camera_distance(TARGET_RADIUS, LENS_MM, SENSOR_MM, FILL_FACTOR)
    if NORM_MODE == "fit":
        fit_scale_to_views(args.uid, radius, args.resolution)
    setup_lighting()
    cam = setup_camera()
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
            # kanonik indexler gorunum listesinden turetilir (sema-bagimsiz):
            # ring12 -> [0,1,2,3], sphere20 -> [0..5] (ust/alt dahil)
            "canonical_indices": [v["index"] for v in views_meta
                                  if v.get("role") == "canonical"] or [0, 1, 2, 3],
            "views": views_meta}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("RENDER_OK", args.uid, len(views_meta))


if __name__ == "__main__":
    main()
