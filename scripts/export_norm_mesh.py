"""Blender icinde calisir: GT mesh'i RENDER ILE BIREBIR AYNI normalizasyonla disa aktarir.

    blender -b -P scripts/export_norm_mesh.py -- --object_path X.glb --uid U --out Y.ply

NEDEN AYRI BIR SCRIPT (2026-08-27):
Geometri metrikleri (F-score, Chamfer, Normal Consistency) tahmin edilen mesh'i
GERCEK mesh'e karsi olcer. Ama render pipeline'i objeyi normalize ediyor:
remove_floor_planes -> normalize_scene -> fit_scale_to_views (5 yinelemeli
izdusum oturtma). Bu donusum uygulanmadan karsilastirma yapmak ANLAMSIZ olur:
tahmin normalize cerceve icinde, GT ise glb'nin rastgele olceginde durur.

Donusumu trimesh'te YENIDEN YAZMAK yerine BLENDER'DA AYNI FONKSIYONLARI cagiriyoruz.
Boylece tanim geregi ayni; iki uygulamanin sessizce ayrismasi imkansiz.
(Bu projede "iki kol sessizce farkli veri okudu" hatasi iki kez yasandi.)

Isik/kamera KURULMAZ -> sahnede sadece obje mesh'leri kalir, hepsi disa aktarilir.
"""
import argparse
import os
import sys

import bpy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import camera_poses as cp
import render_object as ro


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--object_path", required=True)
    p.add_argument("--uid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=512)
    a = p.parse_args(argv)

    ro.reset_scene()
    ro.load_glb(a.object_path)
    ro.remove_floor_planes()
    ro.normalize_scene()
    radius = cp.camera_distance(ro.TARGET_RADIUS, ro.LENS_MM, ro.SENSOR_MM,
                                ro.FILL_FACTOR)
    if ro.NORM_MODE == "fit":
        ro.fit_scale_to_views(a.uid, radius, a.resolution)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print("EXPORT_FAIL", a.uid, "mesh yok")
        return
    for o in bpy.context.scene.objects:
        o.select_set(o.type == "MESH")
    bpy.context.view_layer.objects.active = meshes[0]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    try:                                   # Blender 4.x
        bpy.ops.wm.ply_export(filepath=a.out, export_selected_objects=True,
                              apply_modifiers=True, export_normals=True,
                              export_uv=False, export_colors="NONE",
                              ascii_format=False)
    except AttributeError:                 # 3.x geri uyum
        bpy.ops.export_mesh.ply(filepath=a.out, use_selection=True,
                                use_normals=True, use_uv_coords=False,
                                use_colors=False)
    print("EXPORT_OK", a.uid, os.path.getsize(a.out))


if __name__ == "__main__":
    main()
