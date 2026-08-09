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
