"""Saf kamera poz/geometri matematiği. bpy'ye bağımlı DEĞİL —
hem sistem Python'da (testler) hem Blender gömülü Python'unda import edilir."""
import hashlib
import math
import os
import random

CANONICAL = [(0.0, 20.0), (90.0, 20.0), (180.0, 20.0), (270.0, 20.0)]

# Gorunum semasi. "ring12" = eski (12 supervision, elevation -10..80: objenin ALTI
# hic gorunmez, azimuth boslugu ort 70 derece). "sphere20" = yapili tam-kure + jitter.
VIEW_SCHEME = os.environ.get("RENDER_VIEW_SCHEME", "ring12")

# sphere20 kanonik seti = kupun 6 yuz yonu (on/arka/sag/sol/UST/ALT).
# Olculdu: objelerin %30'unda 4 yan gorunum birbirinin neredeyse ayni (donel simetrik
# objeler: sise, vazo, varil, tekerlek). Orada 4. girdi bilgi katmiyor; ust/alt katiyor.
# Yan gorunumler el=+20: masadaki bir objeyi ceken kullanicinin dogal acisi.
CANONICAL6 = [(0.0, 20.0), (90.0, 20.0), (180.0, 20.0), (270.0, 20.0),
              (0.0, 88.0), (0.0, -88.0)]   # 90 degil: TRACK_TO kutupta dejenere olur

# sphere20 halkalari: (merkez elevation, gorunum sayisi). 3 x 6 = 18 supervision.
_RINGS = [(50.0, 6), (10.0, 6), (-35.0, 6)]
_JITTER_EL = 12.0
_JITTER_AZ = 15.0
_CAP_EL = (78.0, 88.0)   # 90 degil: TRACK_TO kisiti kutupta dejenere olur


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


def sample_supervision_views_sphere(uid):
    """Yapili tam-kure ornekleme + obje-basina jitter.

    Rastgele ornekleme (ring12) kapsamayi GARANTI ETMIYOR: olculdu, objelerin
    sadece %38.5'inde el>70 gorunumu var ve azimuth boslugu ort 70 derece.
    Sabit bir sema (kupun 26 yonu gibi) kapsamayi garantiler ama TUM objeler ayni
    pozlari gorur => model pozu ezberleyebilir (LRM/objaverse rastgele poz kullanir).
    Burada ikisi birlestirilir: halka yapisi kapsamayi garantiler, jitter + halka
    basina rastgele offset pozlari obje basina farklilastirir.
    """
    rng = random.Random(_seed_from_uid(uid) ^ 0x5EED)
    out = []
    for el_c, n in _RINGS:
        base = rng.uniform(0.0, 360.0)                                 # halka offset
        step = 360.0 / n
        for k in range(n):
            az = (base + k * step + rng.uniform(-_JITTER_AZ, _JITTER_AZ)) % 360.0
            out.append((az, el_c + rng.uniform(-_JITTER_EL, _JITTER_EL)))
    return out


def build_view_list(uid, scheme=None):
    scheme = scheme or VIEW_SCHEME
    views = []
    canon = CANONICAL6 if scheme == "sphere20" else CANONICAL
    for i, (az, el) in enumerate(canon):
        views.append({"index": i, "role": "canonical",
                      "azimuth_deg": az, "elevation_deg": el})
    sup = (sample_supervision_views_sphere(uid) if scheme == "sphere20"
           else sample_supervision_views(uid))
    for j, (az, el) in enumerate(sup):
        views.append({"index": len(canon) + j, "role": "supervision",
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
