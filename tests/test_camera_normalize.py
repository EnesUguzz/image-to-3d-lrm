"""cameras.canonicalize (LRM'in normalize_camera'si) regresyon testleri.

NEDEN SIMDI (2026-08-27): fonksiyon `dataset.py`'de yaziliydi ama `train_lrm`'e
HIC BAGLANMAMISTI (0 referans) -- yani bugune kadar hic kullanilmadi ve hic
test edilmedi. Blok 3 Tier 1'de aciliyor, once dogru oldugu kanitlanmali.

Kritik ozellik: SAF ROTASYON olmali. Olcek ya da oteleme sizarsa obje orijinden
kayar, `bound` kupunun disina tasar ve NeRF onu maskeler -- sessizce bos sahne.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from lrm import cameras


def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """Orijine bakan bir c2w uret (OpenGL: -Z ileri)."""
    eye = torch.tensor(eye, dtype=torch.float64)
    t = torch.tensor(target, dtype=torch.float64)
    u = torch.tensor(up, dtype=torch.float64)
    f = (t - eye)
    f = f / f.norm()
    r = torch.linalg.cross(f, u)
    if r.norm() < 1e-8:
        r = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    r = r / r.norm()
    u2 = torch.linalg.cross(r, f)
    m = torch.eye(4, dtype=torch.float64)
    m[:3, 0] = r
    m[:3, 1] = u2
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def _ring(n=4, radius=1.4866, elev_deg=20.0):
    el = math.radians(elev_deg)
    out = []
    for i in range(n):
        az = 2 * math.pi * i / n
        out.append(_look_at((radius * math.cos(el) * math.cos(az),
                             radius * math.cos(el) * math.sin(az),
                             radius * math.sin(el))))
    return torch.stack(out)


def test_referans_kamera_identity_rotasyona_gelir():
    cams = _ring()
    out = cameras.canonicalize(cams[0], cams)
    assert torch.allclose(out[0, :3, :3], torch.eye(3, dtype=out.dtype), atol=1e-6), \
        "referans kameranin rotasyonu identity olmali"


def test_saf_rotasyon_mesafeyi_korur():
    """Olcek/oteleme sizmamali: her kameranin orijine uzakligi degismemeli.
    Sizarsa obje bound kupunun disina cikar ve NeRF onu maskeler."""
    cams = _ring(n=6)
    for ref in (cams[0], cams[3]):
        out = cameras.canonicalize(ref, cams)
        d0 = cams[:, :3, 3].norm(dim=-1)
        d1 = out[:, :3, 3].norm(dim=-1)
        assert torch.allclose(d0, d1, atol=1e-6)


def test_rotasyon_matrisleri_ortonormal_kalir():
    cams = _ring(n=5)
    out = cameras.canonicalize(cams[2], cams)
    for R in out[:, :3, :3]:
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=R.dtype), atol=1e-6)
        assert R.det().item() == pytest.approx(1.0, abs=1e-6)


def test_kameralar_arasi_baginti_korunur():
    """Rijit donusum: iki kamera arasindaki aci degismemeli."""
    cams = _ring(n=8)
    out = cameras.canonicalize(cams[1], cams)

    def ang(a, b):
        v1 = a[:3, 3] / a[:3, 3].norm()
        v2 = b[:3, 3] / b[:3, 3].norm()
        return torch.acos((v1 * v2).sum().clamp(-1, 1))

    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            assert ang(cams[i], cams[j]).item() == pytest.approx(
                ang(out[i], out[j]).item(), abs=1e-5)


def test_farkli_referans_farkli_sonuc_verir():
    """Aksi halde fonksiyon fiilen sabit bir donusum uyguluyor demektir."""
    cams = _ring(n=4)
    a = cameras.canonicalize(cams[0], cams)
    b = cameras.canonicalize(cams[1], cams)
    assert not torch.allclose(a, b, atol=1e-3)


def test_ayni_referansla_idempotent():
    """Kanoniklestirilmis kumeyi kendi ilk kamerasiyla tekrar kanoniklestirmek
    hicbir sey degistirmemeli (ilk kamera zaten identity)."""
    cams = _ring(n=4)
    once = cameras.canonicalize(cams[0], cams)
    twice = cameras.canonicalize(once[0], once)
    assert torch.allclose(once, twice, atol=1e-6)


def test_tek_kamerayla_calisir():
    cams = _ring(n=1)
    out = cameras.canonicalize(cams[0], cams)
    assert out.shape == (1, 4, 4)
    assert torch.allclose(out[0, :3, :3], torch.eye(3, dtype=out.dtype), atol=1e-6)


# --------------------------------------------------- GERCEK VERI (denetim bulgusu)
def test_olcekli_c2w_ile_mesafe_korunur():
    """TESTLER BUNU NEDEN KACIRDI (2026-08-27): yukaridaki _look_at() ORTONORMAL
    matris uretiyor. Gercek meta.json ise Blender'dan gelen OBJE-BASINA UNIFORM
    OLCEK tasiyor (olculdu: 12/12 objede, olcek 0.31-893). Sentetik test bu
    kosulu hic gormedi; fonksiyon gercek veride kameralari orijine cekiyordu."""
    cams = _ring(n=6)
    for olcek in (0.0011, 0.4845, 1.0289, 34.7):
        olcekli = cams.clone()
        olcekli[:, :3, :3] *= olcek          # UNIFORM olcek, gercek veridekiyle ayni
        out = cameras.canonicalize(olcekli[0], olcekli)
        d0 = cams[:, :3, 3].norm(dim=-1)
        d1 = out[:, :3, 3].norm(dim=-1)
        assert torch.allclose(d0, d1, atol=1e-5),             "olcek %.4f: yaricap %.4f -> %.4f (near=0.8 icine dusuyor)" % (
                olcek, float(d0[0]), float(d1[0]))


def test_olcekli_c2w_ile_referans_identity():
    cams = _ring(n=4)
    olcekli = cams.clone()
    olcekli[:, :3, :3] *= 0.0011
    out = cameras.canonicalize(olcekli[0], olcekli)
    R = out[0, :3, :3]
    assert torch.allclose(R, torch.eye(3, dtype=R.dtype), atol=1e-6)
    assert R.det().item() == pytest.approx(1.0, abs=1e-6)


def test_gercek_metadan_okunan_kamera():
    """Diskteki gercek bir meta.json ile uctan uca."""
    import json
    import os
    kok = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "dataset", "renders_opp_score3")
    if not os.path.isdir(kok):
        pytest.skip("render dizini yok")
    uid = sorted(os.listdir(kok))[0]
    with open(os.path.join(kok, uid, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    c2w = torch.stack([torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float64))
                       for v in meta["views"]])
    once = c2w[:, :3, 3].norm(dim=-1)
    out = cameras.canonicalize(c2w[0], c2w)
    sonra = out[:, :3, 3].norm(dim=-1)
    assert torch.allclose(once, sonra, atol=1e-5),         "gercek veride yaricap %.4f -> %.4f degisti" % (float(once[0]), float(sonra[0]))
