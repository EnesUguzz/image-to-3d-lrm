"""GLB dis aktarim ekseni (glTF Y-up) regresyon testi.

NEDEN (2026-08-27): extract_mesh dunya koordinatlarini (Z-up, render pipeline'imizin
konvansiyonu) donusumsuz .glb yaziyordu. glTF standardi Y-UP oldugu icin Blender ve
three.js objeyi 90 derece yan yatiriyordu. Faz C'de her obje donuk gorunurdu.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
trimesh = pytest.importorskip("trimesh")
import extract_mesh as EM


def _dik_kutu():
    """Z ekseninde uzun bir kutu: Z-up dunyada 'dik duran' obje."""
    return trimesh.creation.box(extents=(0.1, 0.1, 1.0))


def test_dis_aktarimda_uzun_eksen_Y_olur(tmp_path):
    m = _dik_kutu()
    assert np.argmax(m.extents) == 2, "kaynak mesh Z'de uzun olmali"
    p = tmp_path / "a.glb"
    EM.export_glb(m, str(p))
    back = trimesh.load(str(p), force="mesh")
    assert np.argmax(back.extents) == 1, \
        "glTF'te en uzun eksen Y olmali (Y-up), bulunan eksen %d" % np.argmax(back.extents)


def test_kaynak_mesh_degismez(tmp_path):
    """Donusum KOPYADA yapilmali; yoksa silhouette_iou gibi olcumler bozulur."""
    m = _dik_kutu()
    once = np.asarray(m.vertices).copy()
    EM.export_glb(m, str(tmp_path / "b.glb"))
    assert np.allclose(np.asarray(m.vertices), once)


def test_donusum_saf_rotasyon(tmp_path):
    """Olcek/yansima sizmamali: hacim ve kenar uzunluklari korunmali."""
    m = _dik_kutu()
    p = tmp_path / "c.glb"
    EM.export_glb(m, str(p))
    back = trimesh.load(str(p), force="mesh")
    assert sorted(np.round(back.extents, 6)) == sorted(np.round(m.extents, 6))
    assert back.volume == pytest.approx(m.volume, rel=1e-4)


def test_saga_yatmiyor_yukari_kaliyor(tmp_path):
    """(x,y,z)->(x,z,-y) dogru yon: +Z'deki tepe noktasi +Y'ye gitmeli."""
    m = _dik_kutu()
    p = tmp_path / "d.glb"
    EM.export_glb(m, str(p))
    back = trimesh.load(str(p), force="mesh")
    assert back.bounds[1][1] == pytest.approx(m.bounds[1][2], abs=1e-6)


# --------------------------------------------------- sRGB -> lineer (renk)
def test_srgb_lineer_bilinen_degerler():
    assert EM.srgb_to_linear(0.0) == pytest.approx(0.0)
    assert EM.srgb_to_linear(1.0) == pytest.approx(1.0)
    # orta gri: sRGB 0.5 -> lineer ~0.214
    assert float(EM.srgb_to_linear(0.5)) == pytest.approx(0.2140, abs=1e-3)


def test_renkler_dis_aktarimda_koyulasir(tmp_path):
    """sRGB -> lineer donusumu orta tonlari DUSURMELI; yoksa tuketici ikinci kez
    aydinlatir ve renk yikanir (olculdu: 0.66 -> ekranda 0.84)."""
    m = _dik_kutu()
    n = len(m.vertices)
    m.visual.vertex_colors = np.tile(np.array([[169, 157, 175, 255]], np.uint8), (n, 1))
    p = tmp_path / "e.glb"
    EM.export_glb(m, str(p))
    back = trimesh.load(str(p), force="mesh")
    out = np.asarray(back.visual.vertex_colors)[:, :3].mean(0)
    assert out.max() < 130, "orta ton lineer uzayda dusmeliydi, bulunan %s" % out
    # 169/255 = 0.663 -> lineer 0.398 -> 101
    assert out[0] == pytest.approx(101, abs=2)


def test_alfa_bozulmuyor(tmp_path):
    m = _dik_kutu()
    n = len(m.vertices)
    m.visual.vertex_colors = np.tile(np.array([[200, 100, 50, 255]], np.uint8), (n, 1))
    p = tmp_path / "f.glb"
    EM.export_glb(m, str(p))
    back = trimesh.load(str(p), force="mesh")
    assert np.asarray(back.visual.vertex_colors)[:, 3].min() == 255


# --------------------------------------------------- Taubin pruzsuzlestirme
def test_puruzsuzlestirme_kapaliyken_dokunmuyor():
    m = _dik_kutu()
    assert np.allclose(EM.puruzsuzlestir(m, 0).vertices, m.vertices)


def test_taubin_hacim_korumasi_kaba_meshte_devreye_giriyor():
    """lamb/nu dengesi MESH YOGUNLUGUNA bagli: olculdu, 5 iterasyon 158k ucgenli
    mesh'te %-0.7, 12 ucgenli kutuda %-99.97 (obje yok oluyor). Koruma olmadan
    Faz C'de kaba bir mesh sessizce buzulurdu."""
    for m in (trimesh.creation.box(extents=(0.1, 0.1, 1.0)),
              trimesh.creation.icosphere(subdivisions=3, radius=0.4)):
        s = EM.puruzsuzlestir(m, 15)
        assert abs(s.volume / m.volume - 1) <= 0.05 + 1e-9,             "koruma calismadi: hacim %%%.1f degisti" % (100 * (s.volume / m.volume - 1))


def test_taubin_yogun_meshte_calismaya_devam_ediyor():
    """Koruma, ISE YARAYAN durumu engellememeli: yogun mesh puruzsuzlesmeli."""
    m = trimesh.creation.icosphere(subdivisions=5, radius=0.4)   # ~20k ucgen
    s = EM.puruzsuzlestir(m, 5)
    assert not np.allclose(np.asarray(s.vertices), np.asarray(m.vertices)),         "yogun meshte hic pruzsuzlestirme yapilmadi"


def test_taubin_yuksek_frekansi_siliyor():
    """Asil iddia: kuresel bir yuzeye eklenen gurultu azalmali."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    rng = np.random.default_rng(0)
    n = np.asarray(m.vertex_normals)
    gurultulu = m.copy()
    gurultulu.vertices = np.asarray(m.vertices) + n * rng.normal(0, 0.004, (len(m.vertices), 1))
    def sapma(x):
        return float(np.std(np.linalg.norm(np.asarray(x.vertices), axis=1)))
    assert sapma(EM.puruzsuzlestir(gurultulu, 15)) < sapma(gurultulu) * 0.7


def test_kaynak_mesh_bozulmuyor():
    m = _dik_kutu()
    once = np.asarray(m.vertices).copy()
    EM.puruzsuzlestir(m, 10)
    assert np.allclose(np.asarray(m.vertices), once)
