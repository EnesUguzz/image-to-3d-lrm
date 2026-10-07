"""OLU veri seti korumasi.

2026-09-01: bench_detay.sh --train_list/--renders_dir vermedi, fit_teacher'in
varsayilanlari olu sete bakiyordu, uc kol da olu veride egitildi ve deney
cope gitti. Iki savunma hatti test edilir:
  1) egitim/bench scriptlerinin VARSAYILANLARI canli seti gosterir
  2) olu set yine de okunursa stderr'e yuksek sesle uyari duser
"""
import ast
import io
import os
import sys

import pytest

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOK, "scripts"))

from lrm.dataset import _olu_veri_uyar  # noqa: E402

CANLI_LISTE = "dataset/train_list_v2.json"
CANLI_RENDER = "dataset/renders_opp_score3"

# Faz A araclari (filter_dataset/run_batch/verify_render) render'i URETEN taraf;
# onlarin varsayilani baska bir anlam tasir, kapsam disi.
EGITIM_SCRIPTLERI = [
    "fit_teacher.py", "distill_lrm.py", "train_lrm.py",
    "bench_overfit.py", "overfit_lrm.py", "overfit_randbg.py",
]


def _varsayilanlar(script):
    """add_argument("--x", ..., default="...") ciftlerini toplar."""
    yol = os.path.join(KOK, "scripts", script)
    agac = ast.parse(io.open(yol, encoding="utf-8").read())
    bulunan = {}
    for d in ast.walk(agac):
        if not (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                and d.func.attr == "add_argument"):
            continue
        if not (d.args and isinstance(d.args[0], ast.Constant)):
            continue
        ad = d.args[0].value
        for kw in d.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                bulunan[ad] = kw.value.value
    return bulunan


@pytest.mark.parametrize("script", EGITIM_SCRIPTLERI)
def test_varsayilanlar_canli_seti_gosterir(script):
    v = _varsayilanlar(script)
    for bayrak, beklenen in (("--train_list", CANLI_LISTE),
                             ("--renders_dir", CANLI_RENDER),
                             ("--render_dir", CANLI_RENDER)):
        if bayrak in v:
            assert v[bayrak] == beklenen, (
                f"{script} {bayrak} varsayilani OLU sete bakiyor: {v[bayrak]}")


def test_olu_set_uyari_basar(capsys):
    _olu_veri_uyar("dataset/train_list.json", "dataset/renders")
    hata = capsys.readouterr().err
    assert "OLU VERI SETI OKUNUYOR" in hata
    assert "train_list.json" in hata and "dataset/renders" in hata


def test_canli_set_sessiz(capsys):
    _olu_veri_uyar(CANLI_LISTE, CANLI_RENDER)
    assert capsys.readouterr().err == ""


def test_sondaki_ayrac_uyariyi_kacirmaz(capsys):
    """`dataset/renders/` (sondaki slash) de olu settir."""
    _olu_veri_uyar(CANLI_LISTE, "dataset/renders/")
    assert "OLU VERI SETI OKUNUYOR" in capsys.readouterr().err
