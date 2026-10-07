"""CHECKPOINT KUNYESI ile DEGERLENDIRME ARASINDAKI UYUM.

2026-09-02 kod incelemesi. Uc ayri sessiz-bozulma yolu bulundu; ucu de
"deneyi cope atan ama hicbir hata vermeyen" sinifta:

B1) `compat.load_lrm` checkpoint'in `render_cfg` kunyesini HIC OKUMUYORDU.
    `LRM(**lrm_kwargs)` kutuphane varsayilanlariyla kuruluyordu:
    density_bias 0.0, bound 0.6, n_samples 96. Bu dort arac onu kullaniyor:
      bench_keskinlik.py (--students), eval_elevation.py,
      eval_photo4.py, kanit_asama12.py
    `density_bias` bir PARAMETRE DEGIL, yogunluk basina eklenen sabit
    (nerf.py:57). `--density_bias 1.0` ile egitilmis bir modeli 0.0 ile
    render etmek TAMAMEN baska bir yogunluk alani verir -- ve state_dict
    uyustugu icin hicbir hata alinmaz. Ustelik `density_bias` bu projede
    UC KEZ yasanan `acc->0` cokusunun panzehiri olarak onerilen deger
    (CLAUDE.md: K2 kolu 15.52 -> 18.42 dB), yani tuzak tam da tavsiye
    edilen ayari kullandigimiz an kurulmus oluyor.

B2) `distill_lrm.py` kunye YAZMIYORDU. Dogrulandi: distilled_v3.pt yalnizca
    {model, opt, sched, step} tasiyor. `train_lrm --init_from` bu durumda
    kunyeyi VARSAYIYOR (train_lrm.py:461) => uyusmazlik kapisi sessizce gecer.

B3) `bench_overfit.py` -- projenin ANA kapi araci -- `--w_lpips` varsayilanini
    2.0'da tutuyordu. CLAUDE.md: 2.0, modeli "ortalama obje" havzasinda
    tutuyor (32 obje: top-1 %25 -> %62, 0.25'e dusurunce). `train_lrm` ve
    `fit_teacher` 0.25 kullaniyor. Yani kapi araci, terk edilmis tarifeyi
    olcuyordu.
"""
import ast
import os
import sys

import pytest

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOK, "scripts"))

from lrm import defaults          # noqa: E402
from lrm.compat import kunye_kwargs, KUNYE_ALANLARI   # noqa: E402


# --------------------------------------------------------------- B1
def test_kunye_ckpt_degerlerini_uygular():
    tk = {"render_cfg": {"density_bias": 1.0, "bound": 0.552, "n_samples": 192}}
    kw, uyari = kunye_kwargs(tk, {})
    assert kw["density_bias"] == 1.0
    assert kw["bound"] == 0.552
    assert kw["n_samples"] == 192
    assert uyari is None


def test_kunye_yoksa_YUKSEK_SESLE_uyarir():
    """Kunyesiz ckpt sessizce varsayilanla yuklenmemeli -- ne varsayildigi yazilmali."""
    kw, uyari = kunye_kwargs({"step": 100}, {})
    assert uyari is not None and "KUNYESIZ" in uyari
    # varsayim zincirin fiili degeri olmali, kutuphane varsayilani DEGIL diye
    # degil, ama ne oldugu ACIKCA raporlanmali:
    assert str(kw["density_bias"]) in uyari
    assert str(kw["bound"]) in uyari


def test_acik_kwargs_kunyeyi_ezer_ama_sessizce_degil():
    """Cagiran bilerek farkli bir deger verebilir; bu sessiz kalmamali."""
    tk = {"render_cfg": {"density_bias": 1.0, "bound": 0.6, "n_samples": 96}}
    kw, uyari = kunye_kwargs(tk, {"n_samples": 320})
    assert kw["n_samples"] == 320          # cagiran kazanir
    assert kw["density_bias"] == 1.0       # digerleri kunyeden
    assert uyari is not None and "n_samples" in uyari


def test_kunye_alanlari_render_sonucunu_degistirenleri_kapsar():
    for alan in ("density_bias", "bound", "n_samples"):
        assert alan in KUNYE_ALANLARI


# --------------------------------------------------------------- B2
def _kaynak(ad):
    with open(os.path.join(KOK, "scripts", ad), encoding="utf-8") as f:
        return f.read()


def test_distill_lrm_kunye_yaziyor():
    """Zincirin 2. asamasi da render kunyesi tasimali; yoksa 3. asamanin
    uyusmazlik kapisi varsayima duser (train_lrm.py:461)."""
    src = _kaynak("distill_lrm.py")
    t = ast.parse(src)
    kayit_var = False
    for node in ast.walk(t):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save"):
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    anahtarlar = {k.value for k in arg.keys
                                  if isinstance(k, ast.Constant)}
                    if "render_cfg" in anahtarlar:
                        kayit_var = True
    assert kayit_var, "distill_lrm.py torch.save cagrilarinda 'render_cfg' yok"


# --------------------------------------------------------------- B3
def _arg_default(ad, bayrak):
    t = ast.parse(_kaynak(ad))
    for node in ast.walk(t):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and a0.value == "--" + bayrak:
                for kw in node.keywords:
                    if kw.arg == "default":
                        return ast.literal_eval(kw.value)
    return None


@pytest.mark.parametrize("ad", ["bench_overfit.py", "bench_triplane_fit.py"])
def test_bench_araclari_terk_edilmis_w_lpips_kullanmiyor(ad):
    """2.0 OLCULEREK terk edildi (top-1 %25 -> %62). Kapi araclarinin
    varsayilani egitimle AYNI olmali; yoksa kapi baska bir tarifeyi olcer."""
    assert _arg_default(ad, "w_lpips") == _arg_default("train_lrm.py", "w_lpips")


def test_egitim_scriptleri_w_lpips_hizasi():
    assert _arg_default("fit_teacher.py", "w_lpips") == \
           _arg_default("train_lrm.py", "w_lpips") == 0.25


# --------------------------------------------------------------------------
# 2026-09-03: `TriplaneHead` iki tabani destekliyor (k2s2 ortusmesiz varsayilan,
# k4s2 ortusmeli). `mimari_tespit` cekirdek boyutunu OKUMUYORDU => k4s2 bir
# checkpoint k2s2 olarak kurulup `load_state_dict`te sekil uyusmazligiyla
# patlardi. Ve bu, SAATLERCE suren egitimin ARDINDAN, olcum adiminda olurdu.
def test_head_tabani_state_dictten_tespit_ediliyor(tmp_path):
    import torch
    from lrm.model import LRM
    from lrm import compat

    class _Sahte(torch.nn.Module):
        embed_dim, patch, n_patch = 384, 14, 256

        def forward(self, x):
            return torch.zeros(x.shape[0], self.n_patch, self.embed_dim)

    kw = dict(dim=64, depth=1, heads=2, triplane_res=8, triplane_ch=4)
    for tip, beklenen_k in (("k2s2", 2), ("k4s2", 4)):
        m = LRM(encoder=_Sahte(), head_tip=tip, **kw)
        p = str(tmp_path / f"{tip}.pt")
        torch.save({"model": m.state_dict(), "step": 1}, p)
        m2, arch, _ = compat.load_lrm(p, device="cpu", encoder=_Sahte(), **kw)
        assert arch["head_tip"] == tip, f"{tip} tespit edilemedi: {arch}"
        assert m2.triplane_head.up.weight.shape[2] == beklenen_k


def test_iki_head_tabani_ayni_cikti_olceginde_basliyor():
    """Head init olcegi bu projede UC KEZ acc->0 cokusune yol acti
    (M_base, M_enc4, K2_bf16). k4s2'nin fan_in'i 4x oldugu icin std
    duzeltilmezse cikti olcegi 2x saparadi ve A/B tek degiskenli olmazdi."""
    import torch
    from lrm.triplane import TriplaneHead
    g = torch.randn(3, 32, 32, 512)
    stds = {}
    for tip in ("k2s2", "k4s2"):
        torch.manual_seed(0)
        with torch.no_grad():
            stds[tip] = float(TriplaneHead(512, 32, 2, tip=tip)(g).std())
    assert abs(stds["k2s2"] - stds["k4s2"]) < 0.03, stds
    for tip, v in stds.items():
        assert 0.15 < v < 0.35, (tip, v)   # HEDEF_STD = 0.24


def test_girdi_cozunurlugu_kunyeden_geri_kuruluyor(tmp_path):
    """448'de egitilmis ckpt, 224 varsayilanli bir surecte 448 olarak kurulmali.

    NEDEN (2026-09-03, encoder duvari deneyi kurulurken yakalandi):
    `input_res` `KUNYE_ALANLARI`nda DEGILDI ve olamaz da -- `LRM.__init__`
    kwarg'i degil, SINIF NITELIGI (`LRM.INPUT_RES`). Import aninda
    `defaults.INPUT_RES`ten baglaniyor, yani ortam degiskeni checkpoint'le
    SEYAHAT ETMEZ.

    Uyusmazlik SESSIZ: `model.py:52` Plucker haritasini
    `scale_intrinsics(K, INPUT_RES, side)` ile kurar; `side` patch sayisindan
    (gercek girdi) gelir, `INPUT_RES` surecin varsayilanindan. Sekil hatasi
    YOKTUR -- sadece baska bir model olculur, hem de saatlerce suren
    egitimin ARDINDAN, olcum adiminda.
    """
    import torch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from lrm import compat, defaults
    from lrm.model import LRM

    m = LRM(n_samples=defaults.N_SAMPLES)
    yol = str(tmp_path / "sahte_448.pt")
    torch.save({"model": m.state_dict(), "step": 1,
                "render_cfg": {"density_bias": 0.0, "noise_std": 0.0,
                               "bound": defaults.BOUND,
                               "n_samples": defaults.N_SAMPLES,
                               "input_res": 448}}, yol)

    yuklenen, arch, _ = compat.load_lrm(yol, device="cpu")
    assert int(yuklenen.INPUT_RES) == 448, (
        f"kunyede 448 yaziyordu, model {yuklenen.INPUT_RES} ile kuruldu")
    assert arch["input_res"] == 448, arch
    # sinif niteligi KIRLENMEMELI: ayni surecte baska bir ckpt 224 kalmali
    assert int(LRM.INPUT_RES) == int(defaults.INPUT_RES)


def test_kunyesiz_ckpt_surecin_varsayilanini_alir(tmp_path):
    """Kunyesiz (eski format) ckpt icin INPUT_RES surecin varsayilani olmali."""
    import torch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from lrm import compat, defaults
    from lrm.model import LRM

    m = LRM(n_samples=defaults.N_SAMPLES)
    yol = str(tmp_path / "kunyesiz.pt")
    torch.save({"model": m.state_dict(), "step": 1}, yol)
    yuklenen, arch, _ = compat.load_lrm(yol, device="cpu")
    assert int(yuklenen.INPUT_RES) == int(defaults.INPUT_RES)
    assert arch["input_res"] == int(defaults.INPUT_RES)
