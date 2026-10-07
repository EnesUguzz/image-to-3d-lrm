"""RENDER KUCULTME DOGRULUGU -- 512'lik master'i denetim cozunurlugune indirme.

2026-09-02, keskinlik teshisi. `_load_rgba` iki ayri hata yapiyordu ve ikisi
de TAM OLARAK keskinligin yasadigi bandi bozuyordu:

1) ANTI-ALIASING YOK. `F.interpolate(mode="bilinear")` PyTorch'ta varsayilan
   `antialias=False` ile gelir: kucultmede cekirdek KAYNAK olcegine
   genisletilmez, yani 8x kucultmede 8x8'lik bir blogun sadece 2x2'si
   ornekleniyordu. Olculdu (12 obje):
       res  64 (8.0x): |bilinear - area| RMS 0.0209 = goruntunun TUM yuksek
                       frekans enerjisinin %72'si
       res 128 (4.0x): %47      res 192: %73      res 384: %31
       res 256 (2.0x): %0  <-- tam 2x'te bilinear zaten kutu ortalamasi
   Bu gurultu poza baglidir (alt-piksel fazi her gorunumde farkli) => gorunumler
   arasi TUTARSIZ, yani hicbir 3B temsil onu fit edemez. Optimize edicinin
   tutarsiz yuksek frekansa verecegi en iyi cevap ORTALAMASI'dir: bulanik doku.
   Uretim ogretmeni res 64'te egitildi; olcum res 256'da yapildi -- yani modelin
   gordugu tek cozunurluk, bozulmanin en yuksek oldugu cozunurluktu.

2) PREMULTIPLY SIRASI TERS. RGBA duz (straight) halde kucultuluyor, premultiply
   SONRA yapiliyordu. Alpha=0 pikselinde Blender RGB=0 (siyah, olculdu: 0.0028)
   yaziyor => siluet kenarinda siyah sizip koyu hale uretiyor. Olculdu (kenar
   pikselleri, RMS):  res 64: 0.1099  res 128: 0.0672  res 256: 0.0058
   res 64'te obje sadece ~35 piksel genis; 1-2 piksellik bir bank citasi ya da
   makas kolu TAMAMEN kenar pikselidir => rengi siyaha cekilir, alpha'si ezilir.
   "Ince yapilar kayboluyor / kalinlasiyor" sikayetinin dogrudan mekanizmasi.

DOGRUSU: alpha ile carp (premultiply) -> alan (box) ortalamasiyla kucult ->
alpha'ya bol. Kompozitleme dogrusaldir, dolayisiyla premultiply UZAYINDA
ortalama almak dogru cevaptir.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOK, "scripts"))

from lrm.dataset import _load_rgba  # noqa: E402
from lrm import imutil  # noqa: E402


def _satranc_rgba(n=512, kare=2):
    """Yuksek frekansli desen + merkezde dolu alpha diski.

    kare=2 piksellik satranc: 8x kucultmede anti-aliasing'siz ornekleme
    bunu tamamen yanlis alir, dogru kucultme duz griye goturur.
    """
    yy, xx = np.mgrid[0:n, 0:n]
    d = (((xx // kare) + (yy // kare)) % 2).astype(np.float32)
    rgb = np.stack([d, 1.0 - d, d], -1)
    a = (((xx - n / 2) ** 2 + (yy - n / 2) ** 2) < (n * 0.35) ** 2).astype(np.float32)
    rgb = rgb * a[..., None]          # alpha=0 -> RGB=0 (Blender'in yazdigi gibi)
    return (np.concatenate([rgb, a[..., None]], -1) * 255).astype(np.uint8)


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "000.png"
    Image.fromarray(_satranc_rgba(), "RGBA").save(p)
    return str(p)


def _dogru(png_yolu, res):
    """Referans: premultiply -> KESIRLI KUTU alan ortalamasi -> (premult, alpha).

    2026-09-02: bu fonksiyon `F.interpolate(mode="area")` kullaniyordu, yani
    test edilen uygulamanin ta kendisini. DONGUSELDI ve gercek defekti
    (kesirli oranlarda area dogru alan filtresi degil) yapisal olarak
    yakalayamiyordu -- `res=192` (oran 2.667, gercek hata %19) yesil geciyordu.
    Referans artik `_kesirli_kutu`, yani sartname.
    """
    a = torch.from_numpy(np.array(Image.open(png_yolu).convert("RGBA"))
                         ).float().permute(2, 0, 1)[None] / 255.0
    pre = _kesirli_kutu(a[:, :3] * a[:, 3:4], res)[0]
    al = _kesirli_kutu(a[:, 3:4], res)[0]
    return pre, al


@pytest.mark.parametrize("res", [64, 128, 192])
def test_kucultme_takma_ad_uretmez(png, res):
    """8x/4x kucultmede yuksek frekans DUZGUN sonumlenmeli (takma ad degil).

    2 piksellik satrancin 64'e indirilmis hali neredeyse duz olmali; anti-alias
    olmadan sozde-rastgele bir desen kalir.
    """
    rgba = _load_rgba(png, res)
    pre_ref, al_ref = _dogru(png, res)
    pre = rgba[:3] * rgba[3:4]
    ic = (al_ref[0] > 0.99)
    assert ic.sum() > 50
    hata = float(((pre - pre_ref)[:, ic] ** 2).mean().sqrt())
    assert hata < 0.02, f"res {res}: obje ICI kucultme hatasi {hata:.4f} (takma ad)"


@pytest.mark.parametrize("res", [64, 128])
def test_premultiply_sirasi_kenarda_hale_uretmez(png, res):
    """Siluet kenari: duz uzayda kucultup sonra premultiply etmek siyah hale uretir."""
    rgba = _load_rgba(png, res)
    pre_ref, al_ref = _dogru(png, res)
    pre = rgba[:3] * rgba[3:4]
    kenar = (al_ref[0] > 0.02) & (al_ref[0] < 0.98)
    assert kenar.sum() > 20
    hata = float(((pre - pre_ref)[:, kenar] ** 2).mean().sqrt())
    assert hata < 0.02, f"res {res}: KENAR premultiply hatasi {hata:.4f} (siyah hale)"


def test_alpha_dogru_kucultulur(png):
    """Alpha kanali da alan ortalamasi olmali (maske kaybi bunun uzerinden olculur)."""
    rgba = _load_rgba(png, 64)
    _, al_ref = _dogru(png, 64)
    assert float((rgba[3:4] - al_ref).abs().max()) < 0.02


def test_buyutme_ve_ayni_cozunurluk_bozulmaz(png):
    """Kucultme disinda davranis degismemeli: res == kaynak -> birebir."""
    rgba = _load_rgba(png, 512)
    ham = torch.from_numpy(np.array(Image.open(png).convert("RGBA"))
                           ).float().permute(2, 0, 1) / 255.0
    assert float((rgba - ham).abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# TEKRAR ENGELI -- bu hata bu projede UC KEZ ayni sekilde tekrarlandi:
#   1) `set_epoch` duzeltmesi 3 asamanin yalnizca 1'ine uygulandi
#   2) premultiply duzeltmesi `_crop_input` kopyasina tasinmadi
#   3) kucultme duzeltmesi `dataset.py`'de yapildi ama eval_suite.py,
#      bench_overfit.py, diag_retrieval.py ve metrics.py'de kaldi --
#      yani BIRINCIL kapi olcum araci bozuk GT'ye karsi olcmeye devam etti.
# Hatirlamaya guvenmek ise yaramadi. Kural artik testle zorlanir.
# ---------------------------------------------------------------------------
import ast    # noqa: E402
import glob   # noqa: E402

SCRIPTS = os.path.join(KOK, "scripts")
# Olcekleme TEK KAYNAK: lrm/imutil.py. Baska yerde bilinear/bicubic
# `F.interpolate` yasak -- cunku kucultme yonunde ANTI-ALIAS'siz calisirlar.
IZINLI_DOSYA = os.path.join("lrm", "imutil.py")
MUAFIYET_ISARETI = "# OLCEK-DENETIMI: buyutme"


def _interpolate_cagrilari(src):
    t = ast.parse(src)
    for node in ast.walk(t):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "interpolate"):
            continue
        mod = {kw.arg: kw.value for kw in node.keywords}
        m = mod.get("mode")
        if not (isinstance(m, ast.Constant) and m.value in ("bilinear", "bicubic")):
            continue
        aa = mod.get("antialias")
        if isinstance(aa, ast.Constant) and aa.value is True:
            continue
        yield node.lineno


def test_olcekleme_tek_kaynakta():
    ihlal = []
    for p in glob.glob(os.path.join(SCRIPTS, "**", "*.py"), recursive=True):
        if p.endswith(IZINLI_DOSYA):
            continue
        with open(p, encoding="utf-8") as f:
            src = f.read()
        satirlar = src.splitlines()
        for ln in _interpolate_cagrilari(src):
            onceki = satirlar[ln - 2] if ln >= 2 else ""
            if MUAFIYET_ISARETI in onceki:
                continue
            ihlal.append(f"{os.path.relpath(p, KOK)}:{ln}")
    assert not ihlal, (
        "Anti-alias'siz bilinear/bicubic F.interpolate bulundu. Kucultmede bu, "
        "goruntunun yuksek frekans enerjisinin %47-%72'si kadar poza-bagli "
        "takma ad gurultusu uretir (bkz. lrm/imutil.py). `imutil.kucult` / "
        "`imutil.kucult_rgba` kullan, ya da gercekten BUYUTME ise ustune "
        f"'{MUAFIYET_ISARETI}' yorumunu koy.\n  " + "\n  ".join(ihlal))


# ---------------------------------------------------------------------------
# 2026-09-02, BAGIMSIZ DENETIM: yukaridaki `_dogru` DONGUSEL.
# Referans olarak `F.interpolate(mode="area")` kullaniyor, yani uygulamayi
# kendisiyle karsilastiriyor -- sartnameyle degil. Bu yuzden asagidaki gercek
# defekti YAKALAYAMAZ ve `res=192` (oran 2.667) parametresi yesil geciyor.
#
# DEFEKT: `mode="area"` = `adaptive_avg_pool2d`, ve TAM SAYI OLMAYAN olcek
# oranlarinda dogru alan filtresi DEGILDIR -- kutu sinirlari tam sayiya
# yuvarlanir, siniri kesen piksel bir bin'e tamamen yazilir.
#
#   64->32 (2.000x)  |area - gercek| RMS 0.00000     temiz
#   64->48 (1.333x)  |area - gercek| RMS 0.08742     sinyalin std'si 0.168
#   64->56 (1.143x)  |area - gercek| RMS 0.10737     sinyalin std'si 0.179
#
# NEDEN ONEMLI: `fit_teacher --render_low 256 --render_high 512` bu araliktan
# TAM SAYI cekiyor (crop.sample_render_res) -- 257 olasiligin sadece 2'si
# (256, 512) temiz oran veriyor. Gercek render'larda olculen hata/HF enerjisi
# ~0.59. Bu, 2026-09-02 sabahi duzeltilen `antialias=False` defektinden
# (%47-72) BUYUK. Ve mekanizma ayni: bin sinirlari goruntu koordinatlarinda
# sabit, obje izdusumu goruumler arasi kayiyor => poza bagli, tutarsiz HF
# gurultusu => optimize edicinin cevabi ortalama = bulanik.


def _kesirli_kutu(x, out):
    """SARTNAME referansi: kesirli sinirlarla dogru alan (kutu) filtresi.

    `mode="area"`'dan bagimsiz -- agirlikli matris elle kuruluyor, yani
    uygulamayi sartnameyle karsilastiriyor. x: (..., n, n).
    """
    n = x.shape[-1]
    s = n / out
    W = torch.zeros(out, n, dtype=torch.float64)
    for j in range(out):
        a, b = j * s, (j + 1) * s
        for i in range(int(a), min(int(b - 1e-9) + 1, n)):
            W[j, i] = min(b, i + 1) - max(a, i)
    W /= W.sum(1, keepdim=True)
    return (W @ x.double() @ W.T).to(x.dtype)


@pytest.mark.parametrize("out", [32, 16])
def test_tam_sayi_oranda_area_dogru(out):
    """Kontrol kolu: tam bolen oranlarda `area` gercek kutu filtresine ESIT.

    Bu test gecmezse asagidaki xfail'in teshisi yanlis demektir (sorun
    kesirli oran degil, baska bir sey).
    """
    torch.manual_seed(0)
    x = torch.rand(64, 64)
    a = F.interpolate(x[None, None], size=(out, out), mode="area")[0, 0]
    assert torch.allclose(a, _kesirli_kutu(x, out), atol=1e-6)


@pytest.mark.parametrize("out", [48, 24])
def test_kesirli_oranda_kucultme_dogru(out):
    """Kesirli olcek oranlarinda da dogru alan filtresi olmali.

    2026-09-02: bu test once `xfail(strict=True)` ile BILINEN DEFEKT olarak
    yazildi (`mode="area"` kesirli oranlarda kutu sinirlarini tam sayiya
    yuvarliyor). `imutil` varsayilani gercek kesirli kutu filtresine
    gecirilince XPASS verdi ve strict=True marker'i kaldirmaya zorladi --
    tam da bunun icin oyle yazilmisti.
    """
    torch.manual_seed(0)
    x = torch.rand(64, 64)
    a = imutil.kucult(x[None, None], out)[0, 0]
    ref = _kesirli_kutu(x, out)
    hata = float(((a - ref) ** 2).mean().sqrt())
    assert hata < 0.02 * float(ref.std()), (
        f"64->{out} (oran {64/out:.3f}): RMS {hata:.5f}, ref std {float(ref.std()):.5f}")
