"""Goruntu olcekleme -- TEK KAYNAK.

NEDEN AYRI MODUL (2026-09-02):
512'lik master'i denetim/olcum cozunurlugune indirme kodu bu projede BES ayri
yere kopyalanmisti ve hepsi ayni iki hatayi yapiyordu:

  1) ANTI-ALIASING YOK. `F.interpolate(mode="bilinear")` PyTorch'ta varsayilan
     `antialias=False` ile gelir: kucultmede cekirdek kaynak olcegine
     genisletilmez, 8x kucultmede 8x8 blogun sadece 2x2'si orneklenir.
     Olculdu (12 gercek render), |bilinear - area| RMS'in goruntunun TUM
     yuksek frekans enerjisine orani:
         res  64 (8.0x): %72     res 128 (4.0x): %47
         res 192 (2.7x): %73     res 384 (1.3x): %72
         res 256 (2.0x): %0   <- tam 2x'te bilinear zaten kutu ortalamasi
     Takma ad POZA BAGLIDIR (alt-piksel fazi her gorunumde farkli) =>
     gorunumler arasi TUTARSIZ. Hicbir 3B temsil tutarsiz sinyali fit edemez;
     optimize edicinin verecegi en iyi cevap ortalamadir = BULANIK DOKU.

  2) PREMULTIPLY SIRASI. Duz (straight) RGBA kucultulup sonra alpha ile
     carpiliyordu. Blender alpha=0 pikseline RGB=0 yazar (olculdu: 0.0028)
     => siluet kenarina siyah sizar. Kenar pikselleri RMS: res 64'te 0.1099.
     res 64'te obje ~35 piksel genistir; 1-2 piksellik bir bank citasi ya da
     makas kolu TAMAMEN kenar pikselidir.

DUZELTME `dataset.py`'de 2026-09-02'de yapildi ama DIGER DORT CAGRI NOKTASINA
TASINMADI (eval_suite.load_view, bench_overfit.load_view, diag_retrieval.load,
metrics.neighbor_indices) -- yani projenin BIRINCIL kapi olcum araci hatali
GT'ye karsi olcmeye devam ediyordu. Bu, ayni sinif hatanin UCUNCU tekrariydi
(`set_epoch` 3 asamanin 1'inde, `_crop_input` dormant kopya, bu).
Hatirlamaya guvenmek ise yaramadi => kod tek yerde, tekrari
`tests/test_kucultme_dogrulugu.py` icindeki `ast` denetimi engelliyor.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------------------- kesirli kutu
# 2026-09-02, bagimsiz denetim + kendi olcumum: `mode="area"`
# (= adaptive_avg_pool2d) TAM SAYI OLMAYAN olcek oranlarinda dogru alan
# filtresi DEGIL -- kutu sinirlari tam sayiya yuvarlanir, siniri kesen piksel
# bir bin'e tamamen yazilir. Olculdu (512 -> out, rastgele goruntu):
#     out 128 (4.000x) |area-gercek| RMS 0.00000   <- tam sayi: DOGRU
#     out 256 (2.000x)                    0.00000   <- tam sayi: DOGRU
#     out 300 (1.707x)                    0.07866   (ref std 0.136)
#     out 384 (1.333x)                    0.08688   (ref std 0.168)
#     out 448 (1.143x)                    0.10839   (ref std 0.180)
# `fit_teacher --render_low 256 --render_high 512` bu araliktan TAM SAYI
# cekiyor (crop.sample_render_res) => 257 olasiligin sadece 2'si temiz.
# Mekanizma yukaridaki (1) ile AYNI: poza bagli, goruumler arasi tutarsiz HF
# gurultusu => fit edilemez => optimize edicinin cevabi ortalama = BULANIK.
#
# NOT: `bilinear + antialias=True` kesirlide 2-5x iyi ama TAM SAYIDA bozuyor
# (out 128'de 0.03977, out 256'da 0.07421). Yani o da dogru cevap degil.
# Asagidaki agirlik matrisi ikisinde de dogru.
#
# VARSAYILAN DEGISMEDI: `OLCEK_YONTEMI` ortam degiskeni "kutu" yapilmadikca
# davranis birebir eskisi gibi ("area"). A/B'nin tek degiskenli olmasi icin.
_KUTU_ONBELLEK = {}


def _kutu_agirlik(n, out, device, dtype):
    """Kesirli sinirlarla dogru alan (kutu) filtresinin (out,n) agirlik matrisi."""
    anahtar = (n, out, str(device), str(dtype))
    W = _KUTU_ONBELLEK.get(anahtar)
    if W is None:
        s_ = n / out
        W = torch.zeros(out, n, dtype=torch.float64)
        for j in range(out):
            a, b = j * s_, (j + 1) * s_
            for i in range(int(a), min(int(b - 1e-9) + 1, n)):
                W[j, i] = min(b, i + 1) - max(a, i)
        W /= W.sum(1, keepdim=True)
        W = W.to(device=device, dtype=dtype)
        _KUTU_ONBELLEK[anahtar] = W
    return W


def _kutu_kucult(y, res):
    """(N,C,H,W) -> (N,C,res,res), ayrilabilir kesirli kutu filtresi."""
    W = _kutu_agirlik(y.shape[-1], res, y.device, y.dtype)
    y = torch.einsum("ij,ncjw->nciw", W, y)      # yukseklik
    return torch.einsum("ij,nchj->nchi", W, y)   # genislik


# 2026-09-02: VARSAYILAN "kutu" YAPILDI.
# Defekt bulaniklik degil WARP: `area` her cikis pikselinde esit agirlikli sabit
# kutu kullaniyor, dogru filtre agirliklari kaydiriyor. 512->384 icin olculdu --
# ornekleme merkezi kaymasi [+0.25, 0.00, -0.25] px, periyodu 3 olan bir
# geometrik warp. Obje izdusumu goruumler arasi kaydigi icin ayni yuzey noktasi
# her goruumde farkli yonde kayiyor => hicbir 3B temsil fit edemez.
# `OLCEK_YONTEMI=area` OPT-OUT olarak kaldi (eski kollari tekrar uretmek icin).
#
# ETKILENEN/ETKILENMEYEN (olculdu, 512 -> out):
#   512->64/128/256 TAM SAYI oran => |area-kutu| RMS 0.0000000  ==> ETKILENMEZ
#     yani: tum bench_keskinlik tablolari (res 256), train_lrm val prob
#     (render_res 128), teacher_1024_v3 (region yok, res 64).
#   512->224 (INPUT_RES, 2.286x) ve region>0 kesirli r => ETKILENIR.
def _yontem():
    import os
    return os.environ.get("OLCEK_YONTEMI", "kutu")


def kucult(x, res):
    """(...,H,W) tensoru `res`e olcekler. Kucultmede ALAN (kutu) ortalamasi.

    Alpha kanali TASIMAYAN veriler icin (RGB, maske, premultiply edilmis RGB).
    RGBA icin `kucult_rgba` kullan -- orada premultiply sirasi da onemli.
    """
    tek = x.dim() == 3
    y = x[None] if tek else x
    if y.shape[-1] == res and y.shape[-2] == res:
        return x
    if y.shape[-1] > res:
        y = (_kutu_kucult(y, res) if _yontem() == "kutu"
             else F.interpolate(y, size=(res, res), mode="area"))
    else:                      # BUYUTME: takma ad sorunu yok
        y = F.interpolate(y, size=(res, res), mode="bilinear", align_corners=False)
    return y[0] if tek else y


def kucult_rgba(arr, res):
    """(4,H,W) RGBA'yi PREMULTIPLY UZAYINDA, alan ortalamasiyla olcekler.

    Duz (straight) RGB dondurur: cagiranlar `rgba[:3] * rgba[3:4]` yapiyor ve
    premult/alpha bolmesi o carpimi birebir geri verir.
    """
    a = arr[None]
    if a.shape[-1] == res:
        return arr
    if a.shape[-1] < res:      # buyutme: eski davranis korunur
        return F.interpolate(a, size=(res, res), mode="bilinear",
                             align_corners=False)[0]
    _k = (_kutu_kucult if _yontem() == "kutu"
          else (lambda t, r: F.interpolate(t, size=(r, r), mode="area")))
    pre = _k(a[:, :3] * a[:, 3:4], res)
    al = _k(a[:, 3:4], res)
    rgb = (pre / al.clamp(min=1e-4)).clamp(0.0, 1.0)
    return torch.cat([rgb, al], 1)[0]


def yukle_rgba(path, res):
    """PNG -> (4,res,res) duz RGBA. Tum GT okuma yollarinin ortak girisi."""
    im = Image.open(path).convert("RGBA")
    arr = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0
    return kucult_rgba(arr, res)


def pil_premult(im):
    """PIL RGBA'yi premultiply eder (yeniden boyutlandirmadan ONCE).

    PIL'in `resize` filtresi kucultmede zaten anti-alias'lidir (cekirdek kaynak
    olcegine genisletilir), yani PIL yolunda tek hata premultiply SIRASI idi.
    """
    a = np.asarray(im, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return Image.fromarray((a * 255).astype(np.uint8), "RGBA")


def unpremult(arr):
    """(4,H,W) premultiply edilmis tensoru duz RGB'ye cevirir."""
    al = arr[3:4]
    return torch.cat([(arr[:3] / al.clamp(min=1e-4)).clamp(0.0, 1.0), al], 0)
