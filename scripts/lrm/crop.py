"""Bolge kirpma (region crop) -- OpenLRM/TripoSR tarzi denetim.

NEDEN VAR:
Bizim objeler kareyi ortalama %9 kapliyor => render kaybinin ~%85'i ARKA PLAN
hakkinda (bkz. docs/lrm-blob-diagnosis.md 5a). Tam kareyi render etmek hem pahali
hem sinyal-fakiri. Referanslar bunu su sekilde cozuyor:
  OpenLRM : render cozunurlugu her ornekte U[64,192], sonra 64x64 BOLGE kirpilir
  TripoSR : 512'lik render'dan 128x128 rastgele yama, "on plani kapsayan
            kirpmalarin secilme olasiligi artirilarak"
Isin butcesi sabit kalir (region^2), ama kirpma icindeki OBJE ORANI cok yukselir.

Ayrica bu, sabit coarse-to-fine faz gecisinin yerine gecer. Olculdu (gece kosusu):
faz gecisi 1.25x hiz kazandirdi (koşunun %11'i) ama mask kaybini 2x sicratti ve
toparlanmasi 540 adim surdu; ustelik kaba fazin 8000 adiminda top-1 hic sanstan
yukari cikmadi.

KRITIK: kirpma intrinsic'i kaydirir. Yanlis kaydirma her seyi SESSIZCE bozar
(bu projede ayni sinifta hatalar yasandi) => tests/test_region_crop.py isin
denkligini birebir dogruluyor.
"""
import torch


def crop_K(K, ax, ay):
    """Kirpma intrinsic'i: odak ayni kalir, ana nokta capaya gore kayar."""
    Kc = K.clone()
    Kc[0, 2] = Kc[0, 2] - float(ax)
    Kc[1, 2] = Kc[1, 2] - float(ay)
    return Kc


def scale_and_crop_K(K_master, master_res, r, ax, ay):
    """master -> r olcekle, sonra (ax, ay) capasindan kirp."""
    from lrm import cameras
    return crop_K(cameras.scale_intrinsics(K_master, master_res, r), ax, ay)


def crop_image(img, ax, ay, region):
    """(..., H, W) tensoru (ax, ay) capasindan region x region kirpar."""
    return img[..., ay:ay + region, ax:ax + region]


def alpha_bbox(alpha, thresh=0.05):
    """Alpha'nin dolu bolgesinin (x0, y0, x1, y1) sinirlari. Bos ise None."""
    a = alpha
    while a.dim() > 2:
        a = a.amax(0)
    m = a > thresh
    if not bool(m.any()):
        return None
    ys = torch.nonzero(m.any(dim=1), as_tuple=False).flatten()
    xs = torch.nonzero(m.any(dim=0), as_tuple=False).flatten()
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def sample_anchor(alpha, r, region, rng, fg_bias=0.75):
    """Kirpma capasi sec. fg_bias olasilikla kirpma ON PLANI kesmek ZORUNDA.

    alpha : (…, r, r) hedef alpha ya da None (o zaman duzgun rastgele)
    r     : render cozunurlugu
    region: kirpma boyutu
    Doner : (ax, ay)
    """
    hi = r - region
    if hi <= 0:
        return 0, 0
    if alpha is None or rng.random() >= fg_bias:
        return rng.randint(0, hi), rng.randint(0, hi)

    bb = alpha_bbox(alpha)
    if bb is None:                       # dejenere/bos hedef: kilitlenme
        return rng.randint(0, hi), rng.randint(0, hi)
    x0, y0, x1, y1 = bb
    # kirpmanin bbox'i kesmesi icin capa araligi:
    #   ax in [x1 - region, x0]  (ve [0, hi] icine kirpilir)
    ax_lo, ax_hi = max(0, x1 - region), min(hi, max(0, x0))
    ay_lo, ay_hi = max(0, y1 - region), min(hi, max(0, y0))
    if ax_lo > ax_hi:
        ax_lo = ax_hi = min(hi, max(0, (x0 + x1) // 2 - region // 2))
    if ay_lo > ay_hi:
        ay_lo = ay_hi = min(hi, max(0, (y0 + y1) // 2 - region // 2))
    return rng.randint(ax_lo, ax_hi), rng.randint(ay_lo, ay_hi)


def sample_render_res(rng, low, high, region):
    """Ornek basina render cozunurlugu: U[low, high], region'dan kucuk olamaz."""
    low = max(low, region)
    high = max(high, low)
    return rng.randint(low, high)
