"""Ortak metrik tanimlari -- eval_suite.py ve train_lrm.py AYNI kodu kullanir.

NEDEN ORTAK MODUL (2026-08-27):
Kapi degerlendirmesi ile egitim ici izleme ayri ayri yazilirsa sessizce ayrisir
(bu projede "iki kol ayni sandigimiz farkli seyi olctu" hatasi iki kez yasandi).
Metrik tanimi TEK yerde durur; her iki taraf da buradan cagirir.

KATMANLAR (bkz. docs/metrik-sartnamesi.md):
  1  goruntu : PSNR, SSIM, LPIPS-AlexNet, CLIP-sim, siluet IoU
  2  cokus   : top-1 retrieval, ortalama-baseline orani, obje-arasi std, acc
  4  cerceve : TAVAN (ogretmen) / TABAN (ortalama-obje) / RAKIP (en-yakin-komsu)

LPIPS AGI -- ALEXNET, VGG DEGIL:
Egitimde LPIPS-VGG kayip fonksiyonunun parcasi. Degerlendirmede de VGG kullanmak
kendi kaybimizi metrik diye raporlamak olurdu. Bagimsizlik icin AlexNet.

DAGILIM, ORTALAMA DEGIL:
Tatarchenko ve ark. (CVPR 2019): ortalama metrik, objelerin bir kisminin
coktugunu gizler. Her metrik ortalama + medyan + p10 (kotu uc) olarak raporlanir.
"""
import numpy as np
import torch
import torch.nn.functional as F

from lrm import imutil

# hi_iyi = buyuk deger daha iyi  ->  kotu uc p10'da
YUKSEK_IYI = ("psnr", "ssim", "clip", "iou", "top1", "fscore", "precision",
              "recall", "normal", "gt_siluet")

_LPIPS = None
_CLIP = None
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def lpips_alex(device="cuda"):
    """Tembel yukleme: agirlik indirme/VRAM maliyeti sadece ilk kullanimda."""
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    return _LPIPS


def clip_model(device="cuda"):
    global _CLIP
    if _CLIP is None:
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        _CLIP = (m.to(device).eval(),
                 torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1),
                 torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1))
    return _CLIP


# --------------------------------------------------------------- katman 1
def psnr_per_object(P, G):
    """P,G: (N,...,3,H,W) -> obje basina PSNR (dB)."""
    dims = tuple(range(1, P.dim()))
    mse = ((P - G) ** 2).mean(dims)
    return -10 * torch.log10(mse.clamp_min(1e-9)), mse


def _gauss(win, sigma, device, dtype):
    c = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    return (g / g.sum())


def ssim_per_object(P, G, win=11, sigma=1.5, data_range=1.0):
    """Torch SSIM (skimage'a CPU turu atmadan). P,G: (N,3,H,W) [0,1]."""
    P, G = P.float(), G.float()
    C = P.shape[1]
    k = _gauss(win, sigma, P.device, P.dtype)
    k2 = (k[:, None] @ k[None, :]).expand(C, 1, win, win).contiguous()
    pad = win // 2

    def flt(x):
        return F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), k2, groups=C)

    mp, mg = flt(P), flt(G)
    mp2, mg2, mpg = mp * mp, mg * mg, mp * mg
    sp = flt(P * P) - mp2
    sg = flt(G * G) - mg2
    spg = flt(P * G) - mpg
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mpg + c1) * (2 * spg + c2)) / ((mp2 + mg2 + c1) * (sp + sg + c2))
    return s.mean((1, 2, 3))


@torch.no_grad()
def lpips_per_object(P, G, device="cuda", chunk=32):
    """AlexNet LPIPS. Girdi [0,1] -> [-1,1]'e cevrilir."""
    net = lpips_alex(device)
    out = []
    for i in range(0, len(P), chunk):
        out.append(net(P[i:i + chunk] * 2 - 1, G[i:i + chunk] * 2 - 1).flatten())
    return torch.cat(out)


@torch.no_grad()
def clip_per_object(P, G, device="cuda"):
    m, mean, std = clip_model(device)

    def emb(x):
        # OLCEK-DENETIMI: buyutme (deger cozunurlugu <= 224; CLIP girisi 224)
        x = F.interpolate(x.float(), size=(224, 224), mode="bicubic", align_corners=False)
        return F.normalize(m.encode_image((x.clamp(0, 1) - mean) / std).float(), dim=-1)

    return (emb(P) * emb(G)).sum(-1)


def mask_iou_per_object(A_pred, A_gt, thresh=0.5):
    """Siluet IoU. A: (N,1,H,W) veya (N,...,1,H,W)."""
    p = (A_pred > thresh).float()
    g = (A_gt > thresh).float()
    dims = tuple(range(1, p.dim()))
    inter = (p * g).sum(dims)
    union = ((p + g) > 0).float().sum(dims).clamp_min(1.0)
    return inter / union


# --------------------------------------------------------------- katman 2
def top1_retrieval(P, G):
    """Her tahmin KENDI hedefine mi en yakin? sans = 1/N.
    Dusukse model girdiyi kullanmiyordur (conditioning testi)."""
    N = len(P)
    flat_dims = tuple(range(2, P.dim() + 1))
    D = ((P[:, None] - G[None]) ** 2).mean(flat_dims)
    return float((D.argmin(1) == torch.arange(N, device=P.device)).float().mean())


def mean_baseline_ratio(P, G):
    """mse / ortalama-obje-baseline. 1.0 = 'herkese ayni blob' cokusu."""
    dims = tuple(range(1, P.dim()))
    mse = ((P - G) ** 2).mean(dims).mean()
    mean_mse = ((P.mean(0, keepdim=True) - G) ** 2).mean()
    return float(mse / mean_mse.clamp_min(1e-9))


def neighbor_indices(inputs, res=32):
    """RAKIP baseline'i icin: her objeye GIRDI goruntusu en cok benzeyen BASKA obje.

    Tatarchenko ve ark. (CVPR 2019): tek-gorunum 3B aglari cogu zaman
    "tani ve en yakin egitim seklini getir" yapiyor. Model bu baseline'i
    yenmiyorsa rekonstruksiyon yaptigi soylenemez.
    """
    X = imutil.kucult(inputs.float(), res)   # 224->32: 7x kucultme, alan ort.
    D = ((X[:, None] - X[None]) ** 2).mean(tuple(range(2, X.dim() + 1)))
    D.fill_diagonal_(float("inf"))
    return D.argmin(1)


# --------------------------------------------------------------- ozet
def summarize(per_obj):
    """{'psnr': tensor/array, ...} -> {'psnr': {'ort','med','p10'|'p90'}}"""
    out = {}
    for k, v in per_obj.items():
        if v is None:
            continue
        v = v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        v = v.astype(np.float64).ravel()
        if not v.size:
            continue
        hi = k.startswith(YUKSEK_IYI)
        out[k] = {"ort": float(v.mean()), "med": float(np.median(v)),
                  ("p10" if hi else "p90"): float(np.percentile(v, 10 if hi else 90))}
    return out


@torch.no_grad()
def image_metrics(P, G, A_pred=None, A_gt=None, device="cuda",
                  use_lpips=True, use_clip=False, use_ssim=True):
    """KATMAN 1'in tamami. P,G: (N,3,H,W) veya (N,V,3,H,W) [0,1].
    Cok gorunumlu girdide gorunumler obje ekseninde duzlestirilir
    (obje basina metrik korunur: her obje kendi gorunumlerinin ortalamasi)."""
    multi = P.dim() == 5
    if multi:
        N, V = P.shape[:2]
        Pf, Gf = P.reshape(N * V, *P.shape[2:]), G.reshape(N * V, *G.shape[2:])
    else:
        N, V, Pf, Gf = len(P), 1, P, G

    def fold(x):                     # (N*V,) -> (N,) obje basina ortalama
        return x.reshape(N, V).mean(1) if multi else x

    psnr, _ = psnr_per_object(P, G)
    out = {"psnr": psnr}
    if use_ssim:
        out["ssim"] = fold(ssim_per_object(Pf, Gf))
    if use_lpips:
        out["lpips"] = fold(lpips_per_object(Pf, Gf, device))
    if use_clip:
        out["clip"] = fold(clip_per_object(Pf, Gf, device))
    if A_pred is not None and A_gt is not None:
        out["iou"] = mask_iou_per_object(A_pred, A_gt)
    return out
