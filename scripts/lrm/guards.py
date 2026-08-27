"""Dejenerasyon dedektoru: kosu 'cokmus' cikti uretiyorsa isaretler.

NEDEN VAR (2026-08-26 teshisi):
Gece tarife matrisinde `M_base` ve `M_enc4`'un IKISI DE yogunlugu sifira cokertip
tamamen beyaz cikti uretmisti (`acc -> 0` olunca renderer bg_color'i aynen basar).
Iki kol da ayni sogurucu duruma dustugu icin karsilastirma hicbir bilgi tasimiyordu,
ama sonuc tablosuna "taban tarife" olarak girdi ve `overnight.sh`'in kazanan
secicisi cop veriyle calisti. 7 saatlik tam-veri kosusu bu yuzden taban tarifeyle
gitti.

ESIK KALIBRASYONU (dataset/lrm_bench/*.png uzerinde olculdu):
    cokmus : M_base 0.0000   M_enc4 0.0000   AB_ESKI 0.0001
    saglam : val_015500 0.0382   A_baseline 0.0452   AB_YENI 0.0520
             M_batch8 0.0584   O1_oracle 0.0723   M_crop08 0.0881
Aradaki bosluk ~400x => esik 0.010'a guvenle konur.

Kullanim:
    rep = guards.collapse_flags(preds=P, acc_mean=acc.mean().item(),
                                psnr_history=son_psnr_listesi)
    logger.info(guards.format_flags(rep))
    if rep["degenerate"]: ...  # siralamaya sokma / kosuyu durdur
"""

EMPTY_COLLAPSE = "EMPTY_COLLAPSE"    # acc ~ 0: model bos sahne uretiyor
MEAN_COLLAPSE = "MEAN_COLLAPSE"      # tum objeler icin ayni cikti (girdi kullanilmiyor)
FROZEN_OUTPUT = "FROZEN_OUTPUT"      # ardisik degerlendirmelerde PSNR kipirdamiyor

# Varsayilan esikler. Degistirmeden once yukaridaki kalibrasyona bak.
ACC_MIN = 0.01          # saglikli kosuda acc.mean() ~ 0.09-0.15 (kaplama ~ %9)
INTER_STD_MIN = 0.010   # olculen bosluk: 0.0001 (cokmus) <-> 0.0382 (saglam)
FROZEN_TOL = 0.01       # dB
FROZEN_N = 3            # bu kadar ardisik degerlendirme ayni kalirsa donmus say


def _to_float(x):
    return float(x.detach().item() if hasattr(x, "detach") else x)


def inter_object_std(preds):
    """Objeler arasi piksel-bazinda standart sapma.

    preds: (N, ...) -- ilk eksen obje. Tum objeler ayni ciktiyi veriyorsa 0 doner.
    N < 2 ise tanimsiz => None.
    torch.Tensor ya da numpy.ndarray kabul eder.
    """
    if preds is None:
        return None
    n = preds.shape[0]
    if n < 2:
        return None
    if hasattr(preds, "detach"):                      # torch
        return float(preds.detach().float().std(dim=0).mean().item())
    import numpy as np                                # numpy
    return float(np.asarray(preds, dtype="float64").std(axis=0).mean())


def is_degenerate_std(inter_std, inter_std_min=INTER_STD_MIN):
    """Objeler arasi std cokme bandinda mi."""
    return inter_std is not None and inter_std < inter_std_min


def _frozen(psnr_history, tol=FROZEN_TOL, n=FROZEN_N):
    if not psnr_history or len(psnr_history) < n:
        return False
    tail = [float(v) for v in psnr_history[-n:]]
    return (max(tail) - min(tail)) <= tol


def collapse_flags(preds=None, acc_mean=None, psnr_history=None,
                   acc_min=ACC_MIN, inter_std_min=INTER_STD_MIN,
                   frozen_tol=FROZEN_TOL, frozen_n=FROZEN_N):
    """Cokme raporu dondurur.

    preds        : (N, ...) obje basina tahmin (opsiyonel)
    acc_mean     : ortalama birikmis alfa (opsiyonel)
    psnr_history : son degerlendirmelerin PSNR listesi (opsiyonel)

    Doner: {"flags": [...], "degenerate": bool, "inter_std": float|None,
            "acc_mean": float|None}
    Sadece EMPTY/MEAN cokusleri "degenerate" sayar; FROZEN_OUTPUT tek basina
    uyaridir (kisa plato normal olabilir) ama loga mutlaka yazilir.
    """
    flags = []
    istd = inter_object_std(preds)
    acc = None if acc_mean is None else _to_float(acc_mean)

    if acc is not None and acc < acc_min:
        flags.append(EMPTY_COLLAPSE)
    if is_degenerate_std(istd, inter_std_min):
        flags.append(MEAN_COLLAPSE)
    if _frozen(psnr_history, frozen_tol, frozen_n):
        flags.append(FROZEN_OUTPUT)

    degenerate = bool({EMPTY_COLLAPSE, MEAN_COLLAPSE} & set(flags))
    return {"flags": flags, "degenerate": degenerate,
            "inter_std": istd, "acc_mean": acc}


def format_flags(report):
    """Log satiri icin tek satirlik ozet."""
    parts = []
    if report.get("acc_mean") is not None:
        parts.append(f"acc={report['acc_mean']:.4f}")
    if report.get("inter_std") is not None:
        parts.append(f"obj_std={report['inter_std']:.4f}")
    if report.get("flags"):
        parts.append(("DEJENERE! " if report.get("degenerate") else "uyari: ")
                     + ",".join(report["flags"]))
    else:
        parts.append("saglikli")
    return "[guard] " + " ".join(parts)
