"""Gercek kullanici fotosunu egitim dagitimina sokar.

Faz C'nin girdi hattinin ta kendisi. Egitim render'lari `fit` cerceveleme
konvansiyonuyla uretildi; bir telefon fotosu bu konvansiyona sokulmadan
modele verilirse model DAGITIM DISI bir girdi gorur.

Konvansiyon 150 egitim objesinin kanonik[0] goruumunden OLCULDU:
  bbox en buyuk kenar / kare = 0.549 (medyan), merkez (0.499, 0.507)

⚠️ SIKI KIRPMA YOK. 2026-08-27 A/B'si sonucu: objenin karedeki buyuklugu
gercek olceginin ipucu; siki kirpma conditioning'i artirsa da genellemeyi
bozuyor (held-out top-1 %35.9 -> %23.4). Bu yuzden hedef oran egitimin
MEDYANI, "objeyi kareye doldur" degil.
"""
import argparse, os
import numpy as np
from PIL import Image

TARGET_EXTENT = 0.549   # olculdu: egitim kanonik[0] bbox en buyuk kenar / kare


def segment(path):
    """rembg ile alfa matte. Donen: RGBA uint8 numpy."""
    from rembg import remove
    im = Image.open(path).convert("RGB")
    out = remove(im)                      # RGBA
    return np.array(out.convert("RGBA"))


def largest_component(mask):
    """En buyuk bagli bileseni birak (sehpa/golge artiklarini atar)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def fit_frame(rgba, out_size=512, target=TARGET_EXTENT, keep_largest=True):
    """Alfa bbox'ini olcup egitim cercevelemesine oturtur."""
    a = rgba[..., 3] > 10
    if keep_largest:
        a = largest_component(a)
        rgba = rgba.copy()
        rgba[..., 3] = np.where(a, rgba[..., 3], 0)
    if not a.any():
        raise SystemExit("alfa bos -- segmentasyon basarisiz")
    ys, xs = np.where(a)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = Image.fromarray(rgba[y0:y1, x0:x1])

    # en buyuk kenar hedef orana gelsin
    scale = (target * out_size) / max(crop.width, crop.height)
    nw, nh = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
    # PREMULTIPLY SIRASI (2026-09-02): duz (straight) RGBA'yi yeniden
    # boyutlandirmak siluet kenarinda hale birakir -- rembg alpha=0 altinda
    # ORIJINAL fotografin arka plan rengini biraktigi icin hale RENKLIDIR.
    # dataset.py'de ayni hata bugun duzeltildi; burasi Faz C'nin girdi yolu ve
    # gorevi tam da egitim dagilimini yeniden uretmek => hale dogrudan
    # dagilim uyusmazligi demek. Kompozitleme dogrusaldir: premultiply uzayinda
    # olceklenir, sonra duz renge donulur.
    _a = np.asarray(crop, dtype=np.float32) / 255.0
    _a[..., :3] *= _a[..., 3:4]
    crop = Image.fromarray((_a * 255).astype(np.uint8), "RGBA").resize(
        (nw, nh), Image.LANCZOS)
    _b = np.asarray(crop, dtype=np.float32) / 255.0
    _al = np.clip(_b[..., 3:4], 1e-4, None)
    _b[..., :3] = np.clip(_b[..., :3] / _al, 0.0, 1.0)
    crop = Image.fromarray((_b * 255).astype(np.uint8), "RGBA")

    canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    canvas.paste(crop, ((out_size - nw) // 2, (out_size - nh) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="ham foto (jpg/png)")
    ap.add_argument("--out", required=True, help="cikti RGBA png")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--target", type=float, default=TARGET_EXTENT)
    ap.add_argument("--keep_all", action="store_true",
                    help="bagli bilesen filtresini KAPAT")
    a = ap.parse_args()

    rgba = segment(a.inp)
    fitted = fit_frame(rgba, a.size, a.target, keep_largest=not a.keep_all)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fitted.save(a.out)

    m = np.array(fitted)[..., 3] > 10
    ys, xs = np.where(m)
    ext = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) / a.size
    print(f"-> {a.out}  kaplama {m.mean():.4f}  bbox-oran {ext:.4f} (hedef {a.target})")


if __name__ == "__main__":
    main()
