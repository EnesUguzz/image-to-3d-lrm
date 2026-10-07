"""Bir egitim kosusunu TEK SAYFADA okunur hale getirir: egriler + onizleme seridi.

NEDEN (2026-08-27): 30.800 adimlik bir kosu ~30 val satiri, ~30 onizleme PNG'si ve
binlerce adim logu birakiyor. Bunlari elle okumak "ne zaman ne oldu"yu gizliyor.
Kullanicinin sozleri: "deneydeki sonuclari iyi gozlemlersek deney bi sonuca ulasacak".

Uretilen:
  <out>/egriler.png   -- PSNR (TABAN ve RAKIP cizgileriyle), oran, top-1, acc & std
  <out>/onizleme.png  -- ayni objelerin adim adim degisimi (zaman serisi olarak)
  <out>/ozet.md       -- ilk/son/en iyi degerler, kapi durumu, uyarilar

Kullanim:
  python scripts/kosu_raporu.py --tag TAM_asama3
"""
import argparse
import glob
import io
import json
import os

import numpy as np

LOG_DIR = "dataset/lrm_logs"
PREVIEW_DIR = "dataset/lrm_val_previews"

# Yon: True = yuksek iyi
YON = {"psnr": True, "ssim": True, "top1": True, "iou": True, "clip": True,
       "lpips": False, "ratio": False}


def satirlari_oku(tag, jsonl, stamp=""):
    """val_metrics.jsonl'dan SADECE bu kosunun satirlarini al.

    Dosya TUM kosular tarafindan append ediliyor; tag/stamp ile ayirmazsak
    farkli kosularin egrileri ic ice gecer (fiilen yasandi)."""
    out = []
    if not os.path.isfile(jsonl):
        return out
    with io.open(jsonl, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            try:
                d = json.loads(satir)
            except json.JSONDecodeError:
                continue
            if stamp:
                # ESKI KOSULAR: jsonl'a `tag` alani 2026-08-27'de eklendi.
                # Oncesindeki satirlar sadece config_hash tasiyor.
                if d.get("stamp") != stamp:
                    continue
            elif tag:
                if d.get("tag") != tag:
                    continue
            out.append(d)
    out.sort(key=lambda d: d["step"])
    # Ayni adim birden fazla kez varsa (resume) SONUNCUSU gecerli
    tekil = {}
    for d in out:
        tekil[d["step"]] = d
    return [tekil[k] for k in sorted(tekil)]


def egriler(kayitlar, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.array([d["step"] for d in kayitlar])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # --- PSNR: TEK BASINA YORUMLANAMAZ, taban ve rakip cizgileriyle ---
    a = ax[0][0]
    a.plot(s, [d["psnr"] for d in kayitlar], "-o", ms=3, label="model")
    if "taban_psnr" in kayitlar[0]:
        a.axhline(np.mean([d["taban_psnr"] for d in kayitlar]), ls="--", c="gray",
                  label="TABAN (ortalama goruntu)")
    if "komsu_psnr" in kayitlar[0]:
        a.axhline(np.mean([d["komsu_psnr"] for d in kayitlar]), ls=":", c="darkred",
                  label="RAKIP (en yakin komsu)")
    a.set_title("PSNR (held-out)"); a.set_xlabel("adim"); a.set_ylabel("dB")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # --- oran: 1.0 = blob cokusu ---
    a = ax[0][1]
    a.plot(s, [d["ratio"] for d in kayitlar], "-o", ms=3, c="tab:orange")
    a.axhline(1.0, ls="--", c="red", label="1.0 = ortalama-blob")
    a.axhline(0.75, ls=":", c="green", label="0.75 = on-kayitli kapi")
    a.set_title("mse / ortalama-baseline  (DUSUK iyi)"); a.set_xlabel("adim")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # --- top-1: conditioning ---
    a = ax[1][0]
    a.plot(s, [100 * d["top1"] for d in kayitlar], "-o", ms=3, c="tab:green")
    n = kayitlar[0].get("n", 64)
    a.axhline(100.0 / n, ls="--", c="red", label="sans (%.1f%%)" % (100.0 / n))
    a.axhline(800.0 / n, ls=":", c="green", label="8x sans = on-kayitli kapi")
    a.set_title("top-1 retrieval (conditioning)"); a.set_xlabel("adim"); a.set_ylabel("%")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # --- cokus dedektoru ---
    a = ax[1][1]
    a.plot(s, [d.get("acc_mean", np.nan) for d in kayitlar], "-o", ms=3, label="acc ort")
    a.plot(s, [d.get("inter_std", np.nan) for d in kayitlar], "-s", ms=3,
           label="objeler arasi std")
    a.axhline(0.010, ls="--", c="red", label="std esigi 0.010")
    a.set_title("cokus dedektoru"); a.set_xlabel("adim")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def onizleme_seridi(tag, path, en_fazla=6):
    """Ayni onizleme karesinin adim adim degisimi -- 'ne zaman duzeldi/bozuldu'."""
    from PIL import Image, ImageDraw
    dosyalar = sorted(glob.glob(os.path.join(PREVIEW_DIR, tag, "val_*.png")))
    if not dosyalar:
        return None
    if len(dosyalar) > en_fazla:                    # esit araliklarla sec
        idx = np.linspace(0, len(dosyalar) - 1, en_fazla).round().astype(int)
        dosyalar = [dosyalar[i] for i in idx]
    imgs = [Image.open(f).convert("RGB") for f in dosyalar]
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + 8 * (len(imgs) - 1)
    tuval = Image.new("RGB", (w, h + 22), "white")
    d = ImageDraw.Draw(tuval)
    x = 0
    for f, im in zip(dosyalar, imgs):
        tuval.paste(im, (x, 22))
        d.text((x + 4, 6), "adim " + os.path.basename(f)[4:-4].lstrip("0"), fill="black")
        x += im.width + 8
    tuval.save(path)
    return path


def ozet(kayitlar, tag, path):
    ilk, son = kayitlar[0], kayitlar[-1]
    L = ["# Kosu raporu — `%s`" % tag, "",
         "%d val noktasi, adim %d → %d" % (len(kayitlar), ilk["step"], son["step"]), ""]
    L += ["| metrik | ilk | son | en iyi | en iyi adim |", "|---|---|---|---|---|"]
    for k in ("psnr", "ratio", "top1", "ssim", "lpips", "iou"):
        if k not in ilk:
            continue
        v = np.array([d.get(k, np.nan) for d in kayitlar], dtype=float)
        if np.all(np.isnan(v)):
            continue
        i = int(np.nanargmax(v) if YON.get(k, True) else np.nanargmin(v))
        L.append("| %s | %.4f | %.4f | **%.4f** | %d |"
                 % (k, v[0], v[-1], v[i], kayitlar[i]["step"]))
    L += ["", "## Referans cizgileri", ""]
    if "taban_psnr" in son:
        L.append("- TABAN (ortalama goruntu): **%.2f dB**" % son["taban_psnr"])
    if "komsu_psnr" in son:
        L.append("- RAKIP (en yakin komsu): **%.2f dB**" % son["komsu_psnr"])
    # ------------------------------------------------------------------
    # KAPILAR IKI KATMAN (2026-08-29, bagimsiz denetim sonrasi yeniden yazildi)
    #
    # Eski tablonun DORDU DE cokus kapisiydi: siluet + obje-basina ortalama renk
    # basan dejenere bir cozum dordunu de geciyordu. "4/4 GECTI" bu yuzden
    # "basarili" degil "cokmedi" demekti. Kalite ekseni (bulaniklik, detay,
    # kuyruk) hic olculmuyordu -- tam da modelin basarisiz oldugu eksen.
    #
    # KATMAN 1 (cokus): kosunun saglikli olup olmadigi. Gecmek ZORUNLU ama YETMEZ.
    # KATMAN 2 (kalite): gercekten iyilesme var mi. Karar bunlara bakilarak verilir.
    #   - lpips  : PSNR bulaniklasmaya KOR, LPIPS degil
    #   - iou    : siluet/geometri
    #   - taban_alti : ortalamanin sakladigi kuyruk -- kac obje ortalama-baseline'dan
    #                  kotu (olculdu: %28, ortalama PSNR bunu hic gostermiyordu)
    #   - psnr_p10   : en kotu dilim
    # ------------------------------------------------------------------
    n = son.get("n", 64)
    cokus = [("oran <= 0.75", son.get("ratio", 9) <= 0.75, "%.3f" % son.get("ratio", float("nan"))),
             ("top-1 >= 8x sans", son.get("top1", 0) >= 8.0 / n,
              "%.1f%% (sans %.1f%%)" % (100 * son.get("top1", 0), 100.0 / n)),
             ("TABAN'i gecti", son.get("psnr", 0) > son.get("taban_psnr", 99),
              "%.2f vs %.2f dB" % (son.get("psnr", 0), son.get("taban_psnr", float("nan")))),
             ("RAKIP'i gecti", son.get("psnr", 0) > son.get("komsu_psnr", 99),
              "%.2f vs %.2f dB" % (son.get("psnr", 0), son.get("komsu_psnr", float("nan"))))]
    L += ["", "## Kapi 1 -- COKUS (gecmesi zorunlu, YETMEZ)", ""]
    L += ["| kapi | durum | deger |", "|---|---|---|"]
    for ad, gecti, deger in cokus:
        L.append("| %s | %s | %s |" % (ad, "✅ GECTI" if gecti else "❌ KALDI", deger))

    # KALITE esikleri: TAM_asama3 taban cizgisine gore ON-KAYITLI
    #   lpips 0.167 -> 0.145 | iou 0.662 -> 0.70 | taban_alti 0.28 -> 0.20
    ta = son.get("taban_alti")
    # OLCULMEMIS metrik "basarisiz" gibi gosterilmemeli: `val_metrics(full=False)`
    # lpips hic uretmiyor ve sentinel 9 "9.0000" diye basiliyordu -- olculmemis
    # bir seyi sert bir basarisizlik gibi okutan rapor, bu projenin tam da
    # kacinmaya calistigi yanilticiliktir.
    def _kapi(ad, deger, iyi_mi, ref, bicim="%.4f"):
        if deger is None:
            return (ad, None, "OLCULMEDI", ref)
        return (ad, iyi_mi(deger), bicim % deger, ref)

    kalite = [_kapi("lpips <= 0.145", son.get("lpips"),
                    lambda v: v <= 0.145, "TAM_asama3: 0.167"),
              _kapi("iou >= 0.70", son.get("iou"),
                    lambda v: v >= 0.70, "TAM_asama3: 0.662"),
              ("taban-alti obje <= %20",
               None if ta is None else (ta <= 0.20),
               ("%.1f%%" % (100 * ta)) if ta is not None else "OLCULMEDI",
               "TAM_asama3: ~%28"),
              _kapi("psnr_p10 >= 17.5 dB", son.get("psnr_p10"),
                    lambda v: v >= 17.5, "en kotu %10 dilim", "%.2f dB")]
    L += ["", "## Kapi 2 -- KALITE (karar bunlara bakilir)", ""]
    L += ["| kapi | durum | deger | referans |", "|---|---|---|---|"]
    for ad, gecti, deger, ref in kalite:
        # gecti is None => metrik HIC OLCULMEDI. "KALDI" yazmak, olculmemis bir
        # seyi basarisizlik gibi okutur (fiilen oluyordu: lpips sentinel 9).
        durum = "— OLCULMEDI" if gecti is None else ("✅ GECTI" if gecti else "❌ KALDI")
        L.append("| %s | %s | %s | %s |" % (ad, durum, deger, ref))
    olculen = [g for _, g, _, _ in kalite if g is not None]
    gecen = sum(1 for g in olculen if g)
    if olculen and gecen == len(olculen) and len(olculen) == len(kalite):
        L += ["", "> ⚠️ KALITE kapilarinin HEPSI gecti. Bu ya gercek bir sicrama,"
              " ya da esikler hala gevsek. Ikincisini eleyin: onizlemelere ve"
              " geometri (F@0.01) sayilarina bakin."]

    bayrakli = [d for d in kayitlar if d.get("flags")]
    L += ["", "## Uyarilar", ""]
    if bayrakli:
        L.append("**%d val noktasinda cokus bayragi:**" % len(bayrakli))
        for d in bayrakli[:10]:
            L.append("- adim %d: %s (acc=%.4f std=%.4f)"
                     % (d["step"], ",".join(d["flags"]), d.get("acc_mean", float("nan")),
                        d.get("inter_std", float("nan"))))
    else:
        L.append("Cokus bayragi yok.")

    v = np.array([d["psnr"] for d in kayitlar])
    if len(v) > 4 and np.argmax(v) < len(v) - 3:
        L += ["", "⚠️ **PSNR ortada tepe yapip dustu** (en iyi adim %d, son adim %d). "
              "`snap_*` arsiv checkpointlerinden en iyisini kullanmayi dusun."
              % (kayitlar[int(np.argmax(v))]["step"], son["step"])]
    io.open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="kosu etiketi (train_lrm --tag)")
    ap.add_argument("--jsonl", default=os.path.join(LOG_DIR, "val_metrics.jsonl"))
    ap.add_argument("--stamp", default="",
                    help="tag yerine config_hash ile sec (2026-08-27 oncesi kosular "
                         "jsonl'a tag yazmiyordu)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join("dataset/lrm_raporlar", a.tag)
    os.makedirs(out, exist_ok=True)

    kayitlar = satirlari_oku(a.tag, a.jsonl, a.stamp)
    if not kayitlar:
        raise SystemExit("'%s' etiketli val kaydi yok: %s" % (a.tag, a.jsonl))
    print("%d val noktasi (adim %d -> %d)"
          % (len(kayitlar), kayitlar[0]["step"], kayitlar[-1]["step"]))

    egriler(kayitlar, os.path.join(out, "egriler.png"))
    print("->", os.path.join(out, "egriler.png"))
    p = onizleme_seridi(a.tag, os.path.join(out, "onizleme.png"))
    print("->", p or "(onizleme bulunamadi)")
    ozet(kayitlar, a.tag, os.path.join(out, "ozet.md"))
    print("->", os.path.join(out, "ozet.md"))


if __name__ == "__main__":
    main()
