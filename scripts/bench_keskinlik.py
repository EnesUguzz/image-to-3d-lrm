"""KESKINLIK OLCUMU -- PSNR'in gizledigi bandi olcer.

NEDEN VAR (2026-08-31): `val_rel = 0.2624` "ogrenilebilir sinyalin %97'si"
diyordu ama gorselde taneli kaya duz gri kutleye cokuyordu. PSNR ve rel
dusuk frekansin (siluet + ortalama renk) hakimiyetindedir; keskinligi
olcmez. Bu betik dogrudan doku bandini olcer.

METRIK: yuksek frekans enerjisi = std(x - 3x3 ortalama(x)), SADECE obje
icinde (siluet 7px asindirilir). Siluet kenari devasa bir basamak
kenaridir ve maskelenmezse metrigi tamamen domine eder -- ilk olcumumde
tam bu oldu, ogretmen "GT'nin %99'u" cikti (yanlis).

REFERANS CIZGILERI: GT'yi r piksele indirip geri buyutmek. "Denetim r
px'te yapilsaydi tek goruntuden en fazla bu kadari tasinirdi" tavani.
Ogretmen bu cizginin USTUNDEyse coklu-gorunum birlestirmesi calisiyor
demektir (fiilen oyle: 64px cizgisi %15, ogretmen %45).
"""
import argparse, math, os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm.dataset import LRMDataset
from lrm.nerf import TriplaneNeRF
from lrm.renderer import volume_render_chunked
from lrm.triplane import sample_triplane
from lrm import defaults, cameras, compat, imutil

DEV = "cuda"


def hf_vek(x, m):
    """Maske icindeki yuksek frekans BILESENLERI (vektor)."""
    k = torch.ones(3, 1, 3, 3, device=x.device) / 9.0
    b = F.conv2d(x[None], k, padding=1, groups=3)[0]
    return (x - b)[:, m].reshape(-1)


def hf_ici(x, m):
    """ENERJI: ne kadar cok yuksek frekans var."""
    d = hf_vek(x, m)
    return float(d.std()) if d.numel() > 100 else float("nan")


def hf_kor_kaydirmali(x, gt, m, k):
    """En iyi +-k piksel kaymada korelasyon.

    NEDEN: ham korelasyon HIZALAMAYA asiri duyarli. Dokusu DOGRU ama
    1-2 piksel kaymis bir rekonstruksiyon da ~0 verir. Kaydirmali deger
    ham degerden cok buyukse sorun TEMSIL degil GEOMETRI/HIZALAMA'dir --
    bu ikisinin caresi tamamen farkli.
    """
    best = float("-inf")
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            v = hf_kor(torch.roll(x, (dy, dx), dims=(1, 2)), gt, m)
            if v == v and v > best:
                best = v
    return best if best > float("-inf") else float("nan")


def hf_kor(x, gt, m):
    """YAPI: yuksek frekans DOGRU YERDE mi (GT ile korelasyon).

    NEDEN SART: enerji tek basina yaniltiyor. 2026-08-31 taramasinda
    `res64/tp128` kolu GT'nin %109'u kadar enerji uretti ama gorselde
    bu dikey cizgi artefaktiydi -- gurultu de "keskinlik" gibi olculur.
    Korelasyon gurultuyu ~0'a, dogru dokuyu 1'e yaklastirir.
    """
    a_, b_ = hf_vek(x, m), hf_vek(gt, m)
    if a_.numel() < 100:
        return float("nan")
    a_ = a_ - a_.mean(); b_ = b_ - b_.mean()
    d = float(a_.norm() * b_.norm())
    return float(torch.dot(a_, b_)) / d if d > 0 else float("nan")


def bant_vek(x, m, k1, k2):
    """ORTA BANT: blur_k1 - blur_k2 (maske icinde, vektor).

    NEDEN EKLENDI (2026-09-02, bagimsiz denetim S3):
    `hf_vek` 3x3 kutu kalintisi ⇒ 256px'te 1.5-3 piksellik yapiyi olcuyor.
    Ama kullanicinin sikayeti "sandalye kalinlasmis / bank citasi yok" --
    bu 4-10 piksellik ORTA BANT ve mevcut metrik onu HIC olcmuyordu.
    Ustelik `asindir(px=7)` 15px'ten ince her yapiyi zaten disliyor, yani
    sikayetin konusu iki kez birden metrigin disinda kaliyordu.
    """
    def blur(t, k):
        ker = torch.ones(3, 1, k, k, device=t.device) / (k * k)
        return F.conv2d(t[None], ker, padding=k // 2, groups=3)[0]
    return (blur(x, k1) - blur(x, k2))[:, m].reshape(-1)


def kor_genel(a_, b_):
    """Iki vektor arasi korelasyon (ortalama cikarilmis). NaN guvenli."""
    if a_.numel() < 100:
        return float("nan")
    a_ = a_ - a_.mean(); b_ = b_ - b_.mean()
    d = float(a_.norm() * b_.norm())
    return float(torch.dot(a_, b_)) / d if d > 0 else float("nan")


def eslesmis_fark(x, y):
    """ESLESMIS fark: ort(x-y) ve %95 GA. Objeler kollar arasi ORTAK oldugu
    icin bu bedava ve eslesmemis ortalamalardan cok daha duyarli.

    NEDEN SART (2026-09-02): E-R kapisi `kor` farki +0.010 ile "gecti" ama
    o esik hata payi HESAPLANMADAN yazilmisti ve arac yalnizca `nanmean`
    basiyordu. n=24'te +0.010'un gurultu olup olmadigi BILINMIYORDU.
    """
    d = np.array([u - v for u, v in zip(x, y)
                  if u == u and v == v], dtype=float)
    if d.size < 2:
        return float("nan"), float("nan"), int(d.size)
    sem = float(d.std(ddof=1) / np.sqrt(d.size))
    return float(d.mean()), 1.96 * sem, int(d.size)


def asindir(alpha, px=7):
    """Siluet kenarini metrikten cikar: alpha=1 bolgesini px kadar asindir."""
    m = (alpha > 0.99).float()[None, None]
    return (-F.max_pool2d(-m, 2 * px + 1, 1, px))[0, 0] > 0.5


def en_yogun_pencere(gt, maske, px):
    """GT dokusunun en yogun oldugu px x px pencereyi bul (maske ici).

    Yakinlastirmayi elle secmek yerine olcuye baglar: karsilastirma hep
    dokunun gercekten oldugu yerden yapilir, duz bir yuzeyden degil.
    """
    k = torch.ones(3, 1, 3, 3, device=gt.device) / 9.0
    d = (gt - F.conv2d(gt[None], k, padding=1, groups=3)[0]).abs().mean(0)
    d = d * maske.to(d.dtype)
    e = F.avg_pool2d(d[None, None], px, 1)[0, 0]
    if e.numel() == 0:
        return 0, 0
    idx = int(torch.argmax(e))
    return int(idx // e.shape[1]), int(idx % e.shape[1])       # (y, x)


def gorsel_yaz(yol, kareler, etiketler, px, olcek=2):
    """Ust panel: tam kare. Alt panel: dokunun en yogun oldugu bolge, buyutulmus."""
    if not kareler:
        return
    R = kareler[0][0][0].shape[-1]
    C, H = len(etiketler), px * olcek
    W = max(R, H)
    im = Image.new("RGB", (C * W, len(kareler) * (R + H + 6) + 18), "white")
    d = ImageDraw.Draw(im)
    for j, t in enumerate(etiketler):
        d.text((j * W + 3, 4), t[:26], fill="black")
    y = 18
    for satir, (cy, cx) in kareler:
        for j, img in enumerate(satir):
            a8 = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            im.paste(Image.fromarray(a8), (j * W, y))
            z = a8[cy:cy + px, cx:cx + px]
            im.paste(Image.fromarray(z).resize((H, H), Image.NEAREST), (j * W, y + R + 2))
        y += R + H + 6
    d.rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], outline="black")
    os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
    im.save(yol)
    print(f"-> {yol}  (ust: tam kare, alt: GT dokusunun en yogun oldugu "
          f"{px}px bolge, {olcek}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", nargs="*", default=[],
                    help="fit_teacher ciktilari (oracle triplane bankasi)")
    ap.add_argument("--students", nargs="*", default=[],
                    help="LRM checkpoint'leri (kendi NeRF'iyle render edilir)")
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--n_obj", type=int, default=12)
    ap.add_argument("--res", type=int, default=256,
                    help="OLCUM cozunurlugu (egitim cozunurlugundan bagimsiz)")
    ap.add_argument("--bant1", type=int, default=3,
                    help="ORTA BANT alt kenari (blur cekirdegi). Ince bant "
                         "3x3 kalinti ~1-3px olcuyor; kullanicinin sikayeti "
                         "(cita/ayak kalinligi) 4-10px bandinda ve daha once "
                         "HIC olculmuyordu.")
    ap.add_argument("--bant2", type=int, default=9,
                    help="ORTA BANT ust kenari. kor_orta = kor(blur3-blur9).")
    ap.add_argument("--erode", type=int, default=7)
    ap.add_argument("--ray_chunk", type=int, default=32768,
                    help="isinlari N'lik parcalara bol. ZORUNLU buyuk izgarada: "
                         "tp256 -> Nyquist n_samples 320, 256^2 isin = 21M nokta; "
                         "sample_triplane tek seferde (21M, 96) float = 8.1 GB "
                         "ayirir ve 16 GB karta sigmaz. Parcalama sonucu "
                         "DEGISTIRMEZ (no_grad + jitter=False; bkz. "
                         "tests/test_render_chunk.py).")
    ap.add_argument("--kaydir", type=int, default=3,
                    help="korelasyonu +-N piksel kaymayla da olc (hizalama ayrimi)")
    ap.add_argument("--n_gorsel", type=int, default=6,
                    help="goruntuye kac obje yazilsin (0 = gorsel yok)")
    ap.add_argument("--zoom_px", type=int, default=96,
                    help="yakinlastirma penceresi (GT dokusu en yogun bolgeden secilir)")
    ap.add_argument("--out", default="dataset/lrm_bench/keskinlik",
                    help="<out>_tam.png ve <out>_zoom.png yazilir")
    a = ap.parse_args()

    R = a.res
    ds = LRMDataset(a.train_list, a.renders_dir, split="train", render_res=R,
                    n_sup=1, augment=False, deterministic=True)

    # GIRDI COZUNURLUGUNE GORE DATASET ONBELLEGI (2026-09-03).
    # Ogrenciler farkli `input_res` ile egitilmis olabilir (encoder duvari
    # deneyi: 224 vs 448). Tek bir `ds` hepsini besleyemez: 448'de egitilmis
    # modele 224 goruntu verilirse patch sayisi 256 olur, `side` 16 cikar ve
    # model HATA VERMEDEN yanlis olculur. Supervision kameralari (`sup_c2w`,
    # `sup_K`) girdi cozunurlugunden BAGIMSIZ, o yuzden GT ve olcum ortak
    # `ds`den gelmeye devam eder -- yalnizca GIRDI dali ayrisir.
    _ds_ires = {ds.input_res: ds}

    def girdi_ds(ires):
        if ires not in _ds_ires:
            _ds_ires[ires] = LRMDataset(
                a.train_list, a.renders_dir, split="train", render_res=R,
                n_sup=1, augment=False, deterministic=True, input_res=ires)
            # AYNI OBJE OLDUGUNU DOGRULA: `deterministic=True` ile gorunum
            # secimi cozunurlukten bagimsiz olmali; olmazsa kollar farkli
            # objeyi/goruntuyu karsilastirirdi.
            assert _ds_ires[ires].uids[:a.n_obj] == ds.uids[:a.n_obj], \
                f"input_res={ires} dataset'i farkli uid sirasi veriyor"
        return _ds_ires[ires]

    def rend(tp, c2w, K, dec, bnd=None, ns=None):
        # bound fit ile AYNI olmali; farkli bound'la fit edilmis triplane
        # baska bir uzayda oturur ve sessizce bozuk render edilir.
        bnd = defaults.BOUND if bnd is None else bnd
        o, d = cameras.rays_from_camera(c2w[0], K[0], R, R)
        o, d = o.to(DEV), d.to(DEV)

        def q(p):
            den, rgb = dec(sample_triplane(tp, p, bound=bnd), p)
            ins = (p.abs().amax(-1, keepdim=True) <= bnd).to(den.dtype)
            return den * ins, rgb
        rgb, acc = volume_render_chunked(
            o, d, defaults.NEAR, defaults.FAR,
            defaults.N_SAMPLES if ns is None else ns, q, bg_color=None,
            chunk=a.ray_chunk, jitter=False)
        rgb = rgb.reshape(R, R, 3).permute(2, 0, 1)
        acc = acc.reshape(R, R, 1).permute(2, 0, 1)
        return (rgb + (1 - acc)).clamp(0, 1)

    def down_up(x, r):
        """GT'yi r'ye indirip R'ye geri buyut = "denetim r px'te olsaydi" tavani."""
        return imutil.kucult(imutil.kucult(x, r), R)   # indir (alan) -> buyut

    # --- sistemleri yukle ---
    sistemler = []   # (etiket, fn(idx, item) -> goruntu)
    for p in a.teachers:
        tk = torch.load(p, map_location="cpu", weights_only=False)
        cfg = tk.get("cfg", {})
        # MIMARI KUNYEDEN OKUNUR. Elle 96/64 yazmak, farkli kapasiteyle fit
        # edilmis bir bankayi sessizce yanlis agla render ettirirdi.
        ch, hid = cfg.get("tp_ch", 32), cfg.get("nerf_hidden", 64)
        bnd0 = float(cfg.get("bound", defaults.BOUND))
        nerf = TriplaneNeRF(in_dim=3 * ch, hidden=hid,
                            layers=cfg.get("nerf_layers", 4),
                            pos_enc=cfg.get("pos_enc", 0),
                            pos_bound=bnd0).to(DEV).eval()
        nerf.load_state_dict(tk["nerf"])
        bnd = float(cfg.get("bound", defaults.BOUND))
        tpr = int(cfg.get("tp_res", 64))
        # NYQUIST TABANI: isin adimi (FAR-NEAR)/N, hucre 2*bound/tp_res.
        # Adim hucreden buyukse triplane'in tasidigi detay ORNEKLEME sirasinda
        # atlanir => buyuk izgarali kol HAKSIZ yere kotu olculur. 2026-08-31'de
        # tam bu hata "triplane 128 fayda etmiyor" sonucunu uretmisti.
        ns_min = int(math.ceil((defaults.FAR - defaults.NEAR) * tpr / (2 * bnd)))
        ns = max(int(cfg.get("n_samples", defaults.N_SAMPLES)), ns_min)
        den = (f"bolge{cfg['region']}<-U[{cfg.get('render_low')},"
               f"{cfg.get('render_high')}]" if cfg.get("region") else
               f"{cfg.get('res', '?')}px")
        et = (f"{os.path.basename(p)[:-3]}  [tp {tpr}^2x{ch}, "
              f"den {den}, ns {ns}, bound {bnd}]")
        assert len(tk["uids"]) >= a.n_obj, f"{p}: sadece {len(tk['uids'])} obje"
        assert ds.uids[:len(tk["uids"])] == tk["uids"], f"{p}: uid sirasi uyusmuyor"
        sistemler.append((et, (lambda tk_, nerf_, b_, ns_: (
            lambda i, it: rend(tk_["triplanes"][i].to(DEV),
                               it["sup_c2w"][:1].to(DEV), it["sup_K"][:1].to(DEV),
                               nerf_, b_, ns_)))(tk, nerf, bnd, ns)))
    for p in a.students:
        mdl, arch, sk = compat.load_lrm(p, device=DEV)
        # D7 (2026-09-02 bagimsiz denetim): burada `rend(...)` `bnd`/`ns`
        # VERILMIYORDU => `defaults.BOUND`/`defaults.N_SAMPLES` kullaniliyor,
        # `compat.load_lrm`in kunyeden okudugu `m_.bound` / `m_.n_samples`
        # SESSIZCE yok sayiliyordu. `--teachers` dali (yukarida) dogru yapiyor;
        # ayni dosyada iki farkli davranis. `bound 0.552` kalibrasyonu
        # uygulandigi an sessizce yanlis render verirdi.
        s_bnd = float(getattr(mdl, "bound", defaults.BOUND))
        s_ns = int(getattr(mdl, "n_samples", defaults.N_SAMPLES))
        # Ogrenci triplane'i daima 64^2 (TriplaneHead upsample=2) => Nyquist
        # tabani ogretmen dalindakiyle ayni kuralla hesaplanir.
        # CEKIRDEK BOYUTUNDAN TURETME HATASI (2026-09-03, kosudan once yakalandi):
        # `up.weight.shape[-1]` k2s2'de 2, k4s2'de 4 -- ama IKISININ DE ciktisi
        # r*stride = 64. Cekirdekten turetseydik k4s2 kolu tp128 sanilir,
        # Nyquist n_samples 128, k2s2 kolu 96 ile render edilirdi
        # => A/B TEK DEGISKENLI OLMAZDI. Belirleyici olan STRIDE.
        _up = mdl.triplane_head.up
        _r = int(getattr(mdl.transformer, "triplane_res", 32))
        s_tpr = int(_up.stride[0]) * _r
        s_ns = max(s_ns, int(math.ceil((defaults.FAR - defaults.NEAR)
                                       * s_tpr / (2 * s_bnd))))
        # GIRDI kunyeden gelen cozunurlukte verilir (compat.load_lrm
        # `mdl.INPUT_RES`i ornek niteligi olarak kurdu). `it` ise ortak
        # `ds`den gelmeye devam eder: supervision kameralari girdi
        # cozunurlugunden bagimsiz.
        s_ires = int(getattr(mdl, "INPUT_RES", defaults.INPUT_RES))
        _gds = girdi_ds(s_ires)
        et = (f"{os.path.basename(p)[:-3]}  [ogrenci, adim {sk.get('step')}, "
              f"tp {s_tpr}^2, ns {s_ns}, bound {s_bnd}, girdi {s_ires}]")
        sistemler.append((et, (lambda m_, b_, ns_, g_: (
            lambda i, it: rend(m_.make_triplane(g_[i]["input_imgs"].to(DEV),
                                                g_[i]["input_c2w"].to(DEV),
                                                g_[i]["input_K"].to(DEV)),
                               it["sup_c2w"][:1].to(DEV), it["sup_K"][:1].to(DEV),
                               m_.nerf, b_, ns_)))(mdl, s_bnd, s_ns, _gds)))

    # --- olc ---
    print(f"\nkeskinlik: {a.n_obj} obje | olcum {R}px | siluet {a.erode}px asindirildi",
          flush=True)
    # referanslar olcum cozunurlugune GORE: R'ye esit bir referans dejenere
    # olur (%100 cikar, hicbir sey soylemez).
    ref_res = [r for r in (R // 2, R // 4) if r >= 16]
    ref = {f"GT {r}->{R} tavani": [] for r in ref_res}
    gt_hf, sys_hf = [], [[] for _ in sistemler]
    gt_orta, alan = [], []
    kareler = []          # gorsel icin: her obje -> [GT, sistem1, ...]
    with torch.no_grad():
        for i in range(a.n_obj):
            it = ds[i]
            gt = it["sup_rgb"][0].to(DEV)
            ham = (it["sup_alpha"][0].to(DEV)[0] > 0.99)
            m = asindir(it["sup_alpha"][0].to(DEV)[0], a.erode)
            # D5/D6: kac piksel hayatta kaldi? Az kalan obje NaN doner ve
            # `nanmean` onu SESSIZCE atardi -- atilanlar tam da INCE objeler.
            alan.append((int(ham.sum()), int(m.sum())))
            gt_hf.append(hf_ici(gt, m))
            gt_orta.append(bant_vek(gt, m, a.bant1, a.bant2))
            for r in ref_res:
                d_ = down_up(gt, r)
                ref[f"GT {r}->{R} tavani"].append(
                    (hf_ici(d_, m), hf_kor(d_, gt, m),
                     hf_kor_kaydirmali(d_, gt, m, a.kaydir),
                     kor_genel(bant_vek(d_, m, a.bant1, a.bant2), gt_orta[-1])))
            satir = [gt.cpu()]
            for j, (_, fn) in enumerate(sistemler):
                img = fn(i, it)
                sys_hf[j].append((hf_ici(img, m), hf_kor(img, gt, m),
                                  hf_kor_kaydirmali(img, gt, m, a.kaydir),
                                  kor_genel(bant_vek(img, m, a.bant1, a.bant2),
                                            gt_orta[-1])))
                satir.append(img.cpu())
            if i < a.n_gorsel:
                kareler.append((satir, en_yogun_pencere(gt, m, a.zoom_px)))

    g = float(np.nanmean(gt_hf))

    # --- D5: KAC OBJE FIILEN OLCULDU? --------------------------------------
    # `nanmean` gecersiz objeleri sessizce atiyordu; tablo "24 obje" diyor,
    # fiilen daha azini olcuyordu. Ve atilanlar INCE objeler, yani sikayetin
    # ta kendisi. Artik katilan obje sayisi ve maskenin ne kadarinin hayatta
    # kaldigi BASILIYOR.
    kucuk = [(h, k) for h, k in alan if k < 100]
    orn = np.array([k / max(h, 1) for h, k in alan], dtype=float)
    print(f"\nmaske: asindirma sonrasi kalan alan ort %{100*orn.mean():.0f} "
          f"(medyan %{100*np.median(orn):.0f}, min %{100*orn.min():.0f}) "
          f"| <100 piksel kalan obje: {len(kucuk)}/{len(alan)}")
    if kucuk:
        print(f"  !! {len(kucuk)} obje OLCUMDEN DUSTU (asindirma {a.erode}px => "
              f"{2*a.erode+1}px'ten ince yapi hayatta kalmaz)")

    def yaz(ad, e, c, ck, cm, n_=None):
        n_s = "" if n_ is None else f"{n_:>4}"
        print(f"{ad:<40} {e:>9.5f} {chr(37)+str(round(100*e/g)):>9} "
              f"{c:>8.3f} {ck:>11.3f} {cm:>9.3f} {n_s}")

    def sut(v, i):
        d = [x[i] for x in v if x[i] == x[i]]
        return (float(np.mean(d)) if d else float("nan")), len(d)

    print(f"\n{'sistem':<40} {'enerji':>9} {'GT orani':>9} "
          f"{'kor':>8} {'kor+kaydir':>11} {'kor_orta':>9}    n")
    print(f"  kor: ince bant (3x3 kalinti, ~1-3px) | kor_orta: {a.bant1}x{a.bant1}"
          f"-{a.bant2}x{a.bant2} bandi (~{a.bant1}-{a.bant2}px, cita/ayak olcegi)")
    print(f"  kor+kaydir: +-{a.kaydir}px kaymaya izin verilirse")
    print("  !! ENERJI TEK BASINA YANILTIR (gurultu de enerjidir); karari `kor` ver.")
    print("  !! `FAYDALI` sutunu KALDIRILDI: oran x kor, enerji FAZLALIGINI")
    print("     odullendiriyordu -- K256 GT'nin 128px referansindan 'faydali'")
    print("     cikmisti, ki imkansiz.")
    print("-" * 96)
    yaz("GT (gercek)", g, 1.0, 1.0, 1.0)
    for k, v in ref.items():
        yaz("  " + k, *[sut(v, i)[0] for i in (0, 1, 2, 3)], sut(v, 1)[1])
    print("-" * 96)
    for (et, _), v in zip(sistemler, sys_hf):
        yaz(et, *[sut(v, i)[0] for i in (0, 1, 2, 3)], sut(v, 1)[1])
    print()

    # --- D8: ESLESMIS FARK + %95 GA ---------------------------------------
    # Objeler kollar arasi ORTAK => eslesmis test bedava ve eslesmemis
    # ortalamalardan cok daha duyarli. Bu olmadan "kol A kol B'den iyi"
    # cumlesi kurulamaz -- E-R kapisi tam bu yuzden savunulamazdi.
    if len(sistemler) >= 2:
        tab = sistemler[0][0].split("  [")[0][:26]
        print(f"ESLESMIS FARK (referans: {tab})")
        print(f"  {'sistem':<28} {'d(kor)':>17} {'d(kor+kaydir)':>17} {'d(kor_orta)':>17}")
        print("  " + "-" * 82)
        for (et, _), v in zip(sistemler[1:], sys_hf[1:]):
            hu = []
            for i in (1, 2, 3):
                m_, ci, nn = eslesmis_fark([x[i] for x in v],
                                           [x[i] for x in sys_hf[0]])
                yildiz = "*" if (m_ == m_ and ci == ci and abs(m_) > ci) else " "
                hu.append(f"{m_:+.3f}+-{ci:.3f}{yildiz}")
            print(f"  {et.split('  [')[0][:26]:<28} {hu[0]:>17} {hu[1]:>17} {hu[2]:>17}")
        print(f"  (* = %95 GA sifiri icermiyor; eslesmis, n yukaridaki tabloda)")
        print()

    if a.n_gorsel > 0:
        etk = ["GERCEK"] + [e.split("  [")[0][:26] for e in
                            [s_[0] for s_ in sistemler]]
        gorsel_yaz(a.out + "_kare.png", kareler, etk, a.zoom_px)


if __name__ == "__main__":
    main()
