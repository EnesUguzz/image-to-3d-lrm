"""LRMDataset: render + poz okur; her ornekte 1-4 girdi + n_sup supervision secer.
Girdiye augmentation uygulanir, supervision temiz kalir."""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lrm import cameras
from lrm import crop as cropmod
from lrm import defaults
from lrm import imutil
from lrm.augment import augment_input

# Girdi gorunum sayisi TAVANI (tasarim: kullanici 1-4 foto verir).
# Havuz 16 goruname acilinca len(pool) artik dogal tavan degil.
MAX_INPUT_VIEWS = 4

RENDER_MASTER_RES = 512  # meta intrinsic bu cozunurluge gore


# Olcekleme TEK KAYNAKTA: lrm/imutil.py. Buradaki isimler geriye donuk uyum
# icin duruyor (testler ve diger scriptler _load_rgba'yi import ediyor).
_kucult_rgba = imutil.kucult_rgba
_pil_premult = imutil.pil_premult
_unpremult = imutil.unpremult


def _load_rgba(path, res):
    return imutil.yukle_rgba(path, res)


def _crop_input(png_path, out_res, master_res, intrinsic, pad=1.15):
    """Girdi goruntusunu siluet bbox'ina KARE kirpar ve intrinsic'i kaydirir.

    Kare kirpma sart: dikdortgen kirpip kareye resize etmek en-boy oranini
    bozar ve kamera modelini gecersiz kilar. Kirpma penceresi bbox merkezli,
    kenari max(bbox_w, bbox_h) * pad.

    Intrinsic: once kirpma (ana nokta kaydir), sonra olcekle. Sira onemli.
    """
    from PIL import Image
    im = Image.open(png_path).convert("RGBA")
    W, H = im.size
    a = np.asarray(im, dtype=np.float32)[..., 3] / 255.0
    ys, xs = np.nonzero(a > 0.05)
    if len(xs) == 0:                      # bos render: kirpma yok
        arr = _unpremult(torch.from_numpy(
            np.asarray(_pil_premult(im).resize((out_res, out_res), Image.BILINEAR),
                       dtype=np.float32) / 255.0).permute(2, 0, 1))
        K = cameras.scale_intrinsics(
            torch.tensor(intrinsic, dtype=torch.float32), master_res, out_res)
        return arr, K
    cx_b = 0.5 * (float(xs.min()) + float(xs.max()) + 1.0)
    cy_b = 0.5 * (float(ys.min()) + float(ys.max()) + 1.0)
    yan = max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)) * pad
    yan = float(min(yan, min(W, H)))      # goruntuden buyuk olamaz
    x0 = float(np.clip(cx_b - yan / 2.0, 0, W - yan))
    y0 = float(np.clip(cy_b - yan / 2.0, 0, H - yan))
    im = _pil_premult(im).resize((out_res, out_res), Image.BILINEAR,
                                 box=(x0, y0, x0 + yan, y0 + yan))
    arr = _unpremult(
        torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    K = torch.tensor(intrinsic, dtype=torch.float32).clone()
    if master_res != W:                   # intrinsic master cozunurlukte saklanir
        K = cameras.scale_intrinsics(K, master_res, W)
    K[0, 2] -= x0                         # kirpma: ana noktayi kaydir
    K[1, 2] -= y0
    K = cameras.scale_intrinsics(K, yan, out_res)   # sonra olcekle
    return arr, K


# ---------------------------------------------------------------------------
# OLU VERI SETI KORUMASI
#
# 2026-09-01: `bench_detay.sh` --train_list/--renders_dir VERMEDI; fit_teacher'in
# o zamanki varsayilanlari `dataset/train_list.json` + `dataset/renders`'ti ve uc
# kol da OLU sette egitildi (CLAUDE.md: eski sphere+Cycles seti conditioning'i
# tamamen cokertiyor -- olculdu: 15.80 dB / top-1 %3, yeni sette 20.11 dB / %75).
# Ayni hata daha once cerceveleme A/B'sinde de yasanmisti.
#
# Varsayilanlar duzeltildi; bu uyari IKINCI savunma hatti. Sert hata DEGIL:
# belgelenmis AB_ESKI karsilastirmasi hala kasitli olarak eski seti okuyabilsin.
OLU_LISTE = "train_list.json"
OLU_RENDER = "renders"


def _olu_veri_uyar(train_list_path, renders_dir):
    import os, sys
    olu = []
    if os.path.basename(str(train_list_path)) == OLU_LISTE:
        olu.append("train_list  = " + str(train_list_path))
    if os.path.basename(str(renders_dir).rstrip("/" + '\\')) == OLU_RENDER:
        olu.append("renders_dir = " + str(renders_dir))
    if not olu:
        return
    bar = '!' * 72
    mesaj = [bar,
             "!! OLU VERI SETI OKUNUYOR -- sonuclar uretimle KIYASLANAMAZ"]
    mesaj += ["!! " + x for x in olu]
    mesaj += ["!! Canlisi: dataset/train_list_v2.json + dataset/renders_opp_score3",
              "!! Kasitli degilse KOSUYU DURDUR.", bar]
    print(chr(10).join(mesaj), file=sys.stderr, flush=True)

class LRMDataset(torch.utils.data.Dataset):
    def __init__(self, train_list_path, renders_dir, split="train",
                 input_res=None, render_res=128, n_sup=4, augment=True, seed=0,
                 normalize_cams=False, deterministic=None, max_input=None,
                 force_n_input=0, region=0, render_low=64, render_high=192,
                 fg_bias=0.75, input_crop=0.0, input_pool="canon", mixed_p=0.5):
        _olu_veri_uyar(train_list_path, renders_dir)
        with open(train_list_path, encoding="utf-8") as f:
            self.uids = json.load(f)[split]
        self.renders_dir = renders_dir
        self.input_res = defaults.INPUT_RES if input_res is None else input_res
        self.render_res = render_res
        self.n_sup = n_sup
        self.augment = augment
        self.base_seed = seed
        self._epoch = 0        # set_epoch() ile artirilir
        # HATA (duzeltildi): seed idx'e sabitlenince her epoch AYNI girdi gorunumu,
        # AYNI supervision gorunumleri ve AYNI augmentation cikiyordu => 16 render'in
        # 12'si hic kullanilmiyor, augmentation rastgele degil (her obje kalici olarak
        # tek bir arka plan rengine yapisik). Egitimde her erisim taze rastgelelik ister;
        # val/onizleme sabit kalsin diye augment kapaliyken deterministik.
        self.deterministic = (not augment) if deterministic is None else deterministic
        # kamera normalizasyonu (LRM +3.7 PSNR): giris kamerasini kanonik poza sabitle.
        # ANCAK ezber kisayolunu kapatir => kucuk veride (2635, hatta 50K) yakinsama
        # cok yavaslar (izole test: mask 400 adimda hala 0.5). 1M+ olcekte faydali.
        # Bu yuzden bizim rejimde VARSAYILAN KAPALI; buyuk veride tekrar denenecek.
        self.normalize_cams = normalize_cams
        # Girdi gorunum sayisi tavani. None = meta'daki kanonik sayisi kadar
        # (ring12 semasinda 4, sphere20 semasinda 6 -> ust/alt de girdi olabilir).
        self.max_input = max_input
        # force_n_input>0: k'yi sabitler. Val metriginde sart -- yoksa her objenin
        # girdi sayisi farkli olur ve "tek fotodan rekonstruksiyon" olcumu karisir.
        self.force_n_input = force_n_input
        # BOLGE KIRPMA (OpenLRM/TripoSR). region>0 ise supervision hedefi, U[low,high]
        # cozunurlukte render edilmis kareden kirpilan region x region bir yamadir.
        # Isin butcesi region^2'de SABIT kalir ama yamadaki obje orani cok yukselir
        # (objelerimiz tam karenin sadece ~%9'unu kapliyor). Bkz. lrm/crop.py.
        self.region = region
        self.render_low = render_low
        self.render_high = render_high
        self.fg_bias = fg_bias
        # GIRDI SIKI KIRPMA (2026-08-27 bagimsiz denetim, olculdu):
        # `fit` normalizasyonu 16 kameradaki EN KOTU bbox kosesini cerceveye
        # oturtuyor => fiilen kure normalizasyonuna dejenere oluyor ve obje
        # kanonik goruumun sadece ~%9'unu kapliyor. DINOv2'nin 256 patch'inden
        # ~19'u obje (p05: 4). Siluet bbox'ina x1.15 payla kirpinca kaplama
        # 0.0885 -> 0.2413, patch ~19 -> ~77 (4x).
        # Faz C ZATEN bunu yapmak zorunda (foto arka plani silinip ortalanacak),
        # yani kapaliyken egitim ve cikarim FARKLI cercevelemede.
        # 0.0 = kapali (eski davranis). >0 = bbox'a uygulanacak pay carpani.
        self.input_crop = float(input_crop)
        # GIRDI GORUNUM HAVUZU (2026-08-29 alan taramasi, S1)
        #   "canon" : sadece 4 kanonik  -> girdi elevation DAIMA +20 (eski davranis)
        #   "mixed" : mixed_p olasilikla 16 gorunumun tamamindan, yoksa kanonik
        #   "all"   : daima 16 gorunumun tamamindan
        # Taranan HICBIR referans girdi elevation'ini sabitlemiyor: LRM/OpenLRM/
        # InstantMesh girdiyi 32 rastgele gorunumden seciyor, Hunyuan3D-1.0
        # kosul goruntulerini elev U[-20,+60]'tan orneklerken HEDEF goruntuleri
        # sabit tutuyor. Bizde tam tersiydi. Diskteki 12 supervision gorunumu
        # zaten elev [-10,+80] araliginda -> bu duzeltme SIFIR render maliyetli.
        assert input_pool in ("canon", "mixed", "all"), input_pool
        self.input_pool = input_pool
        self.mixed_p = float(mixed_p)

    def __len__(self):
        return len(self.uids)

    def _meta(self, uid):
        with open(os.path.join(self.renders_dir, uid, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def set_epoch(self, e):
        """Epoch basina rastgeleligi degistirir ama TEKRAR URETILEBILIR birakir.

        DataLoader worker'lari dataset'in bir KOPYASINI tasidigi icin bu,
        epoch basi ana surecte cagrilmali; persistent_workers=True ise
        worker kopyalari guncellenmez -> loader yeniden kurulmali ya da
        persistent_workers kapatilmali."""
        self._epoch = int(e)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        meta = self._meta(uid)
        if self.deterministic:
            rng = random.Random(self.base_seed * 1_000_003 + idx)
        else:
            # TEKRAR URETILEBILIRLIK (2026-08-27 bagimsiz denetim):
            # eskiden random.Random() (OS entropisi) idi. Epoch'lar arasi
            # cesitlilik dogruydu ama KOSULAR ARASI tekrar uretilemezlik de
            # birlikte geliyordu: ayni komutla iki kol farkli gorunum, farkli
            # augmentation, farkli kirpma cozunurlugu goruyordu -> tek-degiskenli
            # A/B imkansiz. Simdi cesitlilik `epoch` sayacindan geliyor.
            rng = random.Random((self.base_seed * 1_000_003 + idx) * 1_000_003
                                + self._epoch)
        canon = list(meta["canonical_indices"])
        views = meta["views"]
        n_views = len(views)
        master_res = meta.get("resolution", RENDER_MASTER_RES)

        if self.input_pool == "all":
            pool = list(range(n_views))
        elif self.input_pool == "mixed" and rng.random() < self.mixed_p:
            pool = list(range(n_views))
        else:
            pool = canon
        # TAVAN: eskiden havuz=kanonik oldugu icin len(pool)=4 dogal bir tavandi.
        # Havuz 16'ya cikinca k=16 olup supervision'i BOSALTIYORDU (test yakaladi).
        # Tasarim zaten "1-4 gorunum": tavan acikca 4.
        hi = min(self.max_input if self.max_input is not None else MAX_INPUT_VIEWS,
                 len(pool))
        k = self.force_n_input if self.force_n_input else rng.randint(1, hi)
        k = max(1, min(k, len(pool)))
        input_idx = rng.sample(pool, k)
        remaining = [i for i in range(n_views) if i not in input_idx]
        sup_idx = rng.sample(remaining, min(self.n_sup, len(remaining)))

        def K_for(i, dst_res):
            # intrinsic Faz A'da her view'de saklanir (master cozunurluk icin)
            K = torch.tensor(views[i]["intrinsic"], dtype=torch.float32)
            return cameras.scale_intrinsics(K, master_res, dst_res)

        def c2w(i):
            ext = torch.tensor(views[i]["extrinsic"], dtype=torch.float32)
            return torch.linalg.inv(ext)  # extrinsic = world->camera

        def path(i):
            return os.path.join(self.renders_dir, uid, views[i]["file"])

        input_imgs, input_c2w, input_K = [], [], []
        for i in input_idx:
            if self.input_crop > 0:
                rgba, Ki = _crop_input(path(i), self.input_res, master_res,
                                       views[i]["intrinsic"], self.input_crop)
            else:
                rgba = _load_rgba(path(i), self.input_res)
                Ki = K_for(i, self.input_res)
            if self.augment:
                img = augment_input(rgba, random.Random(rng.random() * 1e9))
            else:
                img = rgba[:3] * rgba[3:4] + (1.0 - rgba[3:4])  # temiz: BEYAZ bg
            input_imgs.append(img)
            input_c2w.append(c2w(i))
            input_K.append(Ki)

        sup_rgb, sup_premult, sup_alpha, sup_c2w, sup_K = [], [], [], [], []
        for i in sup_idx:
            if self.region:
                # her gorunum kendi cozunurlugunu ve kendi capasini alir (OpenLRM)
                r = cropmod.sample_render_res(rng, self.render_low,
                                              self.render_high, self.region)
                rgba = _load_rgba(path(i), r)
                ax, ay = cropmod.sample_anchor(rgba[3:4], r, self.region, rng,
                                               self.fg_bias)
                rgba = cropmod.crop_image(rgba, ax, ay, self.region)
                K = cropmod.scale_and_crop_K(
                    torch.tensor(views[i]["intrinsic"], dtype=torch.float32),
                    master_res, r, ax, ay)
            else:
                rgba = _load_rgba(path(i), self.render_res)
                K = K_for(i, self.render_res)
            # premultiplied obje rengi (obj*alpha): egitimde her adim RASTGELE bg
            # rengine kompozitlenir => model sabit ciktiyla arka plani tutturamaz,
            # objeyi gercekten kurar (sabit-bg 'renk cokmesi' tuzagini kapatir).
            sup_premult.append(rgba[:3] * rgba[3:4])
            # sup_rgb: beyaz-kompozit (yalnizca gorsel onizleme/val icin)
            sup_rgb.append(rgba[:3] * rgba[3:4] + (1.0 - rgba[3:4]))
            sup_alpha.append(rgba[3:4])
            sup_c2w.append(c2w(i))
            sup_K.append(K)

        in_c2w = torch.stack(input_c2w)
        su_c2w = torch.stack(sup_c2w)
        if self.normalize_cams:
            # referans = ilk giris kamerasi; ayni rotasyonu giris+supervision'a uygula
            ref = in_c2w[0]
            in_c2w = cameras.canonicalize(ref, in_c2w)
            su_c2w = cameras.canonicalize(ref, su_c2w)

        return {
            "uid": uid,
            "idx": idx,
            "input_imgs": torch.stack(input_imgs),
            "input_c2w": in_c2w,
            "input_K": torch.stack(input_K),
            "input_view_idx": input_idx,
            "sup_rgb": torch.stack(sup_rgb),
            "sup_premult": torch.stack(sup_premult),
            "sup_alpha": torch.stack(sup_alpha),
            "sup_c2w": su_c2w,
            "sup_K": torch.stack(sup_K),
            "sup_view_idx": sup_idx,
            "sup_res": self.region or self.render_res,
        }


def lrm_collate(batch):
    """Degisken girdi sayisi (1-4) => padding yerine liste dondur.
    Model tek obje isler; egitim dongusu listeyi gezer."""
    return list(batch)
