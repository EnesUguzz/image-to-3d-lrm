# 3D Object Project — Claude Talimatları

Bu dosya, bu projede çalışan her Claude oturumu için kalıcı bağlamdır. Kararlar burada; koda başlamadan önce oku.

> 📘 **Yeni oturum açıyorsan önce bunu oku: [`docs/PROJE-DEVIR-BELGESI.md`](docs/PROJE-DEVIR-BELGESI.md)**
> Projenin tüm geçmişi, deney defteri (her koşunun sonucu + kanıt dosyası),
> metodolojik uyarılar ve açık sorular tek dosyada. Bu dosya (CLAUDE.md) kararları
> özetler; devir belgesi kanıtları taşır.

## 🎯 Proje Hedefi (Kuzey Yıldızı)

Tek bir **transformer-tabanlı LRM (Large Reconstruction Model)** modeli eğitmek:
kullanıcı **1 foto** (ör. sadece "ön") ya da **4 kanonik açı** (ön/arka/sol/sağ) verir →
model triplane NeRF üretir → mesh (`.glb`) çıkarılır → web app'te three.js ile gösterilir.

- Difüzyon (Zero-1-to-3) DEĞİL, **transformer/LRM** yaklaşımı seçildi.
- Tek model hem 1 hem 4 görüntüyü kabul eder (eğitimde rastgele 1–4 görünüm kullanılır).
- Bağlam: **kişisel öğrenme projesi**. Amaç SOTA değil, uçtan uca çalışan+anlaşılan pipeline.

## 🧭 Mimari — 3 Faz

Her faz kendi `spec → plan → implementation` döngüsüne girer (superpowers akışı).

- **Faz A — Render pipeline** *(BİTTİ)*
  Objaverse `.glb`'lerini sabit kanonik açılardan çok-görünümlü foto + kamera pozlarına çevir.
  **14.454 temiz obje (13.009 train + 1.445 val)**, `dataset/train_list_opp_score3.json`,
  render'lar `dataset/renders_opp_score3/`.
  ⛔ Eski set (`dataset/renders` + `train_list.json`, 2635+292) **ÖLÜ** — ölçüldü,
  conditioning'i çökertiyor. Aşağıdaki "gece deney sonuçları" bölümüne bak.
- **Faz B — Eğitim (asıl iş)** *(ŞU AN BURADAYIZ — eğitim çalışıyor)*
  Multi-view LRM: donuk DINOv2 ViT-S/14 → transformer (dim 512, 12 kat) → triplane
  (3×64×64×32) → NeRF → volume render → fg-ağırlıklı MSE+LPIPS+mask loss.
  Kod: `scripts/lrm/` + `scripts/train_lrm.py` + `scripts/overfit_lrm.py`.
  Spec: `docs/superpowers/specs/2026-08-18-faz-b-lrm-egitimi-design.md`.
- **Faz C — Web app**
  Backend: foto → inference → mesh (marching cubes) → `.glb`. Frontend: three.js viewer.

## 📐 Faz A — Kesinleşmiş Kararlar

- **Render çözünürlüğü:** 512×512 RGBA PNG (alpha/mask dahil), diskte master olarak.
- **Görünüm sayısı:** obje başına **16 görünüm**
  - **4 kanonik** (input adayı): azimuth 0°/90°/180°/270°, elevation **~20°** (ön/arka/sol/sağ)
  - **+12 ek görünüm** (supervision): halka + rastgele elevation
  - Her görünümün kamera pozu (extrinsic + intrinsic) `meta.json`'a yazılır.
- **Eğitim çözünürlüğü (Faz B):** 256×256 (kod içinde 512→256 downscale). Gerekirse 512 finetune.
- **Subset:** önce **~1000 obje** ile doğrula, çalışınca 5–10K'ya çıkar.
- **Girdi ≠ eğitim çözünürlüğü:** Kullanıcı fotosu Faz C'de arka planı silinir + ortalanır + sabit
  boyuta (256) resize edilir. Kullanıcının çözünürlüğü önemsiz; kod normalize eder.

### Dosya/isimlendirme şeması
```
dataset/
  renders/<uid>/000.png ... 015.png   # 512×512 RGBA
  renders/<uid>/meta.json              # intrinsics + her görünümün extrinsic'i,
                                        # kanonik görünüm indexleri, azimuth/elevation
  index.json                           # tamamlanan uid'ler + train/val split
  manifest.jsonl                       # resume/skip-completed logu (uid, durum, süre)
```

## 💻 Ortam & Donanım (kritik)

- **GPU:** NVIDIA RTX 5080, 16 GB VRAM, **Blackwell (sm_120)**.
  → Stabil desteği yeni geldi; **CUDA 12.8+ ve güncel PyTorch** kur. Eski wheel'ler sm_120 desteklemez.
- **OS:** Windows 11. **Shell:** PowerShell (birincil), Bash (POSIX) da mevcut.
- **Python:** 3.10.8 (sistemde). torch henüz KURULU DEĞİL.
- **Blender:** 4.4 kurulu. `C:\Program Files\Blender Foundation\...`

## 📦 Veri Konumu

- İnen objeler: `%USERPROFILE%\.objaverse\hf-objaverse-v1\glbs\`
  → **51.534 adet `.glb`**, ~490 GB, `<uid>.glb` formatında 160 alt klasörde.
- URL haritası: `%USERPROFILE%\.objaverse\hf-objaverse-v1\object-paths.json.gz`
- Ayrıca: `%USERPROFILE%\.objaverse\smithsonian\objects\` (2392 glb, alternatif set).

## 🔗 Referans Repolar

- `allenai/objaverse-xl` — veri seti + indirme paketi.
- `allenai/objaverse-rendering` — orijinal render scriptleri. Local kopya:
  `%USERPROFILE%\objaverse-rendering\scripts\` (blender_script.py, distributed.py, ...).
- OpenLRM — Faz B için referans implementasyon.

## ⚠️ Bilinen Tuzaklar

- Orijinal `blender_script.py` **Blender 3.2** için yazılmış; **4.4'te API farklı**
  (ör. `BLENDER_EEVEE` → `BLENDER_EEVEE_NEXT`, bazı render ayarları değişti). Uyarlanmalı.
- `distributed.py` + `start_xserver.py` **Linux'a özel** (xserver, `export DISPLAY`).
  Windows'ta Blender doğrudan çağrılır; xserver GEREKMEZ. Bu scriptler Windows'a yeniden yazılacak.
- Orijinal script 12 görünüm/elevation~30° kullanır; biz **16 görünüm + 4 kanonik @20°** kullanıyoruz.

### Faz A uygulama sırasında bulunan/çözülen sorunlar (2026-08-09)
- **Göreli çıktı yolu:** Blender `render.filepath`'i göreli yolu açık .blend'e göre çözer;
  blend yokken sürücü köküne (`C:\`) düşer. Çözüm: `render_object.py` çıktı yolunu `abspath`'ler.
- **Windows kodlama:** `subprocess`'te Blender stdout'u cp1254 ile çözülünce `UnicodeDecodeError`.
  Çözüm: `encoding="utf-8", errors="replace"` (test + `run_batch.py`).
- **Dev zemin (FLOOR) plane'leri:** Bazı Objaverse glb'lerinde objeden ~5× büyük düz bir zemin
  mesh'i var → bbox'ı şişirip objeyi minik ölçekliyor + render'da gölge gibi görünüyor.
  Çözüm: `remove_floor_planes()` — 'düz VE içerikten çok büyük' mesh'i normalize öncesi siler.
- **Normalize iki-geçişli olmalı:** tek geçişte world-center ile local-location karışıyordu.
  Çözüm: ölçekle → güncelle → bbox yeniden ölç → dünya-uzayında merkezle (objaverse ile aynı).
- **Aydınlatma:** tek tepe ışığı bazı açıları karartıyordu → world ambient 1.5 + üst/alt yumuşak
  area ışıklar (her açıda albedo görünür).
- **Doğrulanan hız:** RTX 5080'de 512×512, 16 görünüm ≈ **obje başına 7–14 sn**.

### Curation + log + veri stratejisi (2026-08-17)
- **Veri stratejisi:** Önce **LVIS kürate ~3036 obje** (indirilen 50k ∩ LVIS 46k) ile eğit
  (temiz, insan-doğrulamalı, 1.0 uid'leriyle eşleşir). Yetmezse Objaverse++ (50k) indir;
  fallback Objaverse-XL alignment (1.3M, ama XL-keyed + yeni indirme + aylarca render).
  Küçükle başla = pahalı render'a girmeden tüm döngüyü ucuza doğrula.
- **Curation iki katman:** (1) `build_subset.py --curated` LVIS kesişimi (cache: `dataset/lvis_uids.json`),
  (2) `filter_dataset.py` render sonrası **alpha kaplama + kenar-taşma** filtresi → `train_list.json` (train/val split).
- **Robustluk (kötü kullanıcı fotosu) = input augmentation**, bozuk hedef DEĞİL. Temiz 3D hedef +
  girdide blur/crop/jpeg/bg simülasyonu (Faz B dataloader'a not).
- **Log sistemi:** `run_batch.py` zaman damgalı log (`dataset/renders/logs/render_*.log`) +
  koşu sonu özeti (`render_summary.json`: done/failed/atlandı, toplam+ort süre, en yavaşlar, hata dökümü).
- **Kodlama tuzağı:** proje yolu **Ğ** (kullanıcı adında) içeriyor → tüm json okuma/yazmalarda
  **`encoding="utf-8"`** şart (cp1254 varsayılanı çöker). Tüm scriptlerde uygulandı.

### Faz B — LRM eğitimi: KÖK NEDEN ÇÖZÜLDÜ (2026-08-21)

Tam teşhis zinciri ve tekrar üretilebilir koşular: **`docs/lrm-blob-diagnosis.md`**.
Aşağıdakiler ölçümle kanıtlanmıştır; 2026-08-19 tarihli önceki notların bir kısmı
(beyaz bg, bf16, coarse-to-fine reçetesi) bu bulgularla **aşılmıştır**.

- **Ortam:** `torch 2.11.0+cu128`, RTX 5080 sm_120. DINOv2 (torch.hub) + LPIPS-VGG cache'li.
- **Meta.json şeması:** `intrinsic` her view içinde, `c2w = inv(extrinsic)`.

**🔑 KÖK NEDEN: pipeline'da hata yoktu — "ortalama obje" yerel minimumu.**
Model rastgele başlangıçtan, girdiden bağımsız "veri setinin ortalaması"nı basmayı
öğreniyordu (dataset ortalama rengi `[0.469,0.418,0.387]` = basılan kahverengi ton).
Ölçümle **elenenler**: renderer, kameralar, NeRF MLP derinliği, triplane çözünürlüğü,
donuk DINOv2 encoder, transformer mimarisi (cross-attention denendi → fayda yok),
veri kalitesi, veri miktarı. Kanıt zinciri:
`oracle triplane 25.04 dB` → `aynı ağ, triplane'e regresyon 23.94 dB` →
`aynı ağ, render kaybıyla sıfırdan 17.06 dB` → `distile ağırlıklardan render kaybı 24.99 dB`.
İyi çözümün render kaybı 0.23, blob'un 0.47 → **hedef doğru, optimizasyon sıkışıyor.**

**Düzeltmeler (ölçülmüş):**
1. **`w_lpips` 2.0 → 0.25.** LRM'in 2.0'ı 730k obje + batch 1024 rejimi için; bizimkinde
   LPIPS baskın olunca "makul genel doku" ödüllendirilip ortalama havzasında kalınıyor.
   32 obje: top-1 %25 → **%62**. (0.0 daha da conditioning verir ama gürültülü + yüksek
   LR'de ıraksar.) `train_lrm.py` varsayılanı 0.25.
2. **Öğretmen triplane önyüklemesi.** Obje başına serbest triplane render kaybıyla hızla
   doğru çözüme gider (araya transformer girmez); LRM ona distile olunca doğru havzada
   başlar, öğretmen kapanınca kendi başına gelişir.
   32 obje: 17.06 dB/%25 → **26.68 dB/%100**. 250 obje: 18.97/%30 → **25.90/%98**.
   Yeni (eğitimde hiç görülmemiş) kamera açısı: **23.11 dB / top-1 %100**.
3. **`dataset.py` RNG hatası.** `random.Random(seed*1000003+idx)` her epoch aynı seed'i
   veriyordu → her objenin girdi/supervision görünümleri ve augmentation'ı sabitti
   (16 render'ın 12'si ölü). Düzeltildi + regresyon testleri.

**⚠️ ÖĞRETMEN NUMARASI ÖLÇEĞE BAĞLI — 2635'te ÇALIŞMADI.**
Öğretmenin obje başına aldığı güncelleme `batch ÷ veri_boyutu`:
32→250, 250→128, **2635→34**, 50k→1.8. Üç deneme de bu yüzden başarısız oldu
(öğretmen öğrenciden kötü kaldı; sonra paylaşılan NeRF çöküp öğretmeni de öldürdü).
İki aşamaya ayırmak (`fit_teacher.py` + `distill_lrm.py`) öğretmeni 27.43 dB'ye
çıkardı ama öğrenci 1024 objede bağıl hata ~0.71'de takıldı (32 objede 0.41'di)
→ sınır artık **donuk encoder + model kapasitesi**. Referans implementasyonların
üçü de (LRM, OpenLRM, TripoSR) encoder'ı **eğitiyor**.

**Karar:** 2635 ara ölçeği kovalamak bırakıldı. Sistem kanıtı 32/250 objede tamam;
büyüme yolu veri (50k) + encoder'ı çözmek + efektif batch'i büyütmek.

**Araçlar:** `scripts/bench_overfit.py` (N-obje ezberleme kapısı, top-1 retrieval +
ortalama-baseline oranı + yeni-görünüm), `bench_triplane_fit.py` (oracle/temsil tavanı),
`bench_distill.py`, `bench_teacher.py`, `diag_retrieval.py`, `diag_teacher_quality.py`,
`fit_teacher.py`, `distill_lrm.py`.

### Faz A — render ayarları (2026-08-24, ölçülmüş)

**⚠️ 2026-08-21 tarihli "çerçeveleme 1.87× kazanç" notu YANLIŞTI, düzeltildi.**
O ölçüm 6 objeyle yapılmıştı ve taşma 0.000 çıkmıştı. 100 objede tekrarlanınca
`bbox` modunda render'ların **%5'i kenardan taşıyor** (13/100 obje, en kötü 0.65).
Kazancın tamamı objeleri kırpmaktan geliyormuş. Ayrıca bbox modu bizim pipeline'da
**doğruluk hatası**: NeRF `bound=0.6` dışını maskeliyor ama bbox normalizasyonunda
obje yarıçapı 0.866'ya çıkabiliyor.

**`NORM_MODE` (varsayılan `"fit"`):**
- `"sphere"` (eski): bbox yarı-köşegeni = 0.5. Taşma yok, kaplama 0.0843.
- `"bbox"`: en uzun kenar = 1.0 (Objaverse konvansiyonu). Kaplama 0.1695 **ama %5 taşma** → KULLANMA.
- `"fit"` (yeni): iki geçişli — bbox normalize, sonra objenin 16 kamera görünümündeki
  gerçek izdüşüm genişliği ölçülüp kareye tam oturtulur. **Kaplama 0.0899, taşma %0.00.**

Yani doğru çerçevelemede eski moda göre kazanç ~%7, ihmal edilebilir. 16 görünümlü
bir kamera halkasında en kötü izdüşüm zaten sınırlayıcı küreye yakın çıkıyor.

**Render motoru: CYCLES → EEVEE_NEXT (asıl kazanç burada).**
Objaverse'in kendi `blender_script.py`'si de `BLENDER_EEVEE` varsayılanını kullanır
(Cycles seçilse bile `samples=32`, `bounces=1` ile hafifletilmiş; bizimki 48 örnek +
tam bounce ile daha ağırdı). Ölçüm (24 obje × 16 görünüm = 384 render çifti):
alpha kaplama farkı **0.00001**, obje RGB farkı ort **%2.4**, PSNR ort **37.4 dB**
(en kötü 27.3 dB, metalik/yansımalı obje). EEVEE'nin en büyük zaafı cam/kırılma —
Objaverse++ filtresinde `is_transparent` elendiği için bizi ilgilendirmiyor.
`RENDER_ENGINE=CYCLES` ile geri alınır.

**Throughput (ölçüldü, 100 obje, EEVEE_NEXT 32 örnek):**

| config | sn/obje | 14.594 obje |
|---|---|---|
| Cycles 48 + 4 worker | 3.68 | 14.9 saat |
| **EEVEE 32 + 8 worker** | **0.66** | **~2.7 saat** |

Worker taraması: Cycles'ta 4 optimal, EEVEE'de **8** optimal (12 daha kötü).
Blender süreç başlatma 0.30 sn — ihmal edilebilir, optimize edilecek yer değil.
Cycles'ta örnek sayısını 48→24 düşürmek hiç fark etmiyor (süre BVH kurulumunda).

**100-obje doğrulaması (fit + EEVEE):** 0 hata, meta şeması tam (16 görünüm,
512×512, kanonik [0,1,2,3], kamera yarıçapı 1.4866 — hepsinde tutarlı),
taşma %0.00, boş render yok.

### 14.5k render TAMAM + gece deney sonuçları (2026-08-25)

**KULLANILACAK VERİ:**
`dataset/renders_opp_score3/` (14.597 obje × 16 görünüm, 37 GB, `fit`+EEVEE) +
`dataset/train_list_opp_score3.json` (**13.009 train / 1.445 val**).
Render: 0 hata, 2s27dk, 0.61 sn/obje. Filtre sonrası %99,04 geçti.

**⛔ `dataset/renders/` ÖLÜ — kullanma.** Ölçüldü (`AB_ESKI` vs `AB_YENI`, aynı 32
uid, tek değişken render dizini): eski `sphere`+Cycles seti conditioning'i tamamen
çökertiyor. Eski **15.80 dB / top-1 %3 / oran 1.000** ↔ yeni **20.11 dB / %75 / 0.60**.

**Alt/üst görünüm GEREKMİYOR** (`COV_alt0` vs `COV_alt1`, `bench_view_coverage.py`).
Adil kurulum (her iki koşulda 12 fit görünümü, ikisi de aynı held-out ALT
görünümlerde ölçüldü): alt yok **27.26 dB**, alt var **27.74 dB** → +0.48 dB,
belirsizlik eşiğinin altında. Yandan/üstten 12 görünüm objenin altını zaten
belirliyor. `sphere20` şeması kodda var (`RENDER_VIEW_SCHEME=sphere20`, 24 görünüm,
6 kanonik) ama **yeniden render'a değmez.**
*Uyarı: bu supervision tarafı. Üst/alt fotoyu GİRDİ vermek ayrı soru, ölçülmedi.*

**Tarife matrisi** (32 obje, aynı alt küme, tek değişken):
| etiket | değişken | PSNR | top-1 | oran |
|---|---|---|---|---|
| `M_crop08` | `--crop 0.8` | 17.48 | **%38** | **0.764** |
| `M_batch8` | `--batch 8` | **19.29** | %31 | 0.805 |
| `M_base` | — | 17.80 | %3 | 1.000 |
| `M_enc4` | `--unfreeze_last 4` | 17.80 | %3 | **GEÇERSİZ** |
| `M_encall` | `--train_encoder` | 16.93 | %12 | 1.013 |

→ **ön-plana yanlı kırpma** ve **büyük batch** işe yarıyor.

**⛔ DÜZELTME (2026-08-26): bu matris GEÇERSİZ.** `M_base` ve `M_enc4`'ün ikisi de
yoğunluğu sıfıra çökertip **tamamen beyaz** çıktı üretmiş (`pred_min=1.000`,
`pred_mean=1.0000`, objeler arası std `0.0000`; PSNR 500/1000/1500/1999. adımlarda
birebir 17.80 dB'de donmuş). `acc→0` ⇒ renderer `bg_color`'ı aynen basar; softplus
gradyanı öldüğü için encoder'ı açmak da açmamak da bir şey değiştiremez —
"bit-bazında aynı" tespiti de yanlıştı (loss 5. ondalıkta farklı: 0.5465 / 0.54651).
Yani `M_base` bir *taban tarife* değil bir *çöküş*; `oran=1.000` referansına göre
yapılan tüm sıralama çöp. Aynı tarife farklı 32-obje kümesiyle (`AB_YENI`) çökmedi
(20.11 dB / %75) → çöküş hem tarifeye hem alt kümeye bağlı.
**Kısmi encoder çözme hâlâ CEVAPSIZ ama sebebi artık biliniyor.**
`TriplaneNeRF`'in `density_bias` / `noise_std` çengelleri tam bu çöküşün panzehiri
ama `LRM.__init__` hiç bağlamıyor (daima 0.0) — bağlanmalı.

**İlk tam-veri genelleme eğrisi** (13k obje, 16.000 adım, taban tarife — BİTTİ):
```
adım  500–8000 : top-1 = şans (%1.6), oran ≥1.0      → tam çöküş
adım      8500 : sıçrama (coarse→fine, res 64→128)
adım 9000–11000: top-1 %3.1→%6.2→%7.8, PSNR 15.8→16.8
adım 11000–15500: PSNR 16.7–16.8 SABİT, top-1 %9–14, oran 1.04–1.08
FINAL (15500)  : PSNR 16.72  top-1 %14.1 (şans %1.6)  oran 1.054
```
**Yorum:** Taban tarife tam çöküşten kurtuluyor (top-1 şansın ~9 katı) ama
**gerçek conditioning'e ulaşmıyor.** `oran > 1.0` demek: her tahmin, kendi hedefi
için hâlâ veri-seti-ortalaması blob'undan daha kötü. Son 4.500 adım (cosine
kuyruğu, lr 3e-05→0) **hiçbir şey kazandırmadı** — PSNR 16.8'de sabit kaldı.
→ Taban tarife 13k'da yetersiz. Matrisin kazananları (crop, büyük batch) denenmeli.

**`train_lrm.py` artık SAYISAL VAL üretiyor** (`build_val_probe` + `val_metrics`).
Girdi = kanonik[0], hedef = kanonik[1..2]; kanonik açılar tüm objelerde aynı olduğu
için metrik objeler arası karşılaştırılabilir. Çıktı: `psnr`, `top1` (şans 1/N),
`oran = mse/ortalama-baseline` (**1.0 = blob çöküşü**). → `dataset/lrm_logs/val_metrics.jsonl`

### ⚠️ ÖLÇÜM YAPARKEN — bu projede fiilen yapılmış hatalar

1. **32-obje kapısı HANGİ 32 objeye aşırı duyarlı.** Aynı veri + aynı ayar, farklı
   alt küme: top-1 %75 / %19 / %3. **Sadece aynı uid kümesi içinde karşılaştır**,
   A/B kollarını `--uids` ile sabitle. Farklı koşulardan gelen top-1'leri yan yana koyma.
2. **A/B kolları iki kez sessizce aynı veriyi okudu.** (a) EEVEE/Cycles karşılaştırmasında
   iki kol farklı `--uid` aldı — kamera açıları uid'den seed'leniyor (`_seed_from_uid`).
   (b) Çerçeveleme A/B'sinde "yeni" kolu `--render_dir` almamıştı, varsayılan eski dizini
   okudu. **Kolu başlatmadan çalışan sürecin komut satırını oku.**
3. **Küçük örneklemle ölçme.** "bbox 1.87× kazanç" 6 objeyle ölçülmüştü, 100 objede çürüdü.
   En az 24–40 obje kullan.
4. **32-obje sonucu ölçeğe transfer etmeyebilir** — öğretmen numarası 32'de %100, 2635'te kırıldı.
5. **Eski çıktılar klasörlerde duruyor** (`lrm_val_previews/`, `lrm_bench/`). Dosya tarihine bak.

### 📋 Açık işler → **`docs/egitim-oncesi-hazirlik-plani.md`** (2026-08-26)

Tam plan, ön-kayıtlı kapılar, senaryolar ve GO/NO-GO listesi orada. Özet:

- **Kök neden netleşti:** tarifemiz OpenLRM'in kendi `train-sample.yaml`'ından
  **5 noktada** sapıyor ve beşi de aleyhimize: encoder DONUK (referans açık),
  denetim tam kare (referans **64² bölge kırpma**), fp32 (referans bf16),
  `normalize_camera` kapalı (referans açık), warmup 500 (referans 3000).
  Üstüne biz **~10 epoch** yaptık, referans **60**. Mimari aynı (dim 512, 12 kat,
  triplane 32→64×32, DINOv2 ViT-S/14) → mimari tartışması kapandı.
- **Hız: ölçülmüş 3,0× hazır** — 192,3 → 63,5 ms/obje (bf16 1,79× + 3 görünüm × 64²).
  Elenenler: `torch.compile` (%2), obje-batch'leme (fayda yok), ışın-AABB (küçük).
  Veri yükleme senkron 28,5 ms/obje → worker'lı `DataLoader` şart.
- **Veri tavanı:** diskte temiz score≥2 = **24.571** obje (score-2'nin 9.977'si
  henüz render edilmedi, ~1,8 sa). 50k için **~25,5k yeni indirme (~245 GB)**.
- **Strateji:** referans tarifeyi ablasyonsuz benimse; deney bütçesini sadece
  bize özgü sorulara harca (LPIPS ağırlığı, mask loss biçimi, `density_bias`,
  girdi çözünürlüğü, `bound`, kırpma tipi).
- **İlk iş — KARAR KAPISI:** `fit_teacher` (1024 obje, YENİ render setinde) →
  `distill_lrm` **tam program** (24k adım). `train_rel < 0.25` ise sorun tarife,
  `≥ 0.50` ise kapasite. Önceki "0,71'de takıldı" sonucu **%27'de kesilmiş** koşudan.
- **Eğitimden ÖNCE yapılmamış olanlar:** mesh çıkarma doğrulaması (oracle triplane
  üzerinde), tam veri denetim raporu, dokunulmamış test seti, resume testi,
  gerçek foto değerlendirme seti, `bound` kalibrasyonu.

#### Blok 0 BİTTİ (2026-08-26) — yeni araçlar ve zorunlu alışkanlıklar

- `scripts/lrm/guards.py` — **çöküş dedektörü**. `bench_overfit.py` ve
  `train_lrm.py`'ye bağlı; her değerlendirmede `acc` + objeler arası piksel std
  loglanıyor. Eşik **0,010** (çökmüş koşular 0,0000–0,0001; sağlamlar 0,038–0,088).
  Dejenere koşu `<<< SIRALAMAYA SOKMA` uyarısı basar — **sıralamaya sokma.**
  Eğitilmemiş model tanımı gereği `MEAN_COLLAPSE`'tadır; sesli alarm
  `2 × warmup` sonrası çalar (`--collapse_after`).
- `scripts/lrm/runstamp.py` — koşu künyesi (`git_sha`, argv, torch, GPU, config
  hash) + **ağırlık-değişti kontrolü**. Bir bayrağın "açıldığını" iddia eden her
  koşu artık ağırlıkların gerçekten değiştiğini kanıtlıyor.
  *(Doğrulandı: `--unfreeze_last 2` → 30/30 encoder tensörü değişiyor. Yani
  `M_enc4` gizemi tamamen kapandı: gradyan akıyordu, koşu beyaz çöküşteydi.)*
- `scripts/lrm/defaults.py` — `N_SAMPLES`/`BOUND`/`NEAR`/`FAR` **tek kaynak**.
  `fit_teacher.py` near/far'ı elle yazıyordu (öğretmen–öğrenci sessizce ayrışabilirdi).
- `dataset/bench_uids_{32,250,1024}.json` — **sabit, iç içe** (32⊂250⊂1024) tezgâh
  kümeleri, artık repoda (`.gitignore`'da istisna var). **A/B kollarını daima
  `--uids_file` ile sabitle**, `--n_obj` ile listeden adımlama.
- `scripts/bench_speed.py` — hız tezgâhı. Ölçülen: taban 196,7 ms/obje (5,1 obje/s)
  → hedef (bf16 + 3 görünüm × 64²) **69,0 ms/obje (14,5 obje/s), VRAM 2,72 GB**.
  `n_samples` 48→96 sadece %8 maliyetli.

- **⛔ `dataset/lrm_ckpts/last.pt` (16.000 adım) KAYBEDİLDİ.** 40 adımlık bir duman
  testi ezdi: `train()` sonunda `save_checkpoint` **koşulsuz** çağrılıyordu.
  Val eğrisi (`val_metrics.jsonl`) ve önizlemeler duruyor. Tuzak kapatıldı —
  `save_checkpoint` artık küçük adımlı kaydı büyük adımlının üzerine `force=True`
  olmadan yazmıyor, 3 regresyon testi var. Duman testi checkpoint'i
  `dataset/lrm_ckpts/SMOKE_step40_KULLANMA.pt` olarak kenara alındı
  (`--resume` sessizce yüklemesin).
- Testler: **101 geçiyor** (önce 63).

#### Blok 1 BİTTİ (2026-08-26) — hız 2,8× + iki kök bulgu

**Hız (uçtan uca, gerçek eğitim döngüsü):** 5,06 → **14,2 obje/s**.
Değişenler: worker'lı `DataLoader` (6), bf16 (kayıp daima fp32'de),
**bölge kırpma**, `n_sup` 3, coarse-to-fine kapalı. VRAM 7,25 → 2,8 GB.

- `scripts/lrm/crop.py` — OpenLRM/TripoSR bölge kırpma. Supervision hedefi
  `U[64,192]` çözünürlükte render'dan kırpılan **64² ön-plana yanlı yama**.
  Işın bütçesi sabit, yamadaki obje oranı çok yüksek. **9 test**, en kritiği
  ışın denkliği (kırpılmış `(i,j)` = tam karedeki `(i+ax, j+ay)`).
- **Coarse-to-fine KALDIRILDI** (`--coarse_frac 0`, bayrak A/B için duruyor).
  Ölçüldü: 1,25× hız (koşunun %11'i) veriyordu ama geçişte mask kaybını 2×
  sıçratıyor (540 adım toparlanma) ve kaba fazın 8000 adımında top-1 hiç
  şanstan yukarı çıkmamıştı. 8500'deki 3 dB "sıçrama" **ölçüm artefaktıydı**
  (val hep 128'de ölçülüyor, model 64'te eğitiliyordu) — top-1 o an kıpırdamadı.

**🔑 K2 çözüldü — "bf16 rengi öldürüyor" YANLIŞ TEŞHİSTİ.**
Sabit 32 uid, aynı seed, 2000 adım (`bench_uids_32`):

| kol | PSNR | top-1 | oran | not |
|---|---|---|---|---|
| fp32 | 17,82 | %31 | 0,876 | sağlıklı |
| bf16 | 15,52 | %3 | 1,244 | **1500 adım `acc=0.0000` çöküşünde** |
| **bf16 + `density_bias 1.0`** | **18,42** | **%41** | **0,823** | hiç çökmedi |

Renderer izole test edildi: bf16'da bağıl hata **0,0003** — sayısal sorun YOK.
Gerçek mekanizma: **tarife bıçak sırtı.** `acc→0` olunca softplus gradyanı ölür
ve model soğurucu durumdan çıkamaz. Aynı çöküş üç kez ısırdı: `M_base`,
`M_enc4`, `K2_bf16`.
**Panzehir `density_bias`** — `nerf.py`'de yazılıydı ama `LRM.__init__` hiç
bağlamıyordu (K4). Bağlandı (`--density_bias`, `--noise_std`), 4 test eklendi.

#### Veri: denetim + nihai split (2026-08-26)

- `scripts/audit_dataset.py` — **14.454 obje × 16 görünüm** tarandı.
  Eksik/bozuk meta, bozuk PNG, boş render, kamera/intrinsic tutarsızlığı,
  **kenar taşması: hepsi 0**. Sorunlu: 79 (%0,55 — 62 tek renk, 18 karanlık).
- **`bound` kalibre edildi (E8):** obje yarıçapı medyan 0,453 / p99 0,512 /
  **maks 0,540** ⇒ `bound` **0,6 → 0,552**, aynı ızgarada **1,28× etkin
  hacim yoğunluğu**.
- **⛔ "Kopya" bulgusu YANLIŞ POZİTİFTİ.** Tek-görünüm (kanonik[0]) imzası düz
  objelerde yanılıyor: plastik kılıflı koleksiyon kartları ön/arka açıdan
  kenardan görünüp ince çizgiye iniyor. **4 kanonik açıyla: 0 kopya, 0 sızıntı.**
  → Bu veri setinde tek görünüme dayanan imza kullanma.
- Gerçek sorun: **girdi görünümü boş 134 obje (%0,93)** — elendi.
- `scripts/build_split.py` → **`dataset/train_list_v2.json`**:
  **train 12.320 / val 1.424 / test 500**. 210 obje elendi. Küme bölünmesi 0.
  **Test seti hiçbir tarife kararında kullanılmaz** (E4).

### Eğitim öncesi netleştirme (2026-08-24 gecesi)

Tam liste + gece koşan deneyler: **`docs/egitim-oncesi-netlestirme-plani.md`**.

- **Aydınlatma kararı: DEĞİŞTİRİLMİYOR.** Referans `blender_script.py` tek geniş
  yumuşak tepe ışığı kullanıyor; bizde ek olarak **alt dolgu** (z=−3, energy 400)
  + ambient 1.5 var. Gerekçe: `TriplaneNeRF.forward(feats)` **bakış yönü almıyor**
  → model bakış-bağımsız görünüm dışında bir şeyi temsil edemez; düz aydınlatma
  kapasiteyle eşleşiyor (oracle tavanı 25.04 dB bunu doğruluyor). Alt dolgu, yeni
  alt görünümler için zaten gerekli. Karanlık veri = bilinen "renk siyaha çökme"
  tuzağı. **Bedeli:** zayıf gölge-ile-şekil ipucu — tek-girdi kalitesi darboğaz
  çıkarsa 32 objeyi iki ışıkla render edip `bench_overfit` ile karşılaştır (~35 dk).

- **`train_lrm.py`'de sayısal val EKLENDİ** (`build_val_probe` + `val_metrics`).
  Önceden sadece görsel GT|tahmin ızgarası vardı → genelleme hakkında tek sayı yok.
  Tasarım: girdi = kanonik[0], hedef = kanonik[1..2]. Kanonik açılar tüm objelerde
  aynı olduğu için metrik objeler arası karşılaştırılabilir. Çıktı: PSNR, top-1
  (şans = 1/N), `mse/ortalama-baseline` oranı (1.0 = blob çöküşü).

- **Sabit görünüm sayısı hardcode'ları temizlendi**: `render_object.py`'de
  `canonical_indices` artık görünüm listesinden türetiliyor, `run_batch.is_done`
  beklenen sayıyı şemadan alıyor (sabit 16 kalsaydı sphere20'de yarım render
  edilmiş objeler "tamam" sayılırdı), `dataset.py` girdi tavanı `min(4,…)` yerine
  `max_input` parametresi.

- **A/B kurarken tek değişkenin gerçekten tek olduğunu komut satırından doğrula.**
  Çerçeveleme A/B'sinin "yeni" kolu `--render_dir` almamıştı → varsayılan (eski)
  dizini okuyordu, yani A kolunun tekrarıydı. Aynı hata daha önce EEVEE/Cycles
  karşılaştırmasında uid üzerinden yapılmıştı (uid kamera açılarını seed'ler).

## 🚦 KOŞU ÖNCESİ KONTROL — ZORUNLU

> **Bu bölüm 2026-08-27'de kullanıcı talebiyle eklendi.** Sebep: art arda yapılan
> yapılandırma hataları saatler kaybettirdi. Kural basit: **uzun bir işi
> başlatmadan önce ne koşacağını fiilen oku.** "O bayrağı ben vermedim" bir
> savunma değil — varsayılanlar da senin sorumluluğun.

**15 dakikadan uzun sürecek HİÇBİR koşu, önce bu kontrol yapılmadan başlatılmaz.**

```bash
python scripts/train_lrm.py <TAM KOMUT> --dry_run     # 10 saniye
```

`--dry_run` ETKİN konfigürasyonu basar ve çıkar. Satır satır niyetle karşılaştır.
Otomatik uyarı verdiği tuzaklar (üçü de fiilen yaşandı):

- `teacher_subset > 0` → ortak öğretmen fazı açık (Blok 2'de terk edildi)
- `warmup > 0.2 × steps` → kısa kolda warmup koşunun yarısı
- obje başına maruziyet < 40 → yetersiz eğitimi tarife farkı sanma riski

**Diğer koşu tipleri için aynı disiplin:**

| kontrol | neden (fiilen yaşandı) |
|---|---|
| uid dosyası doğru split'ten mi? | `bench_uids_1024.json` eski split'ten; 97 val + 39 test objesi içeriyordu → sızıntı. Artık `train_lrm` reddediyor. |
| A/B kollarında tek değişken gerçekten tek mi? | İki kez sessizce aynı/farklı veri okundu (`--render_dir` verilmedi; uid kamera açılarını seed'liyor) |
| çıktı yolu bir öncekini ezer mi? | 16.000 adımlık checkpoint 40 adımlık duman testiyle ezildi → `--ckpt_dir` + `--tag` kullan |
| metrik kalibre mi? | `rel` üzerine eşik yazıldı, kalibre değildi → `docs/metrik-sartnamesi.md` §5 |

**Kod yamalarken:** her `str.replace` için `assert old in s`. Eşleşmeyen replace
**hata vermez, sessizce hiçbir şey yapmaz** — bu da bir kez yaşandı (`--dry_run`
imzası eklenmedi, koşu `TypeError` ile düştü). Yazmadan önce `ast.parse`.

**Bir koşu bittiğinde:** sonucu raporlamadan önce sayının makul olup olmadığına
bak. "Öğretmen F@1% = 0.000" imkânsızdı ve teşhis edilince eşik seçimi hatası
çıktı. Makul olmayan sayı raporlanmaz, teşhis edilir.

### 2026-08-27 — tam eğitim öncesi son blok

**▶ Başlatma talimatı: [`docs/TAM-EGITIM-BASLATMA.md`](docs/TAM-EGITIM-BASLATMA.md)**
Komut, her bayrağın gerekçesi, nöbet tablosu (ne görünce kes), bitiş eval'leri.

**Sıfırdan render kaybı ölçekte ÇALIŞMIYOR — ikinci kez ölçüldü.**
1024 objede, referans tarifenin tamamıyla (normalize_cams, unfreeze 4, bf16,
bölge kırpma): kol A 6000 adımda **8,8 dB**'de kaldı (taban 17,36!). Önizleme
teşhisi koydu: model `bound` küpünü dolduran yarı-saydam kütle basıyor,
`acc≈0.50` sabit. Kol D (`mask_fg_weight 0`) ters yöne, boş sahneye çöktü.
→ **3 aşamalı zincir zorunlu**, tercih değil.

**Öğretmen 12k objeye ÇIKAMAZ (sert VRAM sınırı).** `fit_teacher.py:65` tüm
triplane'leri tek GPU parametresinde tutuyor: obje başına 1,57 MB ⇒ 12.320 obje
= 19,4 GB (Adam'la 3×). Tavan ~2.000 obje. **Bu yüzden 3. aşama (render ince
ayarı) tüm veriyi gören TEK aşama.**

**3. aşama kararı: KALIYOR.** `distilled_v2` ↔ `C2`, 128 obje:
in-sample PSNR 24,33 → 22,00 (kötü), held-out IoU 0,540 → **0,574**,
top-1 %30,5 → **%33,6**, CLIP +0,015, LPIPS −0,013 (iyi). C2'nin 3. aşaması
**hiç yeni veri görmedi** (distilasyonla aynı 1024 obje); tam koşuda
12.320'nin **11.296'sı (%92) yeni**.

**Yüzeydeki "köpük" — kaynağı ölçüldü, mimari DEĞİŞMİYOR.**
`nerf.py` sadece triplane özelliği alıyor (konum kodlaması yok) + `grid_sample`
bilineer 64² ⇒ alan bir hücre altında yapı taşıyamaz. Spektrum: gücün %99,9'u
**0,97 hücre**, %95'i 3,5 hücre (~0,06 dünya birimi) üstünde. Öğretmende de,
öğrencide de aynı ⇒ veri/transformer suçlu değil.
Triplane 128²: +%5 süre ama %95 bandı sadece **−%6** (PSNR +0,5 dB) **ve
öğretmen bankasını geçersiz kılar** → reddedildi. Çözüm son-işlem:
`extract_mesh.py --smooth N` (Taubin, hacim korumalı).

**İki Faz C hatası — sadece çıktıya bakarak bulundu:**
1. **glTF Y-up dönüşümü yoktu** (dünyamız Z-up) → three.js'te her obje yan yatardı.
2. **Renkler sRGB→lineer çevrilmiyordu** → glTF ikinci kez aydınlatıp yıkıyordu.

**`--resume` KIRIKTI.** PyTorch 2.6+ `torch.load` varsayılanı `weights_only=True`;
künyedeki `torch.__version__` bir *TorchVersion nesnesi* olduğu için künyeli hiçbir
checkpoint okunamıyordu (17 çağrı noktası). Kök neden düzeltildi (künye alanları
düz string) + uçtan uca test edildi.

**Yeni güvenlik ağları:**
- Checkpoint **render künyesi** taşıyor (`density_bias`/`bound`); `--init_from`'da
  uyuşmazsa koşu **başlamıyor**. (Kol C bu hatayla 24 → 16,3 dB vermişti.)
- `--snapshot_every`: `last_*.pt` eziliyor; kalite ortada tepe yapıp düşerse
  (C2 tam bunu yaptı) en iyi model kaybolmasın.
- Önizlemeler **koşu başına klasöre**; önceden farklı koşular birbirini eziyordu.
- `--mask_fg_weight` RGB'nin `fg_weight`'inden ayrıldı.

**⚠️ Ölçüm dersi:** `eval_suite` varsayılanı `--split train` (**in-sample**),
`train_lrm`'in val probe'u **held-out**. Aynı checkpoint 24,33 ↔ 16,87 dB.
Bu ikisi karşılaştırılamaz. Dünkü "ölçek eğrisi düz" sonucu da in-sample'dı,
yani genelleme hakkında hiçbir şey söylemiyor.

**Testler: 137 → 155.** Silinen yetim scriptler: `diag_attn_cond`,
`diag_dino_color`, `diag_train_vs_val` (hepsi `eval_suite --split` ile ikame).

### 2026-08-27 gecesi — bağımsız denetim + girdi kırpma A/B'si

**▶ Başlatma talimatı: [`docs/TAM-EGITIM-BASLATMA.md`](docs/TAM-EGITIM-BASLATMA.md)**
(komut, her bayrağın gerekçesi, belirsizlik bantlı nöbet tablosu, açık işler)

Üç bağımsız denetçi (model/eğitim, metrikler, veri/render) projeyi eleştirdi;
**7 kritik bulgunun 7'si de kodla doğrulandı.** Hepsi düzeltildi. Testler 137 → **164**.

**⛔ `meta.json` extrinsic'leri ORTONORMAL DEĞİL** — 12/12 objede, ölçek 0,31–893.
Ölçek eksenler arası düzgün ve ışın yönleri normalize edildiği için render'lar,
öğretmen fitleri ve eğitim **etkilenmiyor** (bağımsız uzay-oyma doğrulaması:
yeniden izdüşüm IoU 0,990). Ama ham matrisi okuyan yerler bozuktu.
→ **`normalize_cams` ve cross-attention hakkındaki TÜM sonuçlar geçersiz**;
kol D'nin çöküşü de `mask_fg_weight`'e atfedilemez. `canonicalize` düzeltildi.

**⛔ "distilasyon 0,71'de takıldı" ÖLÜ SAYI** — o koşu %27'de kesilmişti.
Gerçek plato **`val_rel = 0.3029`** (`blok2_distill.log`). Mimari taban 0,177
(ölçüldü) ⇒ distilasyon tavanına neredeyse doymuş; kalan pay `TriplaneHead`'de.

**⛔ GİRDİ KIRPMA A/B'Sİ: REDDEDİLDİ.** Sıkı kırpma conditioning sinyalini
2,72× artırıyor (kaplama 0,092→0,250, DINOv2 patch 24→64) ama 250 obje ×
1500 adımda held-out **top-1 %35,9 → %23,4**, IoU 0,482 → 0,447. In-sample'da
ise DAHA İYİ (22,72 ↔ 22,48). → **Kırpma ezberi kolaylaştırıp genellemeyi bozuyor.**
Mekanizma: tam karede objenin görüntüdeki büyüklüğü *gerçek ölçeğinin ipucu*;
kırpma bunu yok ediyor.
🔴 **Faz C sonucu: kullanıcı fotosu SIKI KIRPILAMAZ**, `fit` çerçeveleme
konvansiyonu yeniden üretilmeli.

**Ölçüm altyapısındaki kritik hatalar (hepsi düzeltildi):**
- Öğretmen **öğrencinin** NeRF'iyle render ediliyordu → TAVAN çizgisi eğitim
  boyunca kayardı (%10 kayma = 1,6 dB). `render_view(nerf=...)`.
- **In-sample probe yoktu** → "az mı eğitildi / genellemiyor mu" ayırt edilemezdi.
  Eklendi; val satırının altında `ACIK (in-sample − val)` basılıyor.
- `eval_suite --split` **sessizce yok sayılıyordu** (öğretmen dosyası varsa).
- `eval_geometry`'de **`--split` yoktu** → held-out geometri hiç ölçülemezdi.
- F-score'un **iyi** kuyruğu **kötü** etiketiyle raporlanıyordu (p10/p90).
- `eval_suite` **obje-başına veriyi atıyordu** → "hangi objeler çöktü" geri gelmezdi.
- SSIM `eval_suite` ve `metrics.py`'de farklı parametrelerle (0,053 sistematik fark).
- **Eğitim tekrar üretilemezdi** (`random.Random()`, OS entropisi) → iki A/B kolu
  aynı komutla farklı veri görüyordu. `set_epoch` + seed.
- Önizlemeler düz klasörde, farklı koşular birbirini eziyordu.

**Yeni araç:** `scripts/kosu_raporu.py --tag X` → eğriler (PSNR+TABAN+RAKİP,
oran, top-1, çöküş dedektörü) + önizleme zaman serisi + ön-kayıtlı kapı tablosu.

**⚠️ `--lr 4e-4` tek ölçülmemiş hiperparametre** — OpenLRM'den kopyalandı ama
onun çok-GPU global batch'i kopyalanmadı (bizimki 16). Nöbet maddesi eklendi.

**Yama disiplini eklemesi:** `assert old in s` yetmiyor — **`s.count(old)`**
kontrol et. `str.replace` iki `query` tanımını birden değiştirdi ve testler yakaladı.

### 2026-08-29 — V3 koşusu: iki uzman incelemesi + zincirde kök hata

**▶ Koşu betikleri:** `scripts/zincir_v3_asama23.sh` (2+3), `zincir_v3_bekle.sh` (bağlayıcı).
1. aşama ayrı süreç olarak başlatıldı — **koşan bash betiğini asla düzenleme**,
   bash betiği bayt konumundan artımlı okur, yürütmeyi bozar. Aşamaları ayır.

**🔑 KÖK HATA: `set_epoch()` üç aşamanın yalnız birinde çağrılıyordu.**
`grep -rn set_epoch scripts/` → sadece `train_lrm.py`. `fit_teacher.py` ve
`distill_lrm.py` hiç çağırmıyordu ⇒ `LRMDataset._epoch` daima 0 ⇒ RNG donuk ⇒
**her obje 16 görünümün SABİT 4'ünden öğreniliyordu, 12 render ölü.**
Bu, yukarıda "Düzeltildi + regresyon testleri" diye kayıtlı hatanın ta kendisi;
düzeltme bir çağrı noktasına uygulanmış, ikisine taşınmamıştı.
Ölçüldü: `obj0` supervision `[10,8,11,9]` — 3 erişimde de birebir aynı.
Düzeltme sonrası 200 adımda 186 benzersiz kombinasyon, **16/16 görünüm**.

**⚠️ Hata kendi ölçümünü gizliyordu.** `fit_teacher`'ın değerlendirme bloğu da
`ds[i]` okuyor ⇒ eğitildiği aynı 4 donmuş görünümde ölçüyordu. Yani
**"öğretmen tavanı 30,7 dB" rakamı in-sample'dı.** Artık ölçüm sabit
`EVAL_EPOCH = 10_000_003`'te yapılıyor (eğitimde asla kullanılmayan epoch)
⇒ eski dB'lerle **doğrudan karşılaştırılamaz**, daha düşük ama dürüst.

**Weight decay norm/bias'lara uygulanıyordu (referans uygulamıyor).**
Ölçüldü (`last_TAM_asama3.pt` ↔ orijinal DINOv2): `norm.weight` ölçek 0,9659,
yön değişimi ~1e-5 ⇒ **saf çürüme, öğrenme değil**. Saf AdamW decay tahmini
0,9697 ↔ gözlenen medyan 0,9703. `train_lrm.py` + `distill_lrm.py`'de
`ndim<=1` parametreleri `weight_decay=0.0` grubuna alındı (110 tensör, 0 kaçak).
*Yan bulgu:* `[encoder] en cok degisen` log satırı yanlış tensörü gösteriyor —
`blocks.11.ls1.gamma` delta'sı büyük ama yön değişimi 0,0005; gerçek öğrenme
hiç raporlanmayan mlp/attn matrislerinde (**%24 dönme**). A/B'de yanlış karar verdirir.

**REDDEDİLEN iki tavsiye** (ikisi de deney; koşuda zaten 4 mimari değişiklik var):
- `--train_encoder`: `unfreeze_last 4` son 4 bloğun matrislerini **%24 döndürmüş**
  ve tek başarılı gerçek-foto sonucumuz (2 görünüm IoU 0,691) **tam o
  konfigürasyonda** alındı. `patch_embed` + blok 0–7 domain transferini taşıyor,
  ölçüm yok. Lehine de kanıt yok: `train_lrm`'de hiç denenmedi (`M_encall`
  sadece `bench_overfit`, 32 obje, geçersiz matris). → sonraki tek-değişkenli A/B.
- `--mixed_p 0.25`: `elev_TAM_taban.json` tersini söylüyor —
  IoU 0,619 (+20°) → 0,489 ([40,55), −%21) → 0,419 ([55,75), −%32);
  kanonik-dışı 12 görünümün **%55,5'i [10,55) telefon-fotosu bandında**.
  Yarıya indirmek Faz C açığını kapatan sinyali yarıya indirirdi. Val probe'un
  dar olması eğitimi ona göre ayarlama gerekçesi değil (Goodhart).

**KABUL EDİLENLER** (hepsi tespit edilmiş defekt): `set_epoch`, norm/bias'a wd yok,
`--resume` (kodla doğrulandı: ckpt yokken no-op), **`eval_photo4` zincire eklendi**
— iki uzmanın tartışması da gerçek-foto üzerineydi ama 19 saatin sonunda o konuda
tek sayı üretilmeyecekti.

**⚠️ Veri sorusu şimdilik KAPALI.** ACIK (in-sample − val) = +0,57 dB; prob kümesi
zorluğu için kalibre edilince (`taban_psnr` train[:64] 17,68 ↔ val[:64] 17,36)
**+0,25 dB**. Model eğitim verisini bile kuramıyor ⇒ ezberleme yok ⇒ **daha fazla
veri şu an çare değil**, darboğaz kapasite/tarife. Yeniden açma koşulu: bu koşu
sonrası ACIK > 1,5 dB.

**Yama disiplini — yine ısırdı.** `assert old in s` + `s.count(old)` yetmedi:
çapa olarak `"DEV = "` seçtim, satırın **ortasına** denk geldi ve `DEV`'i
`DEV = EVAL_EPOCH = 10_000_003` yapıp stringi kopardı. **`ast.parse` geçti**
(sözdizimi geçerliydi). → Çapa daima **satır sonu (`
`) içermeli**; ve yamadan
sonra grep değil **davranışsal doğrulama** yap (sabitin değerini `ast` ile oku).

**🔍 KEŞKİNLİK KAYBI DİSTİLASYONDA DOĞUYOR (görsel kanıt, 2026-08-29).**
`scripts/kanit_asama12.py` → `dataset/lrm_val_previews/KANIT_asama12.png`
(GT | öğretmen | öğrenci, aynı obje, aynı kamera, aynı renderer).
8 train objesi: öğretmen **29,93 dB**, öğrenci **26,87 dB** (−3,06).
Ama asıl bilgi görselde: öğretmen taneli kaya dokusunu, makasın ince kollarını,
bankın sırt çıtalarını yakalıyor; **öğrencide taneli kaya düz gri kütleye,
banka çıtaları yok oluşa, makas tanınmaz lekeye** çöküyor (o obje −6,47 dB).
→ Kullanıcının "şekil düzeliyor ama keskinlik/detay net değil" şikâyetinin
kaynağı **3. aşama değil, distilasyon**. Objeler öğrencinin eğitim setinde
⇒ genelleme değil **kapasite sınırı**.

**⚠️ Sayılar bunu gizledi.** `val_rel = 0,2624` "öğrenilebilir sinyalin %97'si"
diyordu; kaçırılan %3 tam da keskinliği oluşturan yüksek frekans bandı.
İpucu metrikte vardı — `rel_smooth 0,1707` (yumuşatılmış hedefe karşı iyi) ↔
`val_rel 0,2624` farkı bu bandı ölçüyor — ama **görsele bakmadan okunmuyordu.**
`distill_lrm.py` hiç önizleme yazmıyor (`fit_teacher` tek kare yazıyor); bu
yüzden gece boyunca yalnız sayı takip edildi. → [[ciktiya-bak-metrige-degil]]

**Tekrar üretilebilirlik:** kod commit'lenmedi (künye `+kirli`); tam anlık görüntü
`dataset/lrm_logs/v3_kod_snapshot/` + `v3_kod_degisikligi.patch`.

Testler: 173 → **175**.

### 2026-08-31 — V3 sonucu kapandı (tam ölçüm tablosu)

**Tam tablo + okuma: [`docs/SIRADAKI-ISLER.md`](docs/SIRADAKI-ISLER.md) §0.2.**
Zincirde çağrılmayan 4 ölçüm sonradan koşturuldu (`scripts/eksik_evals_v3.sh`):
`eval_suite` in2/in4, `eval_geometry`, `kanit_asama12`, `extract_mesh`.

**Özet:** V3 ↔ TAM_asama3 (ikisi de adım 30800, test split).
Kanonik val ve test PSNR **hareketsiz** (19,38 ↔ 19,49). Kazanç iki yerde:
**geometri F@1% 0,260 → 0,298** ve **elevation [55,75) IoU 0,419 → 0,518**.
Kayıp: **keskinlik**. Kapı 1 (çöküş) 4/4 geçti, Kapı 2 (kalite) 4/4 kaldı.
→ `set_epoch` düzeltmesi (16 görünümün 4'ü yerine 16'sı) **açı dayanıklılığına**
yaradı, keskinliğe değil. En iyi checkpoint `last_V3.pt` (24k↔30k farkı 0,05 dB).

**🔴 3. aşama keskinliği GERİ KAZANMIYOR, KAYBEDİYOR.** Aynı 8 train objesi,
her sistem kendi decoder'ıyla: öğretmen **29,93** → distile öğrenci **26,87**
→ 3. aşama **22,52 dB**. Görselde arı şeritleri, makas kolları, iki ayrı deniz
kabuğu tamamen kayıp (`dataset/lrm_val_previews/KANIT_asama13_V3.png`).
Çelişki değil **takas**: aynı koşuda held-out val 16,76 → 19,38 çıktı; 3. aşama
1.024 objedeki keskinliği 12.320 objeye yayıyor. Keskinlik açığı artık iki
yerde doğuyor: distilasyonda −3,06, 3. aşamada −4,35. **Kapasite sınırı işareti**;
3. aşamayı kaldırma gerekçesi değil (held-out + elevation kazandıran tek aşama o).

**`kanit_asama12.py --student_nerf` eklendi** — 3. aşama checkpoint'i için
ZORUNLU. Render ince ayarında NeRF de eğitildi; öğrenci triplane'ini öğretmenin
decoder'ından geçirmek geçersiz ölçüm verir.

**⚠️ Zincir `|| true` ile sarıyor.** V3 zinciri hata vermedi ama `eval_geometry`,
`eval_suite --n_input 2/4` ve `extract_mesh` **hiç çağrılmamıştı** — sessiz
başarısızlık değil, eksik kapsam. Zincir yazarken "hangi ölçüm YOK" listesini de
kontrol et, sadece "hangi ölçüm hata verdi"yi değil.

### 2026-08-31 — keskinlik: ölçü aracı + triplane 128² reddedildi

**Yeni araç: `scripts/bench_keskinlik.py`.** PSNR'in gizlediği bandı ölçer.
Üç sayı basar, üçü de gerekli:

| sütun | ne ölçer | tuzak |
|---|---|---|
| **enerji** | `std(x − 3×3 blur)`, sadece obje içi | çok olması iyi demek DEĞİL |
| **kor** | HF haritasının GT ile korelasyonu | hizalamaya aşırı duyarlı |
| **kor+kaydır** | ±3px kaymaya izin verilirse | temsil ↔ hizalama ayrımı |

⚠️ **Siluet maskelenmezse metrik çöp.** İlk ölçümümde öğretmen "GT'nin %99'u"
çıktı; ölçtüğüm şey dokunun değil, beyaz zemine karşı devasa siluet basamağıydı.
Obje içi 7px aşındırılınca gerçek sayı **%45**.

⚠️ **Enerji tek başına yanıltır — fiilen ısırdı.** `denetim 64 / tp 128²` kolu
GT'nin **%109'u** enerji üretti; görselde bu dikey çizgi artefaktıydı. Izgara
denetimden ince olunca alan yetersiz kısıtlanıp gürültü üretiyor. `last_V3`'ün
"%31 enerjisi" de öyle: korelasyonu **0,007** ⇒ neredeyse saf ızgara artefaktı
(görselde kaya üzerinde gözle görülür kafes deseni).

**2×2 taraması (oracle, 12 obje, encoder/transformer YOK):**

| denetim | triplane | enerji | kor+kaydır |
|---|---|---|---|
| 64px | 64² *(mevcut tarife)* | %59 | 0,085 |
| 64px | 128² | %109 | 0,095 |
| **128px** | 64² | %54 | **0,122** |
| **128px** | **128²** | %81 | **0,127** |

- **⛔ Triplane 128² REDDEDİLDİ.** Aynı denetimde 0,122 → 0,127 (**+%4**).
  Bedeli: 4× triplane belleği + `TriplaneHead` değişikliği + öğretmen bankası
  geçersiz + tüm zincir yeniden. Kazanç bedelin yanında yok.
- **✅ Denetim çözünürlüğü 64 → 128 işe yarıyor:** kor 0,085 → 0,122 (**+%44**),
  üstelik enerji *düşerek* (%59 → %54) — daha az ama daha doğru doku.

**⚠️ Kendi hatam, kayda geçsin:** aynı gün önce "denetim çözünürlüğü bağlayıcı
duvar değil" demiştim. O yargı **yalnız enerjiye** bakıyordu (öğretmen %45,
tek-görüntü 64px tavanı %15 ⇒ çoklu-görünüm birleştirmesi çalışıyor — bu kısım
doğru). Doğruluğa bakınca test edilen iki koldan **güçlü olanı buymuş**.
→ Bir metrik ölçeği ölçüyorsa yapıyı da ölç.

**Geometri bağlantısı:** hücre = `2×BOUND/64` = **0,01875 dünya birimi**.
1 birimlik objede bank çıtası/sandalye ayağı ~1 hücre ⇒ ince yüzey olarak
temsil edilemiyor, **şişerek** temsil ediliyor. Kullanıcının "sandalye
kalınlaşmış" tespitinin kaynağı bu, ayrı bir kusur değil.

**Yeni bayraklar:** `fit_teacher.py --tp_res` (triplane çözünürlüğü artık
gömülü değil) ve `--n_sup` (adım başına supervision görünümü; `--res 256` +
`n_sup 4` tek adımda ~50M nokta ⇒ OOM).
`kanit_asama12.py --student_nerf` (3. aşama ckpt'i için ZORUNLU).

**⛔ `bench_triplane_fit` ile alınmış "128² fayda etmiyor" sonucu GEÇERSİZDİ:**
o betiğin varsayılanı `--res 64` ve orada bir hücre zaten **0,88 piksel**
(piksel-altı) — ızgarayı büyütmek tanımı gereği hiçbir şey veremezdi.
Çözünürlük sorusu sorarken denetim çözünürlüğünü sabitleyip **ayrı** değişken yap.

### 2026-09-01 — "neden 512 değil": maliyet matrisi + BELLEK DUVARI

**Soru:** öğretmene neden 64 piksel denetim veriyoruz, 512 versek ne olur?
**Cevap:** üretim öğretmeni (`teacher_1024_v3.pt`) künyesi `res=64` — diskteki
512×512 master render **64'e küçültülüp** veriliyor, pikselin %98,4'ü atılıyor.
Zincirin hiçbir aşaması 64'ün üstünü görmüyor (3. aşama da 128'den kırpılan 64² yama).

**`scripts/bench_res_maliyet.py` (YENİ)** — 2 obje, 1 görünüm, warmup dışı medyan:

| konfig | hücre_px | ms/adım | × üretim | VRAM |
|---|---|---|---|---|
| **res64 tp64** *(üretim)* | 0,88 | 16 | 1,0× | 0,9 GB |
| res128 tp64 | 1,77 | 52 | 3,2× | 3,2 GB |
| res256 tp64 | 3,53 | 251 | 15,3× | 6,2 GB |
| res256 tp128×64 n192 | 1,77 | 2116 | 129× | 14,9 GB |
| res512 tp64 | 7,06 | 1002 | 61× | 6,3 GB |
| res512 tp128×64 | 3,53 | 8694 | 530× | 14,9 GB |
| res512 tp256×64 n320 | 1,77 | 240509 | 14651× | **25,0 GB** |

`hücre_px` = bir triplane hücresinin denetim görüntüsündeki piksel boyu.
**>2 ise denetim temsilden ince ⇒ bedeli ödenir, karşılığı gürültüdür.**
Ölçülmüş kanıt: `last_V3` GT'nin %31'i HF **enerji** üretiyor ama korelasyon
**0,007** — kayanın üstündeki gözle görülür kafes deseni. Denetim temsili aşınca
model doku değil ızgara artefaktı uydurur.

⚠️ `res512 tp256` satırındaki 240 sn/adım **hesaplama süresi değil**: 25,0 GB
ayırma 16 GB karta sığmıyor, sürücü sistem RAM'ine taşıyor. "Yavaş" değil, **sığmıyor**.

**🔑 ASIL DUVAR BELLEK, SÜRE DEĞİL.** `fit_teacher.py:65` tüm objelerin
triplane'ini **tek GPU tensöründe** tutuyor:

| triplane | MB/obje | 1024 obje + Adam(3×) | 16 GB'a sığan obje |
|---|---|---|---|
| **64²×32** *(mevcut)* | 1,57 | **4,8 GB ✓** | 2543 |
| 128²×32 | 6,29 | 19,3 GB ✗ | 635 |
| 128²×64 | 12,58 | 38,7 GB ✗ | 317 |
| 256²×64 *(512 için gereken)* | 50,33 | **154,6 GB ✗** | **79** |

Yani triplane büyüdükçe öğretmen bankası küçülür ⇒ öğrencinin distile olacağı
obje sayısı düşer (V3: 1024). Bu **fizik değil kod**: bir adımda yalnız `batch`
obje kullanılıyor, banka CPU'da tutulup adım başına o kadarı GPU'ya alınabilir
(PCIe ~10 ms, adım ~600 ms ⇒ %2). **Henüz yapılmadı.**

**✅ ÇIKIŞ YOLU: bölge kırpma öğretmende YOKTU.** `scripts/lrm/crop.py` 3. aşamada
kullanılıyordu, `LRMDataset` destekliyordu, ama `fit_teacher.py` bayrağı hiç
açmıyordu. Eklendi: **`--region / --render_low / --render_high`**.
Işın maliyeti `region²`'de SABİT, detay ölçeği `render_high` kadar.
Ölçüldü: `region 128 <- U[192,384]` + tp128×64 + n192 = **~327 ms/adım**,
`res 256` tam kareye göre **~6,5× ucuz** — üstelik detay tavanı 384 piksel.

**Ölçüm tuzağı kapatıldı:** eğitim kırpmalı olsa bile PSNR eğrisi + önizleme
artık **ayrı, kırpmasız, sabit** `--eval_res` dataset'inden ölçülüyor. Yoksa
kollar kıyaslanamazdı (`bench_triplane_fit --res 64` hatasının aynısı).

**Nyquist hatırlatması** (`n_samples` serbest DEĞİL, `tp_res` zorunlu kılar):
tp64 ⇒ ≥75, tp128 ⇒ ≥150, tp256 ⇒ ≥299.

**⛔ İLK KOŞU ÇÖPE GİTTİ — varsayılan tuzağı, ÜÇÜNCÜ kez.** `bench_detay.sh`
`--train_list/--renders_dir` vermedi; `fit_teacher`'ın varsayılanı
`dataset/train_list.json` + `dataset/renders` (ÖLÜ set) idi ⇒ üç kol da ölü
veride eğitildi, 45 dk gitti. Yakalayan tek şey `bench_keskinlik`'in uid
assert'i oldu; o olmasa yanlış tablo sonuç diye raporlanacaktı.
**Alınan mekanizma** (hatırlamaya güvenmek yeterli değil):
- 6 eğitim/tezgâh scriptinin varsayılanı canlı sete çevrildi
  (`fit_teacher`, `distill_lrm`, `train_lrm`, `bench_overfit`, `overfit_lrm`,
  `overfit_randbg`). Faz A araçları kapsam dışı — onlar render'ı ÜRETEN taraf.
- `LRMDataset` ölü set okunursa stderr'e büyük uyarı basıyor (sert hata değil;
  belgelenmiş `AB_ESKI` karşılaştırması kasıtlı okuyabilmeli).
- `tests/test_olu_veri_uyari.py` — `ast` ile `add_argument` varsayılanlarını
  denetliyor. **Testler 188 → 197.**
- Geçersiz kollar silinmedi: `dataset/lrm_bench/gecersiz/OLUVERI_*.pt`.
- `grep`'e `--line-buffered` + `python -u` şart; yoksa kolun içi bitene kadar
  görünmez (aynı gün iki kez yaşandı).

**SONUÇ (canlı veri, 32 obje, obje başına 100 güncelleme = üretimle aynı,
ölçüm 12 objede 256px, siluet 7px aşındırılmış):**

| sistem | enerji | GT oranı | kor | kor+kaydır | FAYDALI | PSNR |
|---|---|---|---|---|---|---|
| GT | 0,01947 | %100 | 1,000 | 1,000 | %100 | — |
| *GT 128→256 tavanı* | 0,00658 | %34 | 0,520 | 0,520 | %18 | — |
| *GT 64→256 tavanı* | 0,00293 | %15 | 0,172 | 0,231 | %3 | — |
| **A_taban** (den 64, tp64²×32, n96) | 0,01149 | %59 | **0,006** | 0,078 | %5 | 26,37 |
| **B_detay** (bölge128←U[192,384], tp64²×32) | 0,00614 | %32 | 0,040 | 0,147 | %5 | **30,31** |
| **C_hepsi** (+tp128²×64, n192) | 0,01120 | %58 | **0,068** | **0,173** | **%10** | 29,63 |

- **🔑 Üretim tarifesinin keskinliği SAHTE.** A enerjinin %59'unu üretiyor ama
  `kor = 0,006` ⇒ sıfır. Görselde kayanın üstündeki pembe hücresel desen GT'de
  yok. `last_V3`'ün 0,007'siyle aynı hastalık.
- **Denetim detayı gerçek kazanç:** B enerjiyi %59→%32 *düşürerek* kor'u
  **6,7×** artırıyor (daha az ama doğru doku). Temsil, bütçe, ölçüm aynı.
- **Büyük ızgara detayı doğru yere koyuyor:** C kor **0,068** = A'nın 11 katı.
- **⚠️ PSNR yine yanılttı:** B 30,31 > C 29,63 ama keskinlikte C kazanıyor,
  görsel de C'yi doğruluyor. Tarife seçimini PSNR'a bırakma.
- **⚠️ SORUN ÇÖZÜLMEDİ:** C'nin 0,173'ü, GT'yi 64 piksele indirip büyütmenin
  (0,231) hâlâ ALTINDA. 128px referansı (0,520) çok uzak.
- **Yeni kusur (görselden):** C'de kafes deseni. `w_tv 0,05` 64² için
  ayarlanmıştı; TV çözünürlükle ölçeklendiği için 128²'de yetersiz. Yeniden ayarla.

**Karar:** bölge kırpma denetimi **benimsenir** (bedeli ~2,5× süre, bellek
maliyeti yok). `tp 128²` kazancı gerçek ama iki iş açıyor: CPU-sayfalı banka
(1024 obje sığmıyor) + `w_tv` yeniden ayarı.

Görsel: `dataset/lrm_bench/DETAY_kare.png`. Betik: `scripts/bench_detay.sh`.

### 2026-09-02 — KESKİNLİK: kök neden bulundu (çözünürlük bütçesi)

**▶ Tam teşhis + plan + ön-kayıtlı kapılar: [`docs/KESKINLIK-TESHISI.md`](docs/KESKINLIK-TESHISI.md)**
Yeni bir keskinlik deneyi önermeden ÖNCE oraya bak.

**🔑 Keskinlik bir tarife sorunu değil, ÇÖZÜNÜRLÜK BÜTÇESİ sorunu.**
"Obje bu aşamada kaç örnek genişliğinde?" sorusu her aşamada soruldu
(100 obje; objenin en uzun kenarı karenin %54'ü):

| aşama | obje genişliği |
|---|---|
| master render 512px | 276 px |
| **encoder 224px / DINOv2 patch 14** | **8,6 patch** 🔴 öğrenci duvarı |
| transformer token ızgarası 32² | 24 token |
| **triplane 64²** | **48 hücre** 🔴 öğretmen duvarı |
| **üretim denetimi 64px** | **35 px** 🔴 öğretmen duvarı |
| ölçüm 256px | 138 px |

Üretim öğretmeni objeyi **35 piksel** genişliğinde gördü; keskinliği 138
piksellik GT'ye karşı ölçüyoruz. Denetim (35) triplane'den (48) daha kaba ⇒
fazla serbestlik gürültüyle doluyor (= `last_V3`'ün %31 enerji / 0,007
korelasyon kafes deseni). **Her deney bu üç duvardan yalnız birini gevşetti,
diğer ikisi bağlayıcı kaldı — "deneyler rastgele" hissinin sebebi bu.**

**Bulanıklık YANAL, derinlikten değil (yeni araç `scripts/diag_yuzey.py`).**
256px denetim + 1000 güncelleme: acc **0,997**, yüzey kalınlığı **1,35 px**
(keskin). Üretim tarifesi: acc 0,988, **3,64 px**. ⇒ "sisli yüzey" kök neden
değil, denetim yoksunluğunun BELİRTİSİ. Bu hipotez dalı KAPANDI.

**⛔ VERİ YOLUNDA İKİ GERÇEK DEFEKT — düzeltildi, testler 197 → 204.**
`_load_rgba` 512'lik master'ı küçültürken:
1. **Anti-aliasing yoktu.** `F.interpolate(mode="bilinear")` varsayılanı
   `antialias=False`; 8× küçültmede 8×8 bloğun 2×2'si örnekleniyordu.
   |bilinear−area| RMS, görüntünün TÜM HF enerjisinin res 64'te **%72**'si,
   128'de %47, 192'de %73. Takma ad **poza bağlı** ⇒ görünümler arası
   tutarsız ⇒ fit edilemez ⇒ optimize edicinin cevabı ortalama = **bulanık**.
2. **Premultiply sırası tersti.** Düz RGBA küçültülüp sonra alpha ile
   çarpılıyordu; Blender alpha=0'a RGB=0 yazıyor ⇒ siluete siyah sızıyor.
   Kenar pikselleri RMS: res 64'te **0,1099**, 128'de 0,0672.
   res 64'te obje 35 px; 1–2 piksellik bank çıtası/makas kolu TAMAMEN kenar
   pikselidir ⇒ **"ince yapılar kayboluyor/kalınlaşıyor"un mekanizması budur.**

⚠️ **Defekti gizleyen tesadüf:** tam 2× küçültmede (512→256) bilinear zaten
kutu ortalamasına eşit. `bench_keskinlik` ölçümü **hep 256'da** ⇒ GT temizdi,
eğitim hedefi bozuktu. Hiçbir metrik göremezdi.

**⛔ BUNDAN ÖNCEKİ KESKİNLİK ÖLÇÜMLERİNİN ÇOĞU GEÇERSİZ.** Defekt her denetim
çözünürlüğünü farklı oranda bozuyor ⇒ denetim çözünürlüğünü değiştiren her
A/B'de defekt de birlikte değişti. Geçersiz: `den_res` merdiveni,
`bench_detay` A/B/C, `kesk_2x2`, `teacher_1024_v3`/`distilled_v3`/`last_V3`
keskinlik sayıları. En az etkilenen `TAVAN` probu (denetim 256px) ama **n=4**.
`den_res` ile `TAVAN` zaten çelişiyordu (tp128 < tp64 ↔ tp128 > tp64) ve
`den_res`'in tp128 kolu **Nyquist ihlalliydi**.

**Düzeltmenin ölçülen etkisi** (üretim tarifesi, birebir aynı komut, tek fark
kod; `scripts/ab_kucultme_duzeltmesi.sh`): sahte HF enerjisi %59 → **%37**,
kor+kaydır 0,078 → **0,106** (+%36), eval PSNR 26,37 → **27,12 dB**.
Ama hâlâ keskin DEĞİL ve olamazdı: **bozuk hedefi düzeltmek, hiç verilmemiş
detayı yaratmaz.** Denetim o tarifede objeyi 35 piksel gösteriyor.

**⏳ Aşama 2 koşuyor:** `scripts/bench_ogretmen_tavani.sh` — 24 obje, denetim
bağlayıcı DEĞİL (bölge 128 ← U[256,512]), tek değişken `tp_res` ∈ {64,128,256}
+ `w_tv` kontrol kolu, Nyquist'e uygun `n_samples`. **Ön-kayıtlı kapılar ve
her sonucun sonraki işi `docs/KESKINLIK-TESHISI.md` §6'da yazılı.**

**Yeni araçlar:** `scripts/diag_yuzey.py` (opaklık + yüzey kalınlığı),
`scripts/bench_ogretmen_tavani.sh`, `scripts/ab_kucultme_duzeltmesi.sh`,
`tests/test_kucultme_dogrulugu.py`.


### 2026-09-02 — bağımsız denetim + 4 deney: KAYITLI SONUÇLARIN DÜZELTİLMESİ

**Bu bölüm önce ÖNCEKİ KAYITLARI DÜZELTİYOR.** Aşağıdaki dört madde CLAUDE.md'de
yazılıydı ve bugün kodla çürütüldü. Üzerlerine inşa etme.

**⛔ 1. `oran` metriğinin tarifi YANLIŞ yazılıydı.** Dört yerde "veri-seti-ortalaması
blob'undan daha kötü" diye geçer. Gerçeği (`lrm/metrics.py:139`):
`mean_mse = ((P.mean(0) - G)**2).mean()` — payda **TAHMİNLERİN ortalaması**, GT'nin
değil. Yani `oran > 1.0` = "model KENDİ ortalama çıktısından kötü". GT ortalamasına
göre olan sayı ayrı bir metrik: `taban_psnr` (`train_lrm.py:201`, `G.mean(0)`).
İkisi aynı rapor satırında yan yana basılıyor ve farklı şeyler ölçüyor.

**⛔ 2. "distilasyon tavanına neredeyse doymuş (plato 0,3029, mimari taban 0,177)"
GEÇERSİZ.** O çıkarım `distill_lrm.py:91`'deki "öğrenilemez taban 0,242"e
yaslanıyordu; o sayı `var(hedef − 3x3blur)/var(hedef)`, yani sadece yüksek frekans
içeriği — ona "gürültü" demek bir VARSAYIM. Oracle kolları altına indi (O1 0,1339,
O2 0,0552). Aynı mimari 0,055'e inebiliyor ⇒ V3'ün 0,3029'u doymuş DEĞİLDİ.

**⛔ 3. `bench_res_maliyet.py` maliyet matrisi ŞİŞKİN, "512 sığmıyor" sonucu YANLIŞ.**
`renderer.py:75`: `if not chunk or origins.shape[0] <= chunk: return volume_render(...)`
— ışın sayısı chunk'a EŞİTSE parçalama VE gradyan checkpoint'i sessizce KAPANIYOR.
Matris sabit `--ray_chunk 32768` kullanıyor ⇒ küçük konfigler parçasız, büyükler
parçalı: tek tabloda iki rejim. Dürüst hesap yükü ~430×, raporlanan 14.651×.
2026-09-01'deki *"25,0 GB ayırma 16 GB karta sığmıyor... 'yavaş' değil, sığmıyor"*
cümlesi yanlış — sığmayan şey `chunk=32768`.

**⛔ 4. `--amp` yardım metnindeki "rengi öğrenmiyor" uyarısı BAYATTI** (K2 bulgusu
2026-08-26'da çürütmüştü). Düzeltildi; `bench_wmask.sh` bu yüzden fp32 koşmaya
başlamıştı.

#### Yeni defekt: `mode="area"` kesirli oranlarda doğru alan filtresi DEĞİL

`imutil.kucult` `F.interpolate(mode="area")` (= `adaptive_avg_pool2d`) kullanıyor;
tam sayı olmayan ölçek oranlarında kutu sınırları tam sayıya yuvarlanıyor.
Ölçüldü (512 → out, referans: elle kurulmuş kesirli ağırlık matrisi):

| out | oran | \|area−gerçek\| RMS | ref std |
|---|---|---|---|
| 128 | 4,000 | **0,00000** | 0,072 |
| 256 | 2,000 | **0,00000** | 0,144 |
| 300 | 1,707 | 0,07866 | 0,136 |
| 384 | 1,333 | 0,08688 | 0,168 |
| 448 | 1,143 | 0,10839 | 0,180 |

`fit_teacher --render_low 256 --render_high 512` bu aralıktan TAM SAYI çekiyor ⇒
257 olasılığın sadece 2'si temiz. Mekanizma aynı sabah düzeltilen `antialias=False`
defektiyle AYNI (poza bağlı, görünümler arası tutarsız HF gürültüsü = bulanık doku)
ve o bantta ondan büyük.

**Neden kaçtı:** `tests/test_kucultme_dogrulugu.py`'nin referansı (`_dogru`)
**`mode="area"`'nın kendisiydi** — test uygulamayı kendisiyle karşılaştırıyordu.

**Düzeltme opt-in eklendi:** `OLCEK_YONTEMI=kutu` (varsayılan DEĞİŞMEDİ, `torch.equal`
ile doğrulandı). Kesirli sınırlı gerçek kutu filtresi, her oranda doğru (RMS 2e-08).
`bilinear+antialias=True` DEĞİL: kesirlide 2–5× iyi ama tam sayıda bozuyor
(out 128'de 0,03977) — bağımsız denetim onu önerdi, ölçüldü, tek yönlü doğruydu.

**Etkisi ÖLÇÜLDÜ (E-R, tek değişken, `bench_yeniden_ornekleme.sh`):**
`kor` 0,158 → 0,168 (**+0,010**), `kor+kaydır` 0,236 → 0,225, eval PSNR 32,26 → 31,86.
⇒ **Ön-kayıtlı kapı: merdiven AYAKTA.** Defekt gerçek ama tp_res sıralamasını
kirletmemiş. (PSNR'ın DÜŞMESİ AÇIKLANMADI; hipotez var, ölçülmedi.)

#### Deney 1 — öğretmen tavanı: tp_res merdiveni (24 obje, denetim bağlayıcı değil)

| sistem | enerji | GT oranı | **kor** | kor+kaydır |
|---|---|---|---|---|
| GT 128→256 tavanı | 0,00704 | %34 | 0,521 | 0,521 |
| GT 64→256 tavanı | 0,00317 | %16 | 0,190 | 0,235 |
| K64 (tp64², ns96) | 0,00781 | %38 | 0,158 | 0,236 |
| K64_notv (w_tv 0) | 0,00802 | %39 | 0,162 | 0,225 |
| K128 (tp128², ns192) | 0,01139 | %56 | 0,269 | 0,327 |
| K256 (tp256², ns320) | 0,01385 | %68 | 0,302 | 0,337 |

**⚠️ "K64, 64px tavanında" çıkarımı YANLIŞTI** — `kor+kaydır` sütunundan verilmişti
(0,236 ≈ 0,235). Birincil sütunda K64 (0,158) 64px referansının (0,190) **ALTINDA**;
eşitlik tamamen tablodaki en büyük kaydırma bonusundan (+0,078) geliyor.
`docs/KESKINLIK-TESHISI.md` §8.3 bunu zaten yasaklıyordu.

**⚠️ Ön-kayıtlı kapıların HİÇBİRİ ateşlemedi.** Kapı A (`kor+kaydır ≥ 0,35`):
K256 0,337 ⇒ HAYIR. Kapı B (her ikiye katlamada ≥ +0,05 VE monoton):
64→128 +0,091 ✓, 128→256 +0,010 ✗ ⇒ KISMEN. **tp128 kararı POST-HOC'tur.**
Lehine: son üç ölçüm üç kolda da düz (K256 33,85/33,97/33,97) ⇒ "eksik eğitildi" değil.

**KARAR: tp128, ama head genişletilerek.** Öğrencinin gizil kapasitesi
`3 × 32² × 512 = 1.572.864`; tp128²×32 hedefi **tam 1.572.864** (sıfır pay, kare
doğrusal sistem), tp256²×32 = 6.291.456 (4×, bugünkü DOĞRUSAL head'le taşınamaz).
⚠️ "ConvTranspose2d örtüşmesiz olduğu için" gerekçesi ALAKASIZ (örtüşmeli de olsa
toplam sınır aynı). Ve tam doygunluk en kötü rejimdir — `LRM.__init__` `upsample`'ı
`TriplaneHead`'e HİÇ geçmiyor (daima 2), `distill_lrm.py`'de `--tp_res` YOK.

#### Deney 2 — oracle koşullandırma (256 obje, force_n_input 1)

| kol | obje başına serbest sayı | val_rel | rel_smooth | öğrenci_std |
|---|---|---|---|---|
| O0_taban (donuk DINOv2) | 0 | 0,4094 | 0,2543 | 0,1274 |
| O1_serbest (256 tok) | 98.304 | 0,1339 | 0,1907 | 0,1723 |
| O2_bol (1024 tok) | 393.216 | 0,0552 | 0,2440 | 0,1827 |

(hedef std 0,2219)

- **`rel_smooth` BİRİNCİL METRİK OLAMAZ:** mükemmel tahminci (`pred=hedef`)
  `nf/(1−nf) = 0,309` alır, bulanık tahminci (`pred=blur`) **0,000** alır. Metrik
  doğru cevabı cezalandırıyor; üç kol da "mükemmel tahminci"den iyi skor aldı.
  0,309 referans çizgisi basılırsa kullanılabilir hale gelir.
- **Kapasite hipotezi ELENDİ:** transformer+head bu hedefi üretebiliyor.
- **AMA kapının önerdiği eylem (girdi 224→448) BURADAN ÇIKMAZ.** Serbest token'lar
  obje kimliğini ezberliyor; O2'nin obje başına serbest sayısı hedefle **birebir
  eşit** (üst sınır değil, dejenere kontrol). Ve üç kol da tek girdi görünümüyle
  koştu ⇒ objenin arkası ilkesel olarak belirsiz.
- **⛔ `val_rel` "val" DEĞİL** (`distill_lrm.py:195`): `range(0, min(n,64), 8)` =
  **8 obje, hepsi EĞİTİM setinden**, sadece farklı epoch/kamera. In-sample.
- ⚠️ Hedef `teacher_1024_v3.pt` = res 64, **küçültme düzeltmesinden ÖNCE** oturtulmuş.

#### Deney 3 — w_mask taraması (1024 obje, 6000 adım, sıfırdan, batch 16, bf16)

| kol | PSNR | top-1 | oran | acc | acc/GT |
|---|---|---|---|---|---|
| WM_0p0 | 16,53 | %10,9 | 0,896 | 0,210 | 2,40× |
| WM_1p0 | 17,09 | %9,4 | 0,967 | 0,123 | 1,40× |
| **WM_0p25** | **17,88** | **%15,6** | 0,912 | 0,0999 | **1,14×** |
| *taban (GT ortalaması)* | *17,48* | | | | |
| *rakip (en-yakın komşu)* | *16,28* | | | | |

**`w_mask`'in İÇ OPTİMUMU var: 0,25.** Sadece o kol `taban`ı ve `rakip`i geçiyor,
ve doluluğu neredeyse tam kalibre (val prob hedefi GT alfa ort = **0,0876**,
128 görüntüde ölçüldü; kol A'nın acc 0,50'si = **5,7×** = dolu küp çöküşü).

**⛔ Betiğin `*** CALISIYOR ***` etiketi ÜÇ KOLA DA basıldı ve FAZLA CÖMERT:**
kapısı `PSNR > 15` diyor ama `taban` 17,48 ⇒ sabit bir blob'dan kötü kolları da
geçiriyor. Betiğin *"herhangi bir kol çalışıyorsa üç aşamalı zincirin gerekçesi
çöküyor"* sonucu KABUL EDİLMEDİ. Dürüst cümle: **sıfırdan render kaybı KIRIK DEĞİL**
(kol A gibi çökmüyor), ama 1024 obje/6000 adımda 17,88 dB, V3 zincirinin 12.320
obje/30.800 adımda verdiği 19,38 dB ile kıyaslanamaz. **Gerekçe zayıfladı, çökmedi.**

**Kol A birebir tekrar EDİLEMEZ:** `normalize_cams` ile koşmuş ve `canonicalize`
o tarihte bozuktu. Ayrıca `--unfreeze_last 4` açıktı (WM'de 0) ve alfa hedefi
küçültme düzeltmesinden önceydi — kaybının %69'u tam o mask terimiydi.
⚠️ `train_lrm` künyeye **argv yazmıyor** (`runstamp.py:52` topluyor ama basılmıyor)
⇒ kol A'nın `density_bias`'ı bugün kurtarılamıyor.

**⚠️ WM_0p25'i "iptal et" diye ÖNERDİM, YANILDIM.** Gerekçe: "iki uç eşit çıktı,
arada eğim yok". Bu bir mantık hatası — iki ucun eşitliği İÇ OPTİMUM'la tamamen
tutarlıdır. Deneyin en değerli bulgusu tam o kolda çıktı.

#### Diğer doğrulanmış defektler (bağımsız denetim, hepsi kodla teyit edildi)

- `renderer.py:75` — `<=` yüzünden ışın==chunk'ta parçalama sessizce kapanıyor.
  TAVAN2'nin dört kolu da `--ray_chunk 32768` verildiği hâlde parçasız koştu;
  K256 bu yüzden 5× yavaştı (0,246 → **1,21 it/s**, chunk 8192 ile İLK KEZ açıldı).
- `bench_keskinlik.py` — aşındırma sonrası <100 piksel kalan obje **NaN** dönüyor,
  `np.nanmean` sessizce atıyor, kaç obje katıldığı basılmıyor (24 diyor, ~21).
  Atılanlar İNCE objeler, yani şikâyetin tam konusu. Ayrıca `asindir(px=7)` 15px'ten
  ince her yapıyı metrikten çıkarıyor; std/aralık/eşleşmiş test yok (K128↔K256 farkı
  0,010!); `FAYDALI = oran × kor+kaydır` enerji fazlalığını ÖDÜLLENDİRİYOR.
- `diag_yuzey.py:102` — her kol KENDİ `ns×2` ızgarasında ölçülüyor (K64: 192 örnek,
  K256: 640) ⇒ "hepsi 1,41–1,56 px" kıyası farklı örnekleme oranlarında.
- `guards.py` — `ACC_MIN` var, **`ACC_MAX` yok**; `acc` hiçbir yerde GT alfa
  ortalamasıyla (0,0876) kıyaslanmıyor ⇒ kol A'nın 5,7× dolu küpü `saglikli` bastı.
- `metrics.neighbor_indices` — komşu 64 objelik val prob'un İÇİNDEN seçiliyor;
  Tatarchenko baseline'ı eğitim setinden (12.320) gelmeli ⇒ RAKİP olması gerekenden
  çok zayıf (16,28 < taban 17,48).

#### Sonraki işler (öncelik sırasıyla)

1. **`distill_lrm`'e HELD-OUT probe** (~15 satır). Bugün "val" diye okunan her şey
   in-sample; bu yapılmadan Deney 2'nin hiçbir tekrarı okunamaz.
2. **`OLCEK_YONTEMI=kutu` varsayılan yapılsın mı?** Doğru olan o, ama mevcut
   öğretmen bankası (`teacher_1024_v3.pt` vb.) eski yolla oturtuldu ⇒ karışık
   kullanılırsa sessiz uyuşmazlık. Karar verilmeli, kendiliğinden değiştirilmemeli.
3. **tp128 + head genişlemesi:** `LRM.__init__` → `TriplaneHead(upsample=...)`,
   `distill_lrm --tp_res`, ve tam doygunluktan kaçınmak için `dim` ya da ara
   nonlineerlik. Öğretmen bankası 1024 objede 19,3 GB ⇒ CPU sayfalama (47,6 GB RAM,
   29,5 GB boş ⇒ sığar).
4. **`bench_keskinlik` düzeltmeleri** — katılan obje sayısını bas, eşleşmiş test +
   hata payı, `FAYDALI` sütununu kaldır, çok-ölçekli bant geçirgen `kor`
   (mevcut 3×3 kalıntı + 7px aşındırma, kullanıcının şikâyet ettiği bandı
   TAMAMEN dışlıyor: "sandalye kalınlaşmış" siluet/orta frekans sorunudur).
5. `w_mask 0,25` benimsensin; 0,1–0,4 arası ince tarama ucuz.


### 2026-09-03 — ZİNCİR3: keskinlik kaybının YERİ bulundu (öğretmenden SONRA)

**Bu bölüm 2026-09-02 bölümündeki "tp128 benimsenir" kararını GEÇERSİZ KILAR.**

#### Ölçüm: üç sistem, AYNI 24 obje, TEK `bench_keskinlik` çağrısı

Bugüne kadar öğretmen / distile öğrenci / 3. aşama hiçbir yerde aynı tabloda
değildi. Elde yalnız PSNR zinciri vardı (29,93 → 26,87 → 22,52 dB).

| sistem | enerji | GT oranı | kor | kor+kaydır | **kor_orta** |
|---|---|---|---|---|---|
| *GT 128→256 tavanı* | 0,00704 | %34 | 0,521 | 0,521 | *0,968* |
| *GT 64→256 tavanı* | 0,00317 | %16 | 0,190 | 0,235 | *0,767* |
| öğretmen `teacher_1024_v3` | 0,00950 | %47 | 0,087 | 0,137 | **0,292** |
| distile `distilled_v3` | 0,00481 | %24 | 0,027 | 0,111 | **0,199** |
| 3. aşama `last_V3` (ÜRÜN) | 0,00592 | %29 | 0,008 | 0,097 | **0,024** |

```
ESLESMIS FARK (ref: ogretmen; objeler ortak => eslesmis test bedava)
  distilled_v3   d(kor) -0.060+-0.040*   d(kor_orta) -0.093+-0.078*
  last_V3        d(kor) -0.079+-0.044*   d(kor_orta) -0.268+-0.092*
```

**⛔ tp128 GÜNDEMDEN DÜŞTÜ.** Zincir kaybı (`kor_orta` −0,268) tp64→tp128'in
vereceği kazançtan (+0,178) BÜYÜK. Keskinlik kaybının **tamamı öğretmenden
sonra** doğuyor; ızgarayı büyütmek yanlış yere yatırım.

**⛔ ÜRETİM ÖĞRETMENİ KENDİ IZGARASINI KULLANMIYOR.** Aynı tp64, tek fark
denetim: `TAVAN2_K64` (bölge128←U[256,512]) **0,418** ↔ `teacher_1024_v3`
(res 64 tam kare) **0,292**; 64px tavanı 0,767. Izgara bağlayıcı DEĞİL.
⚠️ Confound: K64 obje başına 500 güncelleme aldı, üretim 100 →
`ogretmen_v4_bolge.sh` bunu üretimle aynı 100 güncellemede izole ediyor.

**Görsel** (`dataset/lrm_bench/ZINCIR3_kare.png`): distile kayayı düz gri
yapıyor; `last_V3`'te kayanın üzerinde **gözle görülür dokuma/kafes deseni**,
iki ayrı deniz kabuğu **tek kütleye kaynaşmış**, arının şeritleri yok.
Kullanıcının "sandalye kalınlaşmış" tespiti bu: ayrı yapılar birleşiyor.

#### 🔑 KÖK NEDEN: `TriplaneHead` örtüşmesiz — hata blok sınırlarında birikiyor

`scripts/diag_kafes.py` (YENİ). Birincil metrik `blok_sinir` =
ort|fark| BLOK SINIRINDA / BLOK İÇİNDE (2×2 blok ⇒ satır çiftleri).

```
teacher_1024_v3 [serbest parametre, head YOK]  blok_sinir 1.03 +-0.01  KONTROL
TAVAN2_K64      [serbest parametre, head YOK]  blok_sinir 1.01 +-0.07  KONTROL
distilled_v3    [head'den geciyor]             blok_sinir 1.97 +-0.14
last_V3         [head'den geciyor]             blok_sinir 2.91 +-0.36
```
ve bu, `kor_orta` çöküşüyle birebir aynı sırada (0,292 → 0,199 → 0,024).

Öğretmenler serbest parametre (head'den geçmiyorlar) ⇒ kontrol kolu geçti,
ölçüm yöntemi sağlam. `TriplaneHead` = `ConvTranspose2d(k=2, s=2)`; `k == s`
⇒ **örtüşme sıfır**, her token bağımsız bir 2×2 blok üretir.

**CPU tezgâhı** (`scripts/bench_head_kafes.py`; transformer/encoder/renderer
YOK, sadece serbest token + head; hedef `TAVAN2_K64`; kontrol `blok_sinir`
1,046):

| dim | token/obje | rel(k2s2) | rel(k4s2) | blok_sinir(k2s2) | blok_sinir(k4s2) |
|---|---|---|---|---|---|
| 512 | 1.572.864 | 0,0000 | 0,0000 | — | — (DEJENERE) |
| 128 | 393.216 | 0,0001 | 0,0002 | — | — (DEJENERE) |
| 32 | 98.304 | 0,0669 | 0,0450 | 1,26 | 1,01 |
| 16 | 49.152 | 0,1475 | 0,1018 | 1,66 | 1,00 |
| 8 | 24.576 | 0,2423 | 0,1877 | 5,85 | 0,98 |
| **4** | 12.288 | 0,3506 | 0,3145 | **8,67** | **1,01** |

dim=4 gerçek çalışma noktası (`distilled_v3` held-out `rel` 0,3029,
`O0_taban` 0,4094). **Örtüşmenin bedeli NEGATİF:** her dim'de k4s2 daha iyi
uyuyor (%10–32). Takas yok, bedava kazanç.

⚠️ **Kafes BAŞLANGIÇTA YOK** (eğitimsiz k2s2 Nyq 1,08) — **öğreniliyor**.
Mekanizma "örtüşme yok ⇒ süreklilik yok" değil; 4 alt-çekirdek eğitim
sırasında anti-korele bir konfigürasyona sürükleniyor.
⚠️ **RÜTBE DEĞİŞMİYOR** (3×32²×512 sınırı örtüşmeden bağımsız); değişen TABAN.

#### ⛔ İKİ METODOLOJİK HATA (ikisi de bana ait, ikisi de yakalandı)

**1. `nyquist_orani` AYIRT EDİCİ DEĞİL — yazdığım kapı doğru hipotezi
reddederdi.** Gerçek noktada k2s2 Nyq 9,92 ↔ k4s2 **10,31** (örtüşmeli kol
DAHA YÜKSEK). Ön-kayıtlı kapım "k4s2 < 2 ise doğrulandı" diyordu ⇒ 10,31
görüp "kafes head'den DEĞİL" derdi. Doğru metrik `blok_sinir`.
Ayrıca tek FFT kutusunun ~11 kutuluk medyana oranı ağır kuyruklu
(std, ortalamanın 2 katı). `blok_sinir` kararlı (kontrol 1,046, ayrım 8×).

**2. İlk head testim DEJENERE'ydi.** Serbest token 1.572.864, hedef 393.216
⇒ 4× aşırı parametrizasyon, iki taban da TAM fit (`rel=0.0000`) ⇒ ölçülen
şey hedefin kendi oranıydı. Önerdiğim düzeltme (512→128) de işe yaramıyor:
128'de harita kare ve tersinir, `rel` hâlâ 0,0001. **dim ≤ 32 şart.**
Ayrıca `manual_seed(0)` → `Head(tip)` → `tok` sırasında head init'i çekirdek
boyutuna göre farklı RNG tüketiyor ⇒ token kollar arası farklı başlıyordu.
Token ve head'i AYRI tohumla.

#### Elenen alternatif kaynaklar (kafes nereden gelmiyor)

- `sample_triplane`/`grid_sample` bilineer, `volume_render` jitter:
  `diag_kafes` triplane TENSÖRÜNÜ ölçüyor, hiçbir renderer çağrısı yok. Elendi.
- `w_tv`: (a) `distill_lrm.py:185` kaybı **saf MSE, TV YOK** — yine de
  `distilled_v3` kafesli. (b) İki kontrol öğretmeni de `w_tv=0.05` ve temiz.
  (c) TV periyot-2'yi zaten CEZALANDIRIR ⇒ ancak eksik fren olabilir, kaynak
  olamaz. Elendi.
- **Elenemeyen eş-şüpheli:** `transformer.py:78` `tp_tokens` 3×32² serbest
  gömme — komşu token'ları bağlayan uzamsal önsel YOK, o da 64²'de periyot-2
  üretebilir. `blok_sinir` ikisini AYIRAMAZ. Pratikte önemsiz: örtüşmeli taban
  ikisini birden yumuşatıyor (CPU tezgâhı transformer'sız da düzeliyor).

#### Araç düzeltmeleri (2026-09-03)

- `bench_keskinlik`: **katılan obje sayısı** basılıyor (24 dendiğinde fiilen
  ~20; **5 obje NaN dönüp sessizce atılıyormuş**, maskenin ort %33'ü kalıyor),
  **eşleşmiş fark + %95 GA**, **`kor_orta`** (3×3−9×9 bandı = çıta/ayak
  ölçeği; eski metrik bu bandı HİÇ ölçmüyordu), `FAYDALI` sütunu KALDIRILDI.
  D7 kapatıldı: `--students` dalı künyedeki `bound`/`n_samples`'ı yok sayıyordu.
- `guards.py`: **`FULL_CUBE`** (`acc/GT_ALPHA > 3.0`, `GT_ALPHA = 0.0876`
  ölçüldü). Kol A'nın 5,71× dolu küpü 6000 adım "saglikli" basmıştı.
- `train_lrm --w_mask` varsayılanı **0,25** (ölçülmüş iç optimum).
- `imutil` varsayılanı **`kutu`** + `olcek_yontemi` künyeye bağlandı
  (`fit_teacher`/`distill_lrm`/`train_lrm`); `--init_from` uyuşmazlıkta durur.
- `distill_lrm --holdout` (varsayılan 32): **ilk gerçek genelleme sayısı**.
  `--oracle` ile otomatik kapanır (serbest token obje başına öğreniliyor).
  `--head {k2s2,k4s2}` ve `--seed` eklendi.
- `TriplaneHead(tip="k4s2")` opt-in; init ölçeği fan_in'e göre düzeltildi
  (iki taban da çıktı std ~0,24 — bu projede head init ölçeği ÜÇ kez çöküşe
  yol açtı).
- Testler 223.

#### ⚠️ Adam banka tuzağı (CPU sayfalamaya geçilirse ISIRIR)

`fit_teacher.py:120` bankayı tek `nn.Parameter` tutuyor; gradyanı 0 olan obje
için adım oranı her adımda β1/√β2 = 0,9005 kat azalıyor ⇒ geometrik toplam
**1/(1−0,9005) = 10,05**. Yani bir objenin son gradyanı sonraki ~50 adımda
toplam **10× uygulanıyor**, sonra sönüyor. Sınırlı, kendini sönümlüyor, her
objede aynı ⇒ doğruluk hatası ya da yanlılık YOK. **Ama efektif obje-başına
LR ≈ 10× nominal** ve `--lr 1e-2` bu rejimde ayarlandı. CPU sayfalamaya /
seyrek güncellemeye geçilirse bu 10× kendiliğinden kaybolur, öğretmen
"yakınsamıyor" görünür. Geçilirse `--lr 1e-1` ile başla ve ilk 500 adımın
PSNR eğrisini eskiyle üst üste koy.

#### ⚠️ Windows tuzağı: `pgrep` Start-Process süreçlerini GÖRMEZ

`gece_dishead_03eyl.sh` ilk sürümü `pgrep -f fit_teacher.py` ile bekliyordu;
v4 Windows tarafında `Start-Process` ile başlatıldığı için Git Bash onu
görmedi, "GPU boşaldı" deyip v4'ün yanına kuruldu (birkaç saniyede kesildi).
Kuyruk betiklerinde süreç değil **log işareti + VRAM** bekle.

#### v4 ilk denemesi: `CUBLAS_STATUS_INTERNAL_ERROR` = kılık değiştirmiş OOM

`--ray_chunk 0` ile VRAM 15,9/16,3 GB, `backward()`'da çöktü. cuBLAS workspace
ayıramayınca OOM değil INTERNAL_ERROR döner. `--ray_chunk 8192` +
`expandable_segments:True` ile: VRAM **11,9 GB**, hız **2,70 → 3,04 it/s**
(K256'daki desenin aynısı: parçalama hem güvenli hem HIZLI).


#### Gece koşuları (2026-09-03, 00:39–07:24): v4 + DIS-HEAD

**A) Öğretmen v4 — bölge kırpma denetimi, 1024 obje, üretimle AYNI 100 güncelleme**

| sistem | denetim | gün/obje | kor | kor+kaydır | **kor_orta** |
|---|---|---|---|---|---|
| *GT 64→256 tavanı* | | | 0,190 | 0,235 | *0,767* |
| `teacher_1024_v3` (üretim) | 64px tam kare | 100 | 0,087 | 0,137 | 0,292 |
| **`teacher_1024_v4`** | bölge128←U[256,512] | 100 | 0,151 | 0,229 | **0,361** |
| `TAVAN2_K64` | bölge128←U[256,512] | 500 | 0,158 | 0,236 | 0,418 |

```
ESLESMIS FARK (ref v3)
  v4          d(kor) +0.064+-0.079   d(kor+kaydir) +0.093+-0.055*  d(kor_orta) +0.070+-0.087
  TAVAN2_K64  d(kor) +0.071+-0.070*  d(kor+kaydir) +0.099+-0.036*  d(kor_orta) +0.127+-0.076*
```

Kapı `≥0,38` denetim / `≤0,32` güncelleme sayısı idi; **0,361 = KISMİ bant.**
0,292→0,418 açığının kabaca yarısı denetimden, yarısı 5× güncellemeden.
⚠️ v4'ün `kor_orta` farkı n=20'de **anlamlı değil** (GA sıfırı içeriyor); anlamlı
olan tek sütun `kor+kaydır`. K64 kolu `kor_orta`'da anlamlı — aynı yön, daha
büyük etki.
⚠️ **Görsel sayılardan fazlasını gösteriyor** (`OGRETMEN_V4_kare.png`): v3'te
arının şeritleri dağılmış, deniz kabuklarında **gökkuşağı benekleri** var;
v4'te şeritler temiz, göz belirgin, renk doğru. HF korelasyonu bu renk
artefaktını iyi yakalamıyor.
⚠️ v4 bile kendi tavanının **%47**'sinde (0,361 ↔ 0,767) — denetim öğretmeni
iyileştirdi, tavana çıkarmadı. Kalan açık AÇIKLANMADI.

`teacher_1024_v4.pt`: adım 51200, done=True, ölçek=kutu, final PSNR 32,03 dB.

**B) DIS-HEAD — örtüşmeli head tabanı gerçek zincirde (256 obje, 3000 adım, 3 kol)**

**① ARTEFAKT TARAFI: KESİN. Örtüşmeli head kafesi yok ediyor.**

```
teacher_1024_v3 [head YOK]      blok_sinir 1.03 +-0.01   KONTROL
DH_H_k2s2_a  [kontrol]          blok_sinir 3.96 +-0.30   KAFES VAR
DH_H_k2s2_b  [kontrol tohum 2]  blok_sinir 3.99 +-0.30   KAFES VAR
DH_H_k4s2    [ortusmeli]        blok_sinir 0.77 +-0.03   kafes YOK
```
İki kontrol tohumu birbirine 0,03 içinde (3,96 / 3,99), deney 5× aşağıda.
Ön-kayıtlı SAĞLAMA kapısı (`blok_sinir(kontrol) ≥ 3,0`) **GEÇTİ** ⇒ CPU
tezgâhı gerçek zincire transfer oldu. Görselde de net: `DISHEAD_kare.png`'de
kayada k2s2 kollarının dokuma deseni var, k4s2'de yok.

**② KALİTE TARAFI: OKUNAMADI (deney güçsüz).**

```
kor_orta:  k2s2_a 0.077 | k4s2 0.027 | k2s2_b 0.050
ESLESMIS FARK (ref k2s2_a): k4s2 -0.051+-0.080 | k2s2_b -0.027+-0.062
heldout_rel: 0.9349 | 0.9516 | 0.9556
  kontrol tohumlari arasi |fark| = 0.0207  (kapinin sarti < 0.015 IDI)
```
**Kontrolün kendi tohum-tohum yayılımı, aranan etkiden büyük.** Ön-kayıtlı
"okunabilirlik" şartı ihlal edildi ⇒ kalite üzerine **hiçbir sonuç
çıkarılamaz**. Kural harfiyen uygulanırsa `d_kor_orta < +0.01` dalı
"kafes KOZMETİK" der, ama deneyin gücü olmadığı için doğru etiket
**SONUÇSUZ**'dur, "kozmetik" değil.

Sebebi açık: 256 obje / 3000 adımda üç kol da berbat (`kor_orta` 0,03–0,08;
öğretmen 0,292, `distilled_v3` 0,199). Enerjide bile kontrol tohumları 2×
farklı (%17 ↔ %35). Bu rejimde head tabanının kalite etkisi görünmez.

**③ 🔑 Yan bulgu — projedeki İLK gerçek genelleme sayısı ve kötü:**
`insample_rel` 0,377 ↔ `heldout_rel` **0,935**. Görülmemiş obje için öğrenci
hedef varyansının %6,5'ini açıklıyor (1,0 = hiçbir şey öğrenmemiş).
⇒ **Oracle deneyinin (Deney 2) tüm merdiveni in-sample'dı** — O0 0,4094 /
O1 0,1339 / O2 0,0552 aynı 256-obje/3000-adım kurulumunda, held-out'u ~0,93.
O merdiven koşullandırma kalitesini değil **ezber kapasitesini** ölçmüş.
Kayıtlı "plato `val_rel` = 0,3029" da aynı şekilde in-sample.
⚠️ 256 obje bu görev için az; 1024/12.320'de daha iyi olabilir — ama artık
ölçüldü, varsayılmıyor.

**Sonuç:** head değişikliği artefaktı kesin olarak çözüyor, ama keskinliği
iyileştirdiğini gösteren kanıt YOK. Kalite sorusu, öğrencinin gerçekten
öğrenebildiği bir rejimde (≥1024 obje, ≥24k adım) tekrar sorulmalı.

**Araç düzeltmeleri (gece, koşudan ÖNCE yakalandı — ikisi de kendi kodumda):**
- `compat.mimari_tespit` head çekirdek boyutunu okumuyordu ⇒ k4s2 checkpoint
  ölçüm adımında `load_state_dict` şekil hatası verirdi (2,6 sa eğitimden SONRA).
- `bench_keskinlik` tp_res'i **çekirdek boyutundan** türetiyordu; k4s2'de 4 ⇒
  tp128 sanılıp `n_samples` 128, k2s2 kolu 96 ⇒ **A/B tek değişkenli olmazdı**.
  Belirleyici olan **stride**. Düzeltildi + 2 regresyon testi (toplam 225).

#### 3. aşama teşhisi (2026-09-03, 11:11–14:35): İKİ HİPOTEZ DE ÇÜRÜTÜLDÜ

**Soru:** `kor_orta` öğretmenden ürüne 0,292 → 0,024 düşüyor. Aşamalara bölünce:
distilasyon %68'ini **taşıyor** (0,292 → 0,199), 3. aşama %12'sini (0,199 → 0,024).
⇒ Felaket 3. aşamada. Neden?

**A) HİPOTEZ 1 — "3. aşamanın denetimi öğretmeninkinden kaba": ÇÜRÜTÜLDÜ.**

`train_lrm` varsayılanı `region 64 ← U[64,192]` (detay tavanı 192px); öğretmen v4
`region 128 ← U[256,512]` (512px). `zincir_v3_asama23.sh` bunları hiç geçersiz
kılmamış. Betik: `scripts/s3_denetim_ab.sh` (kapılar başlığında, koşudan önce).

İki kol, `distilled_v3`'ten başlayarak 3000 adım, `--dry_run` ile tek değişken teyitli:

| kol | denetim | ışın bütçesi | yamada obje | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|---|
| *başlangıç* `distilled_v3` | | | | | | **0,199** |
| **A** kontrol | `64 ← U[64,192]` | 64² | %29 | −0,002 | 0,053 | **0,044** |
| **B** deney | `64 ← U[192,512]` | 64² *(aynı)* | %55 | 0,032 | −0,001 | **−0,006** |
| *`last_V3`* (A tarifesi, 30800 adım) | | | | | | *0,024* |

```
SAGLAMA kapisi  kor_orta(A) < 0.17  -> GECTI (0.044); deney kisa DEGIL, okunabilir
GO kapisi       B-A >= +0.05 VE B >= 0.12  -> KALDI (B-A = -0.050, B = -0.006)
```

⇒ **Yüksek detaylı denetim dokuyu korumuyor; B, A'dan biraz daha kötü.**
Görsel (`S3DENETIM_kare.png`): B kayada **gözle görülür düzenli nokta ızgarası**
üretiyor, bankın şeklini tamamen kaybediyor. Yani yardım etmemekle kalmıyor,
yeni artefakt katıyor.

⚠️ B tek bir *fiziksel* değişiklikti ama iki etkisi vardı: detay tavanı **ve**
yamadaki obje oranı (%29 → %55). Sabit ışın bütçesinde bunlar ayrılamaz.

**🔑 ASIL BULGU: çöküş İLK 1000 ADIMDA oluyor.** Kademeli aşınma değil —
0,199'dan sıfıra bir anda düşüp orada kalıyor (30800 adımdaki `last_V3` de 0,024).

**B) HİPOTEZ 2 — "doku NeRF decoder'ında ölüyor": ÇÜRÜTÜLDÜ.**

`distill_lrm.py:148` öğrencinin NeRF'ini öğretmenden **kopyalıyor**, distilasyonda
hiç eğitmiyor. Ağırlıklarla doğrulandı: `distilled_v3.nerf` ↔ `teacher_1024_v3.nerf`
**12/12 tensör bit-birebir aynı.** ⇒ `distilled_v3`'ün 0,199'u = öğrenci triplane +
**öğretmenin** decoder'ı. 3. aşama decoder'ı da eğitiyor (bağıl kayma A %8,4, B %9,4).

Eğitimsiz test: aynı triplane, decoder öğretmeninki geri takılı
(`dataset/lrm_logs/nerfswap.log`, `NERFSWAP_kare.png`):

| sistem | enerji | kor | **kor_orta** |
|---|---|---|---|
| `distilled_v3` | %24 | 0,027 | **0,199** |
| `last_S3_A` | %21 | 0,008 | 0,044 |
| `SWAP_A` (A triplane + öğretmen NeRF) | %29 | 0,014 | **0,042** |
| `last_S3_B` | %18 | −0,010 | −0,006 |
| `SWAP_B` | %22 | −0,012 | **0,002** |

Eşleşmiş fark (ref `distilled_v3`): `last_S3_A` −0,154±0,063* ↔ `SWAP_A` −0,157±0,070*.
**Birbirinden ayırt edilemiyor.** Görselde de `SWAP_A` ≈ `last_S3_A`, `SWAP_B`'de
kayadaki ızgara aynen duruyor.

⇒ **Doku triplane'de ölüyor, decoder'da değil.** Öğretmenin decoder'ı aynı
triplane'den daha çok HF **enerjisi** çıkarıyor (%21→%29) ama korelasyon
kıpırdamıyor ⇒ triplane artık doğru yapıyı taşımıyor.

**C) Ne elendi, ne kaldı**

Bugüne kadar elenen keskinlik hipotezleri: `tp_res` (öğretmen ızgara-bağlı değil),
`grid_sample`/jitter, `w_tv`, yüzey kalınlığı/derinlik, denetim detay tavanı (bugün),
NeRF decoder (bugün). Head tabanı (`k4s2`) artefaktı kaldırıyor ama kalite etkisi
ölçülemedi (deney güçsüzdü).

Geriye kalan tek tutarlı açıklama, bağımsız denetçinin işaret ettiği eksen:
**`heldout_rel` 0,935** (görülmemiş objede hedef varyansının %6,5'i),
`ogrenci_std` hedefin %69'u, ve distilasyon **saf MSE** ⇒ belirsizlik altında
koşullu ortalama basılır, ki tanımı gereği bulanıktır. 3. aşama, 1024 objelik
distilasyon çözümünü 12.320 objelik genel çözümle değiştiriyor; genel çözüm
bulanık, çünkü model genellemiyor. **Keskinlik bağımsız bir hedef değil,
genellemenin sonucu.**

**D) ⛔ Kendi hatam (kayda geçsin)**

"bench 24 objesinin sadece 2'si eğitim setinde" dedim — **yanlış dosyaya baktım.**
`dataset/bench_uids_1024.json` öğretmen bankasının listesi DEĞİL (ortak yalnız 67).
Öğretmen `ds.uids[:1024]` üzerine fit edilmiş; `bench_keskinlik.py:259` bunu sert
`assert` ile zaten garanti ediyor. Doğrusu: **24/24 obje eğitim setinde.**

**E) Araç notları**

- `bench_keskinlik --out X` görseli **`X_kare.png`** yazıyor; `--help` metni
  `_tam.png` ve `_zoom.png` diyor, **BAYAT**. (Duman testiyle doğrulandı.)
- `train_lrm --dry_run` dump'ında `render_low`/`render_high` **YOKTU** ⇒ A/B'nin
  tek değişkeni koşmadan doğrulanamıyordu. Eklendi (`train_lrm.py:290`).
- `scripts/s3_olcum.sh` (YENİ): ölçümü koşudan ayırır, **yalnız diskte olan**
  checkpoint'leri ölçer. Sebep: sabit listeli ölçüm, bir kol çökerse sağlam kolun
  görselini de öldürüyordu. Koşan bash betiği düzenlenemediği için ayrı betik şart.
- Koşu öncesi ölçüm komutlarının üçü de (sayısal / görsel / `diag_kafes`) küçük
  ölçekte **fiilen koşturulup exit 0 alındı** — "3 saat sonra ölçüm çöktü"yü
  önlemenin tek yolu bu.

#### Encoder duvarı A/B (2026-09-03, 15:26–20:45): NO-GO — çözünürlük çerçevesi TÜKENDİ

Çözünürlük bütçesi tablosundaki üç duvardan hiç gevşetilmemiş olanı: girdi 224
(DINOv2 patch 14 ⇒ objeye ~8,6 patch). 448'de 17,2 patch.

**Maliyet ölçüldü, beklenti yanlıştı:** `GIRDI_RES` 224 ↔ 448 **ikisi de 1,12 it/s**.
Transformer görüntü + triplane token'ını BİRLEŞTİRİP self-attention yapıyor
(`transformer.py:108`): dizi 3328 → 4096 ⇒ attention 1,51×; encoder donuk ViT-S.
"4× pahalı" beklentisi ölçümle çürüdü — 448 pratikte bedava.

Üç kol, `distill_lrm`, 1024 obje (896 eğitim + **128 held-out**), 6000 adım,
54 maruziyet/obje. Betik `scripts/encoder_duvari_ab.sh`.

| kol | girdi | tohum | `heldout_rel` | `insample_rel` | `kor_orta` |
|---|---|---|---|---|---|
| E224a | 224 | 0 | **0,8067** | 0,4809 | 0,030 |
| **E448** | **448** | 0 | **0,8056** | **0,4646** | 0,050 |
| E224b | 224 | 1 | **0,8117** | 0,4867 | −0,005 |

```
SAGLAMA  heldout_rel(E224a) = 0.8067 < 0.97          GECTI
GURULTU  b = |E224a - E224b| = 0.0050                OLCULDU
ETKI     d = E448 - E224a   = -0.0011                |d| = 0.22 x b
KARAR    NO-GO -- girdi cozunurlugu BAGLAYICI DEGIL
```

**Bu, projedeki ilk düzgün güçlendirilmiş A/B.** Kontrolün tohum yayılımı
b = 0,0050 — DIS-HEAD'de 0,021 idi ve etkiden büyüktü. Burada etki gürültünün
**beşte biri** ⇒ null gerçek bir null, okunamayan bir deney değil.
`kor_orta` de aynı yönde: 448'in etkisi (+0,020) kontrolün kendi yayılımından
(0,035) küçük ⇒ o sütun da bir şey söylemiyor.

**🔑 Ama 448 IN-SAMPLE'da GERÇEKTEN yardım ediyor:** `insample_rel` E448 0,4646 ↔
E224a 0,4809 / E224b 0,4867; kontrol yayılımı 0,0058, etki −0,0163 = **2,8× gürültü**.
Yani daha yüksek girdi çözünürlüğü modele *ezberleyecek daha çok şey* veriyor,
genellemeye **sıfır** katkı yapıyor. Bu, null'u açıklıyor ve mekanizmayı
DESTEKLİYOR: keskinlik bu kurulumda ezberden geliyor.

**Maruziyet–keskinlik ilişkisi üçüncü kez doğrulandı** (hepsi aynı 24 objede,
hepsinde o objeler eğitim setinde):

| sistem | maruziyet/obje | `kor_orta` |
|---|---|---|
| `distilled_v3` | 187 | 0,199 |
| `ENC_E224a/b` | 54 | 0,030 / −0,005 |
| `last_V3` (3. aşama) | 40 | 0,024 |

**Görsel** (`ENCDUVAR_kare.png`): `teacher_1024_v4` gerçekten iyi — kayanın taneli
dokusu, arının şeritleri ve gözü, bankın çıtaları GT'ye yakın. `distilled_v3`
bozulmuş ama tanınabilir. Üç ENC kolu da yumru; **E448 iki 224 kolundan gözle
ayırt edilemiyor.**

#### ⛔ ÇÖZÜNÜRLÜK BÜTÇESİ ÇERÇEVESİ KAPANDI

`docs/KESKINLIK-TESHISI.md`'nin üç duvarı da artık ölçüldü ve **hiçbiri bağlayıcı
değil**:

| duvar | test | sonuç |
|---|---|---|
| triplane 64² | TAVAN2 merdiveni (K64/K128/K256) | öğretmen ızgara-bağlı DEĞİL |
| denetim çözünürlüğü | v4 (öğretmen) + S3 A/B (3. aşama) | öğretmende yardım etti, 3. aşamada ETMEDİ |
| **encoder 224** | **bu koşu** | **etkisi YOK** |

Bunlara ek olarak elenenler: NeRF decoder (takas testi), head tabanı
(artefaktı kaldırıyor, kaliteye etkisi kanıtsız), `w_tv`, `grid_sample`/jitter,
yüzey kalınlığı/derinlik.

**Geriye kalan tek tutarlı açıklama:** model dokuyu *görerek* değil *tekrarla*
öğreniyor. En iyi genelleme sayımız `heldout_rel` **0,8067** — görülmemiş objede
hedef triplane varyansının ancak **%19'u**.

#### 🔑 Bu null'un doğrudan işaret ettiği yer: DONUK ENCODER

`encoder.py:15` tüm DINOv2 parametrelerini `requires_grad_(False)` yapıyor ve
`:27` `forward`'ı `@torch.no_grad()` ile sarıyor. `train_lrm.py:527` gerektiğinde
bu sarmalayıcıyı **açıyor** (`--unfreeze_last` / `--train_encoder`), ama
**`distill_lrm.py` HİÇ açmıyor** — yani görüntü→triplane haritasının fiilen
öğrenildiği aşamada encoder her zaman tamamen donuk.

Bu, 448'in neden hiçbir şey yapmadığını açıklıyor: 4× daha çok patch veriliyor
ama **özellikler donuk ve semantik**; dokuyu taşımıyorlarsa daha çok patch
taşımayan bilgiyi çoğaltmaktan ibaret. Referans uygulamaların **üçü de**
(LRM, OpenLRM, TripoSR) encoder'ı eğitiyor; bizde distilasyonda hiç denenmedi.

#### Araç eklemeleri (koşudan ÖNCE, hepsi fiilen doğrulandı)

- `defaults.INPUT_RES` artık `GIRDI_RES` ortam değişkeninden okunuyor.
- **`compat.load_lrm` künyeden `input_res`'i geri kuruyor** (`GIRDI_RES_ALANI`).
  `KUNYE_ALANLARI`'na konulamaz: `LRM.__init__` kwarg'ı değil, SINIF NİTELİĞİ.
  Eksikken 448'de eğitilmiş ckpt 224 Plucker haritasıyla render edilirdi ve
  **hiçbir şekil hatası vermezdi** (`model.py:52`: `side` gerçek girdiden,
  `INPUT_RES` sürecin varsayılanından gelir).
- `bench_keskinlik` girdi çözünürlüğüne göre **ayrı dataset** kuruyor; supervision
  kameraları ortak kalıyor ⇒ karışık 224/448 öğrenci listesi tek çağrıda,
  eşleşmiş olarak ölçülebiliyor. Sahte bir 448 ckpt ile fiilen koşturuldu.
- Testler **227** (2 yeni: künyeden geri kurma + künyesiz ckpt varsayılanı).

### 📋 Sıradaki işler → **`docs/SIRADAKI-ISLER.md`** (2026-08-29)

Deney kuyruğu (ön-kayıtlı eşiklerle), veri kararı + yeniden açma koşulu,
Faz C işleri, ölçüm altyapısındaki kalan boşluklar. **Deney seçimi buradan
yapılır** ki her oturumda sıfırdan tartışılmasın.

## 🛠️ Çalışma Kuralları

- Yaratıcı/kurulum işine başlamadan **superpowers skill'lerini** kullan
  (brainstorming → writing-plans → TDD → systematic-debugging).
- Spec'ler `docs/superpowers/specs/YYYY-MM-DD-<konu>-design.md` altına yazılır ve commit'lenir.
- Bu proje kişisel öğrenme amaçlı: kararları **açıklayarak** ilerle.
