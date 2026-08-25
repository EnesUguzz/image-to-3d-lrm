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

- İnen objeler: `C:\Users\ENES OĞUZ\.objaverse\hf-objaverse-v1\glbs\`
  → **51.534 adet `.glb`**, ~490 GB, `<uid>.glb` formatında 160 alt klasörde.
- URL haritası: `C:\Users\ENES OĞUZ\.objaverse\hf-objaverse-v1\object-paths.json.gz`
- Ayrıca: `C:\Users\ENES OĞUZ\.objaverse\smithsonian\objects\` (2392 glb, alternatif set).

## 🔗 Referans Repolar

- `allenai/objaverse-xl` — veri seti + indirme paketi.
- `allenai/objaverse-rendering` — orijinal render scriptleri. Local kopya:
  `C:\Users\ENES OĞUZ\objaverse-rendering\scripts\` (blender_script.py, distributed.py, ...).
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
- **Kodlama tuzağı:** proje yolu **Ğ** (`ENES OĞUZ`) içeriyor → tüm json okuma/yazmalarda
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
→ `M_enc4`, `M_base` ile bit-bazında aynı (çıktı PNG md5'i bile aynı) — encoder
açılmış görünüyor (39.6M→46.7M) ve mekanizma izole testte çalışıyor (gradyan akıyor),
o koşuda neden etkisiz kaldığı **bulunamadı**. **Kısmi encoder çözme CEVAPSIZ.**

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

### Açık işler (öncelik sırasıyla)

1. Kısmi encoder çözmeyi **ağırlık-değişti kontrolüyle** yeniden koş (`M_enc4` geçersizdi)
2. `train_lrm.py`'ye **crop desteği** ekle (matrisin kazananıydı, aktarılamadı —
   `overnight.sh` seçicisi aktarılamayan bayrak kazanınca sessizce "(yok)" yazıyor, düzelt)
3. Efektif batch'i büyüt (şu an micro 2 × accum 4 = 8)
4. Encoder için **ayrı düşük LR** (şu an encoder de 4e-4 alıyor)
5. Üst/alt fotoyu **girdi** vermek conditioning'i düzeltir mi (sphere20 render'ları `dataset/_viewcov/` hazır)
6. `normalize_cams` A/B (şu an kapalı)
7. 50k'ya çıkma — 13k'da genelleme görülmeden anlamsız

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

## 🛠️ Çalışma Kuralları

- Yaratıcı/kurulum işine başlamadan **superpowers skill'lerini** kullan
  (brainstorming → writing-plans → TDD → systematic-debugging).
- Spec'ler `docs/superpowers/specs/YYYY-MM-DD-<konu>-design.md` altına yazılır ve commit'lenir.
- Bu proje kişisel öğrenme amaçlı: kararları **açıklayarak** ilerle.
