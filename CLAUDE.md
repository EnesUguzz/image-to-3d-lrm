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

## 🛠️ Çalışma Kuralları

- Yaratıcı/kurulum işine başlamadan **superpowers skill'lerini** kullan
  (brainstorming → writing-plans → TDD → systematic-debugging).
- Spec'ler `docs/superpowers/specs/YYYY-MM-DD-<konu>-design.md` altına yazılır ve commit'lenir.
- Bu proje kişisel öğrenme amaçlı: kararları **açıklayarak** ilerle.
