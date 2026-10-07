# Proje Devir Belgesi — 3D Object Project

**Son güncelleme:** 2026-08-25
**Amaç:** Yeni bir oturuma (insan ya da model) projenin *tüm* geçmişini, ölçülmüş
bulgularını ve açık sorularını tek dosyada aktarmak.

> **Bu belge nasıl okunur:** Her iddianın yanında onu üreten koşunun etiketi var
> (`AB_YENI`, `O1_oracle_mlp2` …). Ham sonuç `dataset/lrm_bench/<etiket>.json`
> içinde, tam config'iyle birlikte. **Kanıtı olmayan hiçbir iddiaya güvenme** —
> bu projede birden fazla "besbelli" varsayım ölçümle çürüdü.

---

## 1. Proje

Tek bir **transformer-tabanlı LRM (Large Reconstruction Model)** eğitmek:
kullanıcı 1 foto (ya da 4 kanonik açı) verir → model triplane NeRF üretir →
mesh (`.glb`) çıkarılır → web'de three.js ile gösterilir.

- Difüzyon (Zero-1-to-3) **değil**, transformer/LRM yaklaşımı seçildi.
- Tek model hem 1 hem 4 görüntüyü kabul eder (eğitimde rastgele 1–4 görünüm).
- **Kişisel öğrenme projesi.** Amaç SOTA değil, uçtan uca çalışan + anlaşılan pipeline.

### Fazlar
| faz | konu | durum |
|---|---|---|
| **A** | Render pipeline — Objaverse `.glb` → çok-görünümlü foto + kamera pozu | ✅ bitti |
| **B** | LRM eğitimi | ▶ **şu an burada** |
| **C** | Web app (backend inference + mesh, frontend three.js) | ⬜ başlanmadı |

### Mimari (Faz B)
```
1–4 foto (224²) → donuk DINOv2 ViT-S/14 → patch token (384-d)
   → transformer (dim 512, 12 kat, joint self-attention)
   → triplane 3×32×64×64
   → NeRF MLP (2 kat, hidden 64) → volume render (48 örnek)
   → fg-ağırlıklı MSE + LPIPS + mask loss
```

---

## 2. Donanım / ortam

- **GPU:** RTX 5080, 16 GB, Blackwell **sm_120** → CUDA 12.8+ şart, eski wheel'ler çalışmaz
- `torch 2.11.0+cu128`, Python 3.10.8, Windows 11, PowerShell (birincil) + Git Bash
- Blender 4.4 (`C:\Program Files\Blender Foundation\...`)
- DINOv2 (torch.hub) ve LPIPS-VGG cache'li

---

## 3. Veri

### Kaynak
- `%USERPROFILE%\.objaverse\hf-objaverse-v1\glbs\` — **51.534 `.glb`**, ~490 GB
- Objaverse++ (arXiv 2504.07334) **Objaverse 1.0 uid'lerini** etiketliyor → diskteki
  glb'lerle eşleşir, yeni indirme gerekmez. Kalite skoru + ikili etiketler
  (Transparency, Scene, Single Color, Not a Single Object, Figure).

### Üretilen render setleri
| dizin | obje | görünüm | not |
|---|---|---|---|
| `dataset/renders/` | 3.086 | 16 | 7,8 GB — **ESKİ** — `sphere` çerçeveleme + CYCLES. Ölçüldü: conditioning'i çökertiyor. Kullanma. |
| `dataset/renders_opp_score3/` | 14.597 | 16 | **GÜNCEL** — `fit` çerçeveleme + EEVEE_NEXT, ~37 GB |
| `dataset/_viewcov/` | 49 | 24 | `sphere20` şeması (üst/alt dahil), 177 MB, sadece kapsama testi için |

### Kullanılabilir listeler
- `dataset/train_list_opp_score3.json` → **13.009 train / 1.445 val** (filtre sonrası %99,04 geçti)
- `dataset/train_list.json` → eski set (2.635 + 292)

### Meta şeması (`dataset/renders_*/<uid>/meta.json`)
```json
{"uid", "resolution": 512, "num_views": 16, "camera": {...},
 "canonical_indices": [0,1,2,3],
 "views": [{"index", "role": "canonical|supervision", "azimuth_deg",
            "elevation_deg", "file", "extrinsic", "intrinsic"}]}
```
- `intrinsic` **her view içinde** saklanır (master 512 çözünürlüğe göre)
- `c2w = inv(extrinsic)` (extrinsic = world→camera)

---

## 4. Faz A — render pipeline

### Kesinleşmiş kararlar
- 512×512 RGBA PNG (alpha dahil), diskte master
- Obje başına **16 görünüm**: 4 kanonik (az 0/90/180/270, el ~20°) + 12 supervision
- Eğitimde 512 → 256/128 downscale
- Kamera: `radius 1.4866`, `lens 35 mm`, `sensor 32 mm`

### `NORM_MODE` — çerçeveleme (varsayılan `"fit"`)
| mod | tanım | kaplama | taşma |
|---|---|---|---|
| `sphere` (eski) | bbox yarı-köşegeni = 0.5 | 0.0843 | %0 |
| `bbox` | en uzun kenar = 1.0 | 0.1695 | **%5 → KULLANMA** |
| **`fit`** | bbox normalize, sonra 16 kameradaki gerçek izdüşüm kareye oturtulur | 0.0899 | **%0.00** |

> ⚠️ 2026-08-21'de "bbox çerçeveleme 1.87× kazanç" diye kaydedilen not **yanlıştı**.
> 6 objeyle ölçülmüştü. 100 objede tekrarlanınca bbox modunda render'ların **%5'i**
> kenardan taşıyor. Kazancın tamamı objeleri kırpmaktan geliyormuş. Ayrıca bbox
> modu doğruluk hatası: NeRF `bound=0.6` dışını maskeliyor ama bbox
> normalizasyonunda obje yarıçapı 0.866'ya çıkabiliyor.

### Render motoru: CYCLES → **EEVEE_NEXT**
24 obje × 16 görünüm = 384 render çifti karşılaştırıldı:
alpha kaplama farkı **0.00001**, obje RGB farkı ort **%2.4**, PSNR ort **37.4 dB**
(en kötü 27.3 dB, metalik/yansımalı obje). EEVEE'nin zaafı cam/kırılma —
Objaverse++ `is_transparent` filtresi bunları zaten eliyor.
`RENDER_ENGINE=CYCLES` ile geri alınır.

| config | sn/obje | 14.594 obje |
|---|---|---|
| Cycles 48 örnek + 4 worker | 3.68 | 14,9 saat |
| **EEVEE 32 örnek + 8 worker** | **0.66** | **~2,7 saat** |

Worker taraması: Cycles'ta 4, EEVEE'de **8** optimal (12 daha kötü).
Blender süreç başlatma 0.30 sn — optimize edilecek yer değil.
Cycles'ta örnek 48→24 hiç fark etmiyor (süre BVH kurulumunda).

### Aydınlatma — **bilinçli olarak referanstan farklı**
Bizde: world ambient 1.5 + iki area ışık (`KeyTop` z=+3, `FillBottom` z=−3, energy 400, size 8).
Referans `blender_script.py`: tek geniş yumuşak tepe ışığı (energy 30000), alt dolgu yok.

**Karar: değiştirilmiyor.** Gerekçe: `TriplaneNeRF.forward(feats)` **bakış yönü almıyor**
(`scripts/lrm/nerf.py`) → model bakış-bağımsız görünüm dışında bir şeyi temsil edemez;
düz aydınlatma kapasiteyle eşleşiyor (oracle tavanı 25,04 dB bunu doğruluyor). Alt dolgu,
alt görünümler için gerekli. Karanlık veri = bilinen "renk siyaha çökme" tuzağı.
**Bedeli:** zayıf gölge-ile-şekil ipucu, tek-girdi yolunu etkiler. Tek-girdi kalitesi
darboğaz çıkarsa: 32 objeyi iki ışıkla render et + `bench_overfit` karşılaştır (~35 dk).

### Faz A'da çözülen somut hatalar
- **Göreli çıktı yolu:** Blender `render.filepath`'i açık .blend'e göre çözer; blend
  yokken sürücü köküne düşer → `abspath` zorunlu
- **Windows kodlama:** Blender stdout'u cp1254 ile çözülünce `UnicodeDecodeError`
  → `encoding="utf-8", errors="replace"`
- **Dev zemin (FLOOR) plane'leri:** bazı glb'lerde objeden ~5× büyük düz mesh bbox'ı
  şişiriyor → `remove_floor_planes()` normalize öncesi siler
- **Normalize iki geçişli olmalı:** tek geçişte world-center ile local-location karışıyor
- **Aydınlatma:** tek tepe ışığı bazı açıları karartıyordu → ambient + üst/alt area
- **`min_cov` eşiği 0.03 → 0.010:** 0.03 eşiği eski `sphere` çerçeveleme için ayarlıydı.
  `fit` modunda ince/uzun objeler (kılıç, yay, merdiven, ayakta insan) diğer açılardan
  çizgiye iniyor; 0.03 bunların **661 tanesini haksız eliyordu.** Gözle doğrulandı.

---

## 5. Faz B — kök neden: "ortalama obje" yerel minimumu

**Belirti:** Model her girdiye aynı bej blob'u basıyordu (dataset ortalama rengi
`[0.469, 0.418, 0.387]`).

**Ölçümle ELENENLER** (yani sorun bunlar DEĞİLDİ): renderer, kameralar, NeRF MLP
derinliği, triplane çözünürlüğü, donuk DINOv2 encoder, transformer mimarisi
(cross-attention denendi, fayda yok), veri kalitesi, veri miktarı.

**Kanıt zinciri:**
| koşu | ne yapıyor | sonuç |
|---|---|---|
| `O1_oracle_mlp2` | her objeye kendi serbest triplane'i (oracle conditioning) | **25.04 dB** |
| `T1_distill_joint` | aynı ağ, triplane'e doğrudan regresyon | **23.94 dB** |
| `A_baseline_joint` | aynı ağ, render kaybıyla sıfırdan | **17.06 dB** |
| `E1_distilinit_renderloss` | distile ağırlıklardan render kaybıyla devam | **24.99 dB / top-1 %100** |

İyi çözümün render kaybı 0.23, blob'un 0.47 → **hedef doğru, optimizasyon sıkışıyor.**

Tam teşhis: **`docs/lrm-blob-diagnosis.md`**

### Neden sıkışıyor (denetim sinyali analizi)
- Kaybın **%85'i arka plan** hakkında
- "Boş üret" bedava doğru çıkıyor (obje kare içinde küçük)
- Çözüm parçaları: rastgele arka plan kompoziti (sabit-bg renk çökmesini kapatır),
  fg-ağırlıklı kayıp (`fg_weight 5.0`), beyaz bg

---

## 6. Deney defteri (tam liste)

Hepsi `dataset/lrm_bench/<etiket>.json`. `oran` = `mse / ortalama-obje-baseline`;
**1.0 = tam çöküş**, düşük = iyi conditioning. `top-1` = tahmin kendi objesinin
GT'sine mi en yakın; şans = 1/N.

### Temsil tavanı (oracle — encoder/transformer YOK)
| etiket | ayar | PSNR |
|---|---|---|
| `O1_oracle_mlp2` | triplane 64²×32, NeRF MLP **2 kat** | **25.04 dB** ← tavan |
| `O2_oracle_mlp10` | NeRF MLP 10 kat | 21.66 dB ← **daha derin DAHA KÖTÜ** |
| `O4_oracle_tp32` | triplane **32²** | 22.89 dB ← 64² doğru seçim |

→ Temsil sağlıklı. OpenLRM/TripoSR'ın 10 katlı MLP'si bizim rejimde işe yaramıyor.

### Mimari
| etiket | ayar | PSNR | top-1 |
|---|---|---|---|
| `A_baseline_joint` | joint self-attention | 17.06 | %25 |
| `B_cross_attn` | cross-attention | 17.05 | %22 |

→ **Cross-attention fayda vermiyor.** Kovalamayın.

### Kayıp fonksiyonu (hepsi 32 obje, aynı 2000 adımlık program)
| etiket | `w_lpips` | PSNR | top-1 |
|---|---|---|---|
| `A_baseline_joint` | 2.0 | 17.06 | %25 |
| **`R3_lpips025`** | **0.25** | **18.83** | **%62** ← optimum |
| `C1_lpips0` | 0.0 | 18.09 | %66 (gürültülü çıktı, yüksek LR'de ıraksıyor) |
| `R4_lpips05_fg20` | 0.5 + fg 20 | 16.69 | %28 |
| `C2_fg20` | 2.0 + fg 20 | 17.59 | %38 |

→ **`w_lpips = 0.25`.** LRM'in 2.0'ı 730k obje + batch 1024 rejimi için; bizim
rejimde LPIPS baskın olunca "makul genel doku" ödüllendirilip ortalama havzasında kalınıyor.

### Öğretmen triplane önyüklemesi
| etiket | ölçek | PSNR | top-1 |
|---|---|---|---|
| `TCH1_distill1` | 32 obje | 26.68 | %100 |
| `TCH4_fgcrop` | 32 obje + fg crop | **26.90** | %100 |
| `TCH3_yeni_gorunum` | 32 obje, **eğitimde görülmemiş kamera açısı** | 26.79 | %100 |
| `G250_teacher` | 250 obje | 25.90 | %98 |
| `G250_ogretmensiz` | 250 obje, öğretmensiz | 18.97 | %30 |
| `P250_kismi19` | 250 obje, 3000 adım | 20.79 | %56 |

**⚠️ ÖĞRETMEN NUMARASI ÖLÇEĞE BAĞLI — BIRAKILDI.**
Öğretmenin obje başına aldığı güncelleme = `batch ÷ veri_boyutu`:
32→250, 250→128, **2635→34**, 50k→1.8. 2635'te üç deneme de başarısız
(öğretmen öğrenciden kötü kaldı; sonra paylaşılan NeRF çöküp öğretmeni de öldürdü).
İki aşamaya ayırmak (`fit_teacher.py` + `distill_lrm.py`) öğretmeni 27.43 dB'ye
çıkardı ama öğrenci 1024 objede bağıl hata ~0.71'de takıldı (32 objede 0.41'di).

### 2026-08-24 gecesi — çerçeveleme A/B
Aynı 32 uid, aynı bayraklar, **tek değişken render dizini**:

| etiket | veri | PSNR | top-1 | oran |
|---|---|---|---|---|
| `AB_ESKI` | `dataset/renders` (sphere+Cycles) | 15.80 | %3 | **1.000 → tam çöküş** |
| **`AB_YENI`** | `dataset/renders_opp_score3` (fit+EEVEE) | **20.11** | **%75** | **0.60** |

→ **Yeni çerçeveleme conditioning'i kurtarıyor.** Yeniden render sorusu kapandı.

### 2026-08-24 gecesi — görünüm kapsaması (alt görünüm gerekli mi?)
`scripts/bench_view_coverage.py`. 24 obje, her iki koşulda da **12 fit görünümü**
(sayı eşit), ikisi de aynı **held-out ALT görünümlerde** (el −25…−70) ölçüldü.

| koşul | fit PSNR | held-out ALT PSNR |
|---|---|---|
| `COV_alt0` — 12 görünümün hepsi el ≥ −10 (alt yok) | 30.47 | **27.26 dB** |
| `COV_alt1` — biri alt kutup (el −88) | 30.10 | **27.74 dB** |

→ **+0.48 dB, belirsizlik eşiğinin altında.** Sadece üstten/yandan 12 görünümle
objenin altı zaten 27 dB'de kurulabiliyor. **sphere20 için yeniden render gerekmiyor.**
*Uyarı: bu supervision tarafı. Girdi tarafı (üst/alt fotoyu GİRDİ olarak vermek) ayrı soru, ölçülmedi.*

### 2026-08-24 gecesi — tarife matrisi (32 obje, aynı alt küme, tek değişken)
| etiket | değişken | PSNR | top-1 | oran |
|---|---|---|---|---|
| `M_crop08` | `--crop 0.8` | 17.48 | **%38** | **0.764** ← en iyi conditioning |
| `M_batch8` | `--batch 8` | **19.29** | %31 | 0.805 ← en iyi PSNR |
| `M_base` | — | 17.80 | %3 | 1.000 (çöktü) |
| `M_enc4` | `--unfreeze_last 4` | 17.80 | %3 | 1.000 → **GEÇERSİZ** |
| `M_encall` | `--train_encoder` | 16.93 | %12 | 1.013 |

→ İşe yarayan iki şey: **ön-plana yanlı kırpma** ve **büyük batch**.

**`M_enc4` neden geçersiz:** `M_base` ile bit-bazında aynı — loss değerleri dört
ondalığa kadar aynı, çıktı PNG'lerinin **md5'i bile aynı**. Encoder açılmış görünüyor
(39.6M→46.7M param, `enc_train=4`) ve mekanizma izole testte çalışıyor (gradyanlar
akıyor, 58 tensörde sıfırdan farklı). O koşuda neden etkisiz kaldığı **bulunamadı.**
→ **Kısmi encoder çözme sorusu CEVAPSIZ.** Yeniden koşulmalı, ağırlıkların gerçekten
değiştiğini doğrulayan bir kontrolle (başlangıç-son ağırlık farkını logla).

---

## 7. ⚠️ Metodolojik uyarılar — bunları okumadan ölçüm yapma

### 7.1. 32-obje kapısı HANGİ 32 objeye çok duyarlı
Aynı veri, aynı ayarlar, farklı obje alt kümesi:

| koşu | seçim | top-1 |
|---|---|---|
| `AB_YENI` | `_ab_uids.json`'daki 32 uid | **%75** |
| `NEW32_lpips025` | `--n_obj 32`, listeden adımlayarak | %19 |
| `M_base` | `--n_obj 32`, listeden adımlayarak | **%3** |

Son ikisi aynı seçim mantığını kullanıyor ama farklı sonuç verdi — aradan
`train_list` yeniden üretildiği için seçilen objeler kaymış olabilir.

**Kural: sadece aynı uid kümesi içinde karşılaştır.** Farklı koşulardan gelen
top-1 sayılarını yan yana koyma. A/B kollarını mutlaka `--uids` ile sabitle.

### 7.2. A/B kurarken tek değişkenin gerçekten tek olduğunu doğrula
Bu projede iki kez sessizce bozuldu:
1. **EEVEE vs CYCLES:** iki kol farklı `--uid` ile render edildi. Kamera açıları
   uid'den seed'leniyor (`_seed_from_uid`) → farklı kameralara bakıldı.
2. **Eski vs yeni çerçeveleme:** "yeni" kolu `--render_dir` bayrağını hiç almamıştı
   → varsayılan (eski) dizini okudu, A kolunun birebir tekrarıydı.

**Kural:** kolu başlatmadan çalışan sürecin komut satırını oku
(`Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select CommandLine`).

### 7.3. Küçük örneklemle ölçme tuzağı
"bbox çerçeveleme 1.87× kazanç" bulgusu 6 objeyle ölçülmüştü, 100 objede çürüdü.
**En az 24–40 obje kullan.**

### 7.4. 32-obje sonucu ölçeğe transfer etmeyebilir
Öğretmen numarası 32 objede %100 verdi, 2635'te tamamen kırıldı. 32-obje kapısı
**ezberleme** ölçer, genelleme değil. Yön verir, karar vermez.

### 7.5. Objaverse++ bayrakları **string**
`"false"` / `"true"` — Python'da `"false"` truthy'dir.
`str(a[f]).lower() == "true"` kullan. Bu hata bir kez yapıldı, tüm filtreleme yanlış çıktı.

### 7.6. Proje yolu **Ğ** içeriyor
kullanıcı adındaki Türkçe karakter (Ğ) → tüm json okuma/yazmalarda `encoding="utf-8"` şart, cp1254 varsayılanı çöker.

### 7.7. Eski çıktı dosyaları klasörlerde duruyor
`dataset/lrm_val_previews/` içinde 08-20 tarihli önizlemeler bu geceki koşununkilerle
karışık. Dosya tarihine bak.

### 7.8. bf16 rengi öldürüyor
Eğitim varsayılanı **fp32**. TF32 açık (Blackwell'de matmul hızlanır, çıktı fp32 kalır).

---

## 8. Son durum (2026-08-25 05:01 — gece zinciri bitti, 558 dk)

13k objede tam eğitim **tamamlandı**: `train_lrm.py`, 16.000 adım, render 128,
micro_batch 2 × grad_accum 4 (efektif 8), `w_lpips 0.25`, öğretmen kapalı.

Taban tarifeyle koştu (crop yok) — çünkü `overnight.sh` seçicisi `M_crop08`'i
kazanan seçti ama `train_lrm.py`'de crop desteği yok, hiçbir bayrak aktarılamadı.
**Bu bir tasarım kusuru, düzeltilmeli.**

### İlk dürüst genelleme eğrisi (görülmemiş 64 val objesi, tek girdi)
```
adım  500–8000  : top-1 = şans (%1.6), oran ≥1.0      → tam çöküş
adım      8500  : sıçrama (coarse→fine geçişi, res 64→128)
adım 9000–11000 : top-1 %3.1→%6.2→%7.8, PSNR 15.77→16.84
adım 11000–15500: PSNR 16.7–16.8 SABİT, top-1 %9–14, oran 1.04–1.08
FINAL (15500)   : PSNR 16.72   top-1 %14.1 (şans %1.6)   oran 1.054
```

**Yorum:** Taban tarife tam çöküşten kurtuluyor — top-1 şansın ~9 katı, yani model
girdiyi *bir miktar* kullanıyor. Ama **gerçek conditioning'e ulaşmıyor**: `oran > 1.0`
demek, her tahmin kendi hedefi için hâlâ veri-seti-ortalaması blob'undan daha kötü.

Son 4.500 adım (cosine kuyruğu, lr 3e-05 → 0) **hiçbir şey kazandırmadı**; PSNR
16.8'de sabit kaldı. Yani bu tarifenin tavanı bu. Ölçüm öncesi beklenti "+0,3–0,5 dB"
idi, gerçekleşen **0 dB** — beklenti bile iyimserdi.

**Sonuç: taban tarife 13k'da yetersiz.** Sıradaki adım matrisin kazananlarını
(ön-plana yanlı kırpma, büyük batch) tam veriye taşımak.

Checkpoint: `dataset/lrm_ckpts/last.pt` (adım 16.000).
Val eğrisi: `dataset/lrm_logs/val_metrics.jsonl` (31 satır).

## 9. Açık sorular (öncelik sırasıyla)

| # | soru | neden önemli | maliyet |
|---|---|---|---|
| 1 | **Kısmi encoder çözme** gerçekten işe yarıyor mu? | `M_enc4` geçersiz çıktı; LRM/OpenLRM/TripoSR **üçü de encoder'ı eğitiyor** | ~20 dk |
| 2 | `train_lrm.py`'ye **crop desteği** | matrisin kazananı, aktarılamadı | kod + koşu |
| 3 | **Efektif batch** büyütmek (şu an 8) | matriste ikinci en iyi | koşu |
| 4 | Encoder için **ayrı düşük LR** | şu an encoder de 4e-4 alıyor, ön-eğitilmiş ViT için yüksek | kod + koşu |
| 5 | **Üst/alt fotoyu GİRDİ olarak** vermek conditioning'i düzeltir mi | objelerin %30'unda 4 yan görünüm neredeyse aynı (MSE<0.01), %6'sı özdeş | ~20 dk (sphere20 render'ları hazır) |
| 6 | `normalize_cams` (LRM kamera normalizasyonu, +3.7 PSNR iddiası) | şu an **kapalı**; küçük veride yakınsamayı yavaşlattığı ölçüldü | koşu |
| 7 | Triplane hacminin sadece **%34'ü** kullanılıyor | `bound` daraltmak kapasite kazandırır mı | koşu |
| 8 | **50k'ya çıkma** (Objaverse++ etiket listesi lazım) | 13k'da genelleme eğrisi görülmeden anlamsız | ~5 saat render |

---

## 10. Kod haritası

### Faz A
| dosya | iş |
|---|---|
| `scripts/render_object.py` | tek objeyi Blender'da render eder + meta.json yazar. `NORM_MODE`, `RENDER_ENGINE` env |
| `scripts/camera_poses.py` | görünüm listesi. `RENDER_VIEW_SCHEME=ring12` (varsayılan, 16 görünüm) / `sphere20` (24, üst+alt) |
| `scripts/run_batch.py` | çok işçili toplu render + zaman damgalı log + `render_summary.json` |
| `scripts/build_subset.py` | LVIS / Objaverse++ kesişimi |
| `scripts/filter_dataset.py` | alpha kaplama + kenar taşma filtresi → `train_list.json`. `min_cov=0.010` |

### Faz B — model
| dosya | iş |
|---|---|
| `scripts/lrm/encoder.py` | donuk DINOv2 ViT-S/14. `forward` `@torch.no_grad()` ile sarılı — çözmek için `__wrapped__` ile aç |
| `scripts/lrm/transformer.py` | joint self-attention, dim 512 × 12 kat |
| `scripts/lrm/triplane.py` | `sample_triplane(tp, pts, bound)` |
| `scripts/lrm/nerf.py` | `TriplaneNeRF.forward(feats)` — **bakış yönü ALMAZ** (görünümden bağımsız renk) |
| `scripts/lrm/renderer.py` | `volume_render(o, d, near, far, n_samples, q, bg_color)` |
| `scripts/lrm/cameras.py` | `rays_from_camera`, `scale_intrinsics`, `canonicalize` |
| `scripts/lrm/dataset.py` | `LRMDataset` — 1–N girdi + n_sup supervision seçer, girdiye augmentation |
| `scripts/lrm/losses.py` | `LRMLoss(use_lpips, w_lpips)` |
| `scripts/lrm/model.py` | `LRM(n_samples, cross_attn)` |

### Faz B — koşu ve teşhis
| dosya | iş |
|---|---|
| `scripts/train_lrm.py` | tam eğitim döngüsü + checkpoint/resume + **sayısal val** (`build_val_probe`, `val_metrics`) |
| `scripts/bench_overfit.py` | **N-obje ezberleme kapısı** — top-1 + ortalama-baseline oranı + yeni görünüm |
| `scripts/bench_triplane_fit.py` | oracle / temsil tavanı |
| `scripts/bench_view_coverage.py` | görünüm kapsaması yeterli mi (yeni) |
| `scripts/bench_teacher.py`, `bench_distill.py` | öğretmen/distilasyon |
| `scripts/fit_teacher.py`, `distill_lrm.py` | iki aşamalı öğretmen |
| `scripts/diag_retrieval.py` | çökme türü (koşullu ortalama mı) |
| `scripts/diag_*.py` | attn/dino-renk/öğretmen-kalite/train-vs-val teşhisleri |
| `scripts/overnight.sh` | gece deney zinciri |
| `scripts/overfit_lrm.py`, `overfit_randbg.py` | erken ezberleme denemeleri (tarihsel) |
| `scripts/probe_conditioning.py` | conditioning sondası |
| `scripts/check_env.py` | ortam/CUDA doğrulaması |
| `scripts/verify_render.py` | render çıktısı doğrulaması |

### Val metriği tasarımı (`train_lrm.py`)
Girdi = kanonik[0] (ön, az 0 el 20), hedef = kanonik[1] ve [2] (yan + arka).
Kanonik açılar tüm objelerde aynı olduğu için metrik **objeler arası
karşılaştırılabilir**; supervision görünümleri uid-seed'li rastgele olduğu için
onlarla bu yapılamaz. Val objeleri eğitimde hiç görülmez.
Çıktı: `psnr`, `top1` (şans = 1/N), `mse`, `mean_mse`, `ratio`.
Doğrulandı: eğitilmemiş modelde `top1 = şans`, `oran = 1.000`.

---

## 11. Nerede ne var

```
dataset/renders_opp_score3/     güncel render seti (14.597 obje × 16 görünüm, 37 GB)
dataset/train_list_opp_score3.json   13.009 train / 1.445 val
dataset/lrm_bench/*.json        her deneyin ham sonucu + tam config
dataset/lrm_bench/*.png         her deneyin GT|tahmin görseli
dataset/lrm_ckpts/last.pt       en son checkpoint
dataset/lrm_logs/train_*.log    eğitim logu (zaman damgalı)
dataset/lrm_logs/val_metrics.jsonl   val eğrisi (makine okunur)
dataset/lrm_val_previews/       val önizleme görselleri
dataset/overnight_logs/         gece zinciri: 00_zincir.log = özet
docs/lrm-blob-diagnosis.md      blob teşhisinin tam zinciri
docs/faz-b-kavramlar.md         kavram sözlüğü (LRM'in her parçası ne işe yarar)
docs/egitim-oncesi-netlestirme-plani.md   açık/kapalı soru envanteri
```

---

## 12. Çalışma kuralları

- Yaratıcı/kurulum işine başlamadan **superpowers skill'lerini** kullan
  (brainstorming → writing-plans → TDD → systematic-debugging)
- Spec'ler `docs/superpowers/specs/YYYY-MM-DD-<konu>-design.md`
- Bu proje kişisel öğrenme amaçlı: kararları **açıklayarak** ilerle
- Testler: `python -m pytest tests/ -q` (şu an 63 test geçiyor)
