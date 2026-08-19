# 3D Object Project — Claude Talimatları

Bu dosya, bu projede çalışan her Claude oturumu için kalıcı bağlamdır. Kararlar burada; koda başlamadan önce oku.

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
  2927 temiz obje (2635 train + 292 val), `dataset/train_list.json`.
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

### Faz B — LRM eğitimi: uygulama + hata ayıklama kazanımları (2026-08-19)
- **Ortam:** `torch 2.11.0+cu128`, RTX 5080 sm_120 doğrulandı (`scripts/check_env.py`).
  DINOv2 (torch.hub) + LPIPS-VGG (~528MB) ilk çalıştırmada iner, sonra cache.
- **Meta.json şeması (Faz A çıktısı):** `intrinsic` **her view içinde** (`views[i]["intrinsic"]`),
  üstte `resolution`/`canonical_indices`; dosya adı `views[i]["file"]`. `c2w = inv(extrinsic)`
  (extrinsic = world→camera; rotasyondaki üniform ölçek yön normalizasyonunda iptal olur).
- **🔑 Overfit 3 kök-neden (fp32 tek-obje ile izole edildi — hepsi "loss tam donuyor" belirtisi):**
  1. **Boş çökme:** arka plan baskın (~%78) → "boş üret" düşük loss + softplus doygunluğu
     gradyanı öldürür. ÇÖZÜM: **fg-ağırlıklı MSE+mask** (`losses.py`, obje pikselleri ağır;
     empty→yüksek mask cezası). Uniform mask "boş üret"i ödüllendirir, KULLANMA.
  2. **Sisli dolgu:** `sample_triplane` `border` padding → objeyi ıskalayan ışınlar kenardan
     density toplar. ÇÖZÜM: `model.py` query'de **bound kübü dışı density = 0** (sınırlı obje = dışı boş).
  3. **Donuk siyah renk:** density raw noise rgb_head'i erken siyaha satüre eder. ÇÖZÜM:
     `nerf.py` **noise_std=0** (varsayılan). density_bias=0 de yeterli.
  - Ayrıca renderer'da **son delta 1e10 DEĞİL sonlu** olmalı (yoksa her ışın zorla opak → şeffaf yok).
  - Overfit objesi **yüksek-kaplamalı** seçilmeli (ince obje "boş üret" tuzağına düşer).
  - Sonuç: tek obje overfit loss 1.26→0.09, preview GT'ye birebir oturdu → pipeline sağlıklı.
- **Eğitim config (doğrulandı):** render 128, micro_batch 2 × grad_accum 4 (efektif 8),
  bf16, **grad_ckpt kapalı** (VRAM sadece ~4.4GB, bol boşluk), lr 4e-4 warmup+cosine.
  Hız ~**0.83 it/s**. Model tek-obje işler; micro_batch python-loop (paralel değil).
  Checkpoint `dataset/lrm_ckpts/last.pt` (optimizer dahil, `--resume`), val preview
  `dataset/lrm_val_previews/val_*.png`, log `dataset/lrm_logs/`.
- **İlk tam koşu:** 15000 adım (~45 epoch, ~5 saat), 2026-08-19 başlatıldı. Dönüşte val
  preview'lar + loss ile kontrol; yetmezse iterasyon (boyut/lr/augmentation/veri).

## 🛠️ Çalışma Kuralları

- Yaratıcı/kurulum işine başlamadan **superpowers skill'lerini** kullan
  (brainstorming → writing-plans → TDD → systematic-debugging).
- Spec'ler `docs/superpowers/specs/YYYY-MM-DD-<konu>-design.md` altına yazılır ve commit'lenir.
- Bu proje kişisel öğrenme amaçlı: kararları **açıklayarak** ilerle.
