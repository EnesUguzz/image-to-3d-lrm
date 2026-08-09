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

- **Faz A — Render pipeline** *(ŞU AN BURADAYIZ)*
  Objaverse `.glb`'lerini sabit kanonik açılardan çok-görünümlü foto + kamera pozlarına çevir.
- **Faz B — Eğitim (asıl iş)**
  Multi-view LRM: ViT encoder → transformer → triplane → NeRF render → görüntü/mask loss.
  Referans mimari: OpenLRM (küçük ölçekte anlayarak uyarlanacak).
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

## 🛠️ Çalışma Kuralları

- Yaratıcı/kurulum işine başlamadan **superpowers skill'lerini** kullan
  (brainstorming → writing-plans → TDD → systematic-debugging).
- Spec'ler `docs/superpowers/specs/YYYY-MM-DD-<konu>-design.md` altına yazılır ve commit'lenir.
- Bu proje kişisel öğrenme amaçlı: kararları **açıklayarak** ilerle.
