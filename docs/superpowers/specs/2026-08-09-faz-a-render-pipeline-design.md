# Faz A — Render Pipeline (Tasarım Dokümanı)

**Tarih:** 2026-08-09
**Durum:** Onay bekliyor
**Kapsam:** Objaverse `.glb` objelerini çok-görünümlü render + kamera pozlarına çeviren pipeline.
Sadece Faz A. Model eğitimi (Faz B) ve web app (Faz C) ayrı spec'lerdir.

---

## 1. Amaç

Faz B'deki multi-view LRM eğitiminin ihtiyacı olan veri setini üretmek: her obje için
sabit kanonik açılardan + çeşitli supervision açılarından render'lanmış görüntüler ve
bu görüntülerin **tam kamera pozları** (extrinsic + intrinsic).

**Başarı kriteri:** ~1000 obje için, her biri 16 görüntü + `meta.json` içeren, Faz B'nin
doğrudan yükleyip eğitebileceği tutarlı bir `dataset/` klasörü. Çıktı gözle doğrulanabilir
(contact sheet), pipeline durup kaldığı yerden devam edebilir (resume).

## 2. Girdi & Ortam

- **Girdi objeler:** `C:\Users\ENES OĞUZ\.objaverse\hf-objaverse-v1\glbs\**\<uid>.glb`
  (51.534 adet, `<uid>.glb`).
- **Blender:** 4.4, `C:\Program Files\Blender Foundation\Blender 4.4\blender.exe`.
- **GPU:** RTX 5080 (Blackwell), Cycles + **OPTIX**.
- **OS:** Windows 11, PowerShell + Python 3.10 (sürücü scriptleri için).

## 3. Kamera Geometrisi (kesinleşmiş)

### 3.0 Normalizasyon & Çerçeveleme (framing tutarlılığı)

Objeler boyut/şekil olarak çok değişir; tutarlı çerçeveleme için **obje kameraya göre
ayarlanır** (kamera objeye göre değil):

1. **Merkezleme:** bbox merkezi origin'e taşınır.
2. **Bounding-sphere ölçekleme:** `radius_norm = |bbox_max − merkez|` (köşe mesafesi, gerçek
   bounding sphere'in güvenli üst sınırı) hesaplanır; `ölçek = TARGET_RADIUS / radius_norm`
   ile obje **sabit yarıçaplı bir küreye** (`TARGET_RADIUS = 0.5`) sığdırılır.
   - **Neden max-bbox değil bounding-sphere?** Küre yönden bağımsızdır → obje 16 açının
     hepsinden ve her objede AYNI açısal boyutta görünür, köşeden bakışta bile taşmaz.

**Kamera mesafesi lens FOV'undan türetilir** (sabit sayı değil):
```
FOV = 2·atan(sensor / (2·lens)) = 2·atan(32/70) ≈ 49.2°  → yarı-FOV ≈ 24.6°
mesafe = TARGET_RADIUS / sin(FILL_FACTOR · yarı-FOV)
       = 0.5 / sin(0.80 · 24.6°) ≈ 1.48   (obje çerçevenin ~%80'ini doldurur)
```
`FILL_FACTOR = 0.80` (ayarlanabilir: yakın=daha çok detay/taşma riski, uzak=güvenli boşluk).
`radius ≈ 1.5` bu hesabın sonucudur.

### 3.1 Kamera pozisyonları

Kamera sabit yarıçapta (yukarıdaki `radius ≈ 1.5`), hep origin'e bakar (`TRACK_TO` empty).
İç parametreler: `lens = 35mm`, `sensor_width = 32mm` (orijinal Objaverse script ile aynı).
Ölçek/mesafe sabit olsa da her görünümün **gerçek extrinsic/intrinsic'i** `meta.json`'a yazılır.

Kamera pozisyonu küresel koordinattan (azimuth θ, elevation φ):
```
x = radius * cos(φ) * cos(θ)
y = radius * cos(φ) * sin(θ)
z = radius * sin(φ)
```

**Toplam 16 görünüm:**

| index | Grup | Azimuth | Elevation | Belirleme |
|---|---|---|---|---|
| 000–003 | **Kanonik** (input) | 0°, 90°, 180°, 270° | **+20°** (sabit) | Tüm objelerde AYNI |
| 004–015 | **Supervision** (12) | rastgele [0°,360°) | rastgele **[−10°, +80°]** | uid-seed'li rastgele |

**Supervision örnekleme kuralları:**
- Seed = `uid`'den türetilir (ör. `hash(uid) & 0xffffffff`) → **tekrar üretilebilir**.
- Elevation, alan-uniform olması için `sin(φ)` üzerinden örneklenir (kutuplarda yığılma olmaz).
- **Tam kutup yok** (±90°) → `TRACK_TO` gimbal sorunu ve anlamsız taban render'ından kaçınmak için.
- Aralık [−10°, +80°]: çoğunlukla üstten + biraz ufuk-altı; objenin üstünü de besler.

## 4. Render Ayarları

- **Motor:** `CYCLES` (EEVEE DEĞİL — 4.4/5.0 arası ad değişimi + headless GPU-context sorunları).
- **Cihaz:** GPU/OPTIX. Fallback: CUDA.
  ```python
  prefs = bpy.context.preferences.addons["cycles"].preferences
  prefs.compute_device_type = "OPTIX"
  prefs.refresh_devices()
  for d in prefs.devices:
      d.use = d.type in {"OPTIX", "CUDA"}
  scene.cycles.device = "GPU"
  ```
- **Çözünürlük:** 512×512, RGBA PNG.
- **Sample:** ~48 + OPTIX denoise (`use_denoising = True`, denoiser = "OPTIX").
- **Şeffaf arka plan:** `scene.render.film_transparent = True` → alpha kanalı = obje maskesi (bedava mask).
- **Işık:** tutarlı, sabit ışık düzeni (even lighting) — tüm açılarda albedo görünür olsun,
  pişmiş sert gölge geometriyi yanıltmasın. World sabit + area light rig.

## 5. Çıktı Formatı

```
dataset/
  renders/<uid>/
    000.png ... 015.png          # 512×512 RGBA (000–003 kanonik, 004–015 supervision)
    meta.json                    # aşağıdaki şema
  index.json                     # tamamlanan uid listesi + train/val split
  manifest.jsonl                 # her obje için 1 satır log (resume/skip için)
```

**`meta.json` şeması:**
```json
{
  "uid": "001abb1a3f4c412fbd707239acb68cd6",
  "resolution": 512,
  "num_views": 16,
  "camera": { "lens_mm": 35, "sensor_mm": 32, "radius": 1.48,
              "target_radius": 0.5, "fill_factor": 0.80 },
  "canonical_indices": [0, 1, 2, 3],
  "views": [
    {
      "index": 0, "role": "canonical", "file": "000.png",
      "azimuth_deg": 0.0, "elevation_deg": 20.0,
      "extrinsic": [[...4x4 world-to-camera...]],
      "intrinsic": [[...3x3...]]
    }
  ]
}
```

**`manifest.jsonl`** (satır başı bir obje):
```json
{"uid": "...", "status": "done|failed", "views": 16, "seconds": 19.3, "error": null, "blender": "4.4"}
```

## 6. Bileşenler (Faz A teslimatları)

Sorumlulukları ayrık 4 küçük birim:

1. **`scripts/render_object.py`** — *Blender içinde* çalışır (`blender -b -P`).
   Tek obje → 16 görünüm + `meta.json`. `bpy` kullanır. Argümanlar `--` sonrası argparse.
   Alt fonksiyonlar: `reset_scene`, `load_glb`, `normalize_scene`, `setup_lighting`,
   `sample_views(uid)`, `render_view`, `write_meta`.

2. **`scripts/build_subset.py`** — *sistem Python*. 51.534 glb içinden N (default 1000) uid'yi
   sabit seed ile seçer → `subset.json` (uid → glb yol). Bozuk/çok büyük glb filtresi (opsiyonel).

3. **`scripts/run_batch.py`** — *sistem Python*, sürücü. `subset.json`'u gezer, her obje için
   `blender.exe`'yi subprocess ile çağırır. Özellikler:
   - **Resume/skip-completed:** `meta.json` + 16 png varsa atla.
   - **Timeout:** obje başına ~120 sn; aşan bozuk glb'yi kill et, `failed` logla, devam et.
   - **manifest.jsonl** yaz, ilerleme çıktısı ver.
   - (Not: Windows'ta xserver GEREKMEZ; `distributed.py`/`start_xserver.py` KULLANILMAZ.)

4. **`scripts/verify_render.py`** — *sistem Python*. Birkaç objenin 16 görünümünü tek bir
   contact-sheet PNG'ye dizer → kaliteyi gözle kontrol. Poz/dosya bütünlüğü sanity-check.

## 7. Test Stratejisi

Blender render'ı birim-test edilemez; ama saf mantık edilebilir (TDD):
- `sample_views(uid)` poz matematiği → deterministiklik, aralık sınırları, seed tekrarı.
- extrinsic/intrinsic matris üretimi → bilinen açı için beklenen kamera konumu.
- `build_subset` → seed tekrarı, N adet, geçerli yollar.
- `run_batch` resume mantığı → tamamlanmışı atlar, yarımı yeniden dener.
- Render kalitesi → `verify_render.py` contact sheet ile gözle (birkaç örnek obje).

## 8. Riskler & Kararlar

- **Blender 3.2 → 4.4 API farkı:** orijinal script uyarlanacak (motor CYCLES seçildiği için
  EEVEE ad sorunu bypass edildi; glTF importer flag'leri 4.4'te doğrulanacak).
- **Bozuk/dev glb'ler:** timeout + failed-logla-devam et ile izole edilir.
- **Işık tutarlılığı:** even lighting rig; ilk contact sheet'te gözden geçirilip ayarlanır.
- **Ölçek:** önce ~1000 obje ile doğrula (~5-6 saat). Çalışınca `build_subset` N'i artırılır.

## 9. Kapsam Dışı (Faz A değil)

- Model mimarisi / eğitim (Faz B).
- Arka plan silme, web app, three.js (Faz C).
- Çok-GPU / çok-makine dağıtımı (tek RTX 5080 yeterli).
- Depth/normal map render (şimdilik sadece RGBA; gerekirse sonra eklenir).
