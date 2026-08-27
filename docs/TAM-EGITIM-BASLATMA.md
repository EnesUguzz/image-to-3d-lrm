# Tam Eğitim — Başlatma Talimatı

*2026-08-27 akşamı hazırlandı. **Bu dosya yeni bir oturumun tek ihtiyacı olan şeydir.**
Kullanıcı "başlat" dediğinde önce §1'i doğrula, sonra §2'deki komutu çalıştır.*

---

## 1. Başlatmadan önce — 3 dakikalık doğrulama

```bash
# a) Test tabanı sağlam mı  (beklenen: 155 passed)
python -m pytest tests/ -q

# b) ETKİN konfigürasyonu OKU (10 sn) — komuta --dry_run ekleyerek
#    Beklenen: HİÇ "!!" uyarısı olmayacak.
#    Özellikle bak: density_bias 0.0, bound 0.6, normalize_cams False,
#    uids_file "-", teacher_subset 0, warmup_orani < 0.2

# c) Girdi dosyaları yerinde mi
ls -la dataset/lrm_ckpts/distilled_v2.pt          # 564 MB, adım 20000
ls -la dataset/train_list_v2.json                 # train 12320 / val 1424 / test 500
ls -d  dataset/renders_opp_score3                 # 14.597 obje × 16 görünüm

# d) Disk  (checkpoint 620 MB × ~7 dosya ≈ 4.5 GB gerekir)
df -h .
```

⚠️ **`git status` temiz olmalı.** Koşu künyesi (`runstamp`) `git_sha` + `git_dirty`
kaydediyor; kirli ağaçta koşulan bir eğitim **tekrar üretilemez**. Kirliyse önce commit et.

---

## 2. Komut

```bash
python scripts/train_lrm.py \
  --train_list dataset/train_list_v2.json --renders_dir dataset/renders_opp_score3 \
  --steps 30800 --micro_batch 8 --grad_accum 2 --workers 6 --amp \
  --teacher_subset 0 --unfreeze_last 4 --enc_lr_scale 0.1 \
  --density_bias 0.0 --bound 0.6 --warmup 3000 --w_lpips 0.25 \
  --val_every 1000 --ckpt_every 2000 --snapshot_every 5000 --val_n 64 \
  --ckpt_dir dataset/lrm_ckpts_tam --tag TAM_asama3 \
  --init_from dataset/lrm_ckpts/distilled_v2.pt
```

12.320 obje × 40 epoch = 30.800 adım ≈ **9,9 saat** (0,86 it/s ölçüldü).

**Kesilirse:** aynı komuta `--resume` ekle. Test edildi (`tests/test_checkpoint_roundtrip.py`
+ 40→80 adım duman testi).

**Uzatmak istersen:** `--steps 46200 --resume` (60 epoch, +5 saat).

---

## 3. Neden bu değerler

| bayrak | gerekçe |
|---|---|
| `--init_from distilled_v2.pt` | Sıfırdan render kaybı **iki kez ölçüldü, ölçekte çalışmıyor** (kol A 6000 adımda 8,8 dB; kol D boş sahneye çöktü). Distile ağırlık doğru havzada başlatıyor. |
| `--density_bias 0.0` | Distile checkpoint'in NeRF'i `TriplaneNeRF(in_dim=96, hidden=64)` ile, yani bias 0 varsayımıyla oturdu (`fit_teacher.py:66`). 1.0 verilirse geometri sisleniyor — ölçüldü, 24 → 16,3 dB. **Kod artık uyuşmazlıkta koşuyu reddediyor.** |
| `--bound 0.6` | Aynı gerekçe. 0,552 kalibrasyonu **sıfırdan** bir koşu için geçerli, distile init'le değil. |
| `--teacher_subset 0` | Ortak-öğretmen fazı Blok 2'de terk edildi. Varsayılanı 512, açık kalırsa koşunun yarısı boşa gider (fiilen yaşandı). |
| `--warmup 3000` | Referans (OpenLRM) değeri; koşunun %9,7'si. |
| `--snapshot_every 5000` | `last_*.pt` her seferinde **eziliyor**. C2 koşusunda kalite ortada tepe yapıp düştü (in-sample 24,33 → 22,00); arşiv olmadan en iyi model kaybolur. |
| `--unfreeze_last 4` | Referans implementasyonların üçü de encoder'ı eğitiyor. `enc_lr_scale 0.1` ile temkinli. |
| `--w_lpips 0.25` | 32 objede ölçüldü: 2.0 → 17,06 dB/%25; **0,25 → 18,83 dB/%62**; 0.0 gürültülü. |

---

## 4. Koşu sırasında ne üretilecek

| ne | nerede |
|---|---|
| adım logu (loss bileşenleri, lr, hız, VRAM) | `dataset/lrm_logs/train_<zaman>.log` |
| **sayısal val, her 1000 adım** (held-out) | aynı log + `dataset/lrm_logs/val_metrics.jsonl` |
| **görsel önizleme, her 1000 adım** (GT \| tahmin) | `dataset/lrm_val_previews/TAM_asama3/val_XXXXXX.png` |
| checkpoint (ezilen) | `dataset/lrm_ckpts_tam/last_TAM_asama3.pt` |
| **arşiv checkpoint (ezilmeyen), her 5000 adım** | `dataset/lrm_ckpts_tam/snap_TAM_asama3_stepXXXXXX.pt` |

Val satırı her seferinde **TAVAN/TABAN/RAKİP** üçlüsünü basar; tek bir PSNR
yorumlanamaz (`docs/metrik-sartnamesi.md`).

---

## 5. Nöbet — koşuyu ne zaman KES

Bunlar önceden yazıldı; koşarken gevşetme.

| işaret | ne demek | ne yap |
|---|---|---|
| `DEJENERE! EMPTY_COLLAPSE` (`acc→0`) | sahne boşaldı, softplus gradyanı öldü | **KES.** `--density_bias 1.0` ile değil — o distile init'i bozar. Önce önizlemeye bak. |
| `acc ≈ 0.5` sabit + `mask` düşmüyor | dolu-küp çöküşü (kol A) | **KES.** `--mask_fg_weight` ara değer (1–2) dene. |
| `obj_std < 0.010` | objeler arası fark yok = blob | **KES.** |
| `oran > 1.0` **ve** adım 10.000'i geçti | ortalama görüntüden kötü | kesme, ama not al |
| `psnr <= komsu_psnr` adım 15.000'de | retrieval yapıyor olabilir (Tatarchenko) | kesme, sonda değerlendir |

**Sağlıklı seyir referansı** (C2, 1024 obje, held-out):
top-1 %25 → %41 tırmandı, PSNR 17,4'te (taban 17,36) sabit kaldı.
Tam koşuda **top-1'in daha yükseğe, `oran`ın 1,0'ın altına** inmesi bekleniyor —
çünkü 3. aşama bu kez 12.320 objenin **11.296'sını (%92) ilk kez** görüyor.

---

## 6. Koşu bitince

```bash
# 1) Sayısal: in-sample ve held-out AYRI raporlanır (karıştırma!)
python scripts/eval_suite.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --teacher dataset/lrm_ckpts/teacher_1024_tv.pt --n_obj 128 --res 128 --amp \
  --tag TAM_insample
python scripts/eval_suite.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --split val --no_teacher --n_obj 128 --res 128 --amp --tag TAM_val

# 2) Tek/çok görünüm (projenin kuzey yıldızı)
for N in 1 2 4; do
  python scripts/eval_suite.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
    --split val --no_teacher --n_input $N --n_obj 128 --res 128 --amp --tag TAM_val_in$N
done

# 3) Geometri (yalnız kapıda)
python scripts/eval_geometry.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --split val --n_obj 32 --tag TAM_geom

# 4) GÖRSEL — asıl çıktı bu. Mesh + render, held-out objelerde.
python scripts/extract_mesh.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --uid <VAL_UID> --auto_level --grid 256 --smooth 10 \
  --out dataset/mesh_tam --tag TAM
```

`--smooth 10`: yüzeydeki "köpük" için Taubin. Ölçüldü — yoğunluk alanı 64²
triplane'in altında yapı taşımıyor (gücün %95'i 3,5 hücre üstünde), yani
pürüzsüzleştirme **veri değil artefakt** siliyor. Hacim koruması var (%5).

---

## 7. Bilinen açık işler *(bu koşuyu engellemiyor, ama Faz C'den önce gerekli)*

1. **Gerçek foto değerlendirme seti (E2) YOK.** Tüm sayılar kendi render'larımız
   üzerinde. Kullanıcının telefon fotosunda nasıl davranacağı **ölçülmedi**.
2. **Mesh eşiği 4 GT silüetinden seçiliyor.** Faz C'de tek maske olacak →
   geometri sayıları bir miktar iyimser.
3. **Volume IoU hiç koşulmadı** (hep `--no_vol_iou` ile atlandı).
4. **Çıkarım süresi ölçülmedi** (foto → mesh, uçtan uca).
5. **Aydınlatma A/B koşulmadı** (~35 dk). Karar gerekçeli ama ölçülmedi.
6. **Öğretmen 12k objeye çıkamaz**: obje başına 1,57 MB × 12.320 = 19,4 GB > 16 GB
   VRAM (Adam'la 3×). Tavan ~2.000 obje. Bu yüzden 3. aşama zorunlu.
