# Tam Eğitim — Başlatma Talimatı

*Son güncelleme: 2026-08-27 gecesi, **üç bağımsız denetimden sonra.***
**Bu dosya yeni bir oturumun tek ihtiyacı olan şeydir.**
Kullanıcı "başlat" dediğinde: §1 doğrula → §2 çalıştır → §5 nöbet tut.

---

## 0. Bu belge en son ne zaman değişti ve neden

2026-08-27 akşamı üç bağımsız denetçi (model/eğitim kodu, metrikler, veri/render)
projeyi eleştirdi. **7 kritik bulgunun 7'si de kodla doğrulandı ve düzeltildi.**
Ayrıca 1 saatlik bir A/B koşuldu ve denetimin 1 numaralı önerisini **reddetti**.
Değişenler §7'de listeli. Testler: 137 → **164**.

---

## 1. Başlatmadan önce — 3 dakikalık doğrulama

```bash
# a) Test tabanı  (beklenen: 164 passed)
python -m pytest tests/ -q

# b) ETKİN konfigürasyon (10 sn) — §2 komutuna --dry_run ekle
#    Beklenen: HİÇ "!!" uyarısı YOK. Şunları gözle doğrula:
#      density_bias 0.0 | bound 0.6 | normalize_cams False | input_crop 0.0
#      uids_file "-"    | teacher_subset 0 | warmup_orani 0.097 | snapshot_every 5000
#    NOT: "!! distilled_v2.pt KUNYESIZ" satırı BEKLENEN ve zararsızdır —
#    o checkpoint render künyesi eklenmeden önce üretildi; kod zincirin
#    fiili varsayılanını (density_bias 0.0, bound 0.6) kabul edip uyarı basar.

# c) Girdiler
ls -la dataset/lrm_ckpts/distilled_v2.pt      # 564 MB, adım 20000
ls -la dataset/train_list_v2.json             # train 12320 / val 1424 / test 500
ls -d  dataset/renders_opp_score3             # 14.595 obje × 16 görünüm
df -h .                                       # ~4,5 GB gerekir (7 checkpoint)

# d) ⚠️ git status TEMİZ olmalı
git status --porcelain
```

`runstamp` `git_sha` + `git_dirty` kaydeder; **kirli ağaçta koşulan eğitim tekrar
üretilemez.** Kirliyse önce commit et.

⚠️ **Windows uykusunu kapat** — 10 saatlik gece koşusu.

---

## 2. Komut

```bash
python scripts/train_lrm.py \
  --train_list dataset/train_list_v2.json --renders_dir dataset/renders_opp_score3 \
  --steps 30800 --micro_batch 8 --grad_accum 2 --workers 6 --amp \
  --teacher_subset 0 --unfreeze_last 4 --enc_lr_scale 0.1 \
  --density_bias 0.0 --bound 0.6 --warmup 3000 --w_lpips 0.25 \
  --input_crop 0.0 \
  --val_every 1000 --ckpt_every 2000 --snapshot_every 5000 --val_n 64 \
  --ckpt_dir dataset/lrm_ckpts_tam --tag TAM_asama3 \
  --init_from dataset/lrm_ckpts/distilled_v2.pt
```

12.320 obje × 40 epoch = 30.800 adım ≈ **9,9 saat** (0,86 it/s ölçüldü).

**Kesilirse:** aynı komuta `--resume` ekle. Uçtan uca test edildi.
**Uzatmak:** `--steps 46200 --resume` (60 epoch, +5 saat).

---

## 3. Neden bu değerler

| bayrak | gerekçe |
|---|---|
| `--init_from distilled_v2.pt` | Sıfırdan render kaybı **iki kez ölçüldü, ölçekte çalışmıyor** (kol A 6000 adımda 8,8 dB — taban 17,36!). Ayrıca öğretmen 12k objeye çıkamıyor (1,57 MB/obje × 12.320 = 19,4 GB > 16 GB VRAM) ⇒ **3. aşama, tüm veriyi gören TEK aşama.** |
| `--density_bias 0.0` | Distile checkpoint'in NeRF'i bias 0 varsayımıyla oturdu (`fit_teacher.py:66`). 1.0 verilirse geometri sisleniyor — ölçüldü, 24 → 16,3 dB. Kod artık uyuşmazlıkta **koşuyu reddediyor**. |
| `--bound 0.6` | Aynı gerekçe. 0,552 kalibrasyonu sıfırdan koşu içindir. |
| `--input_crop 0.0` | **A/B ile ölçüldü ve kırpma REDDEDİLDİ** — §6'ya bak. |
| `--teacher_subset 0` | Ortak-öğretmen fazı Blok 2'de terk edildi; varsayılanı 512, açık kalırsa koşunun yarısı boşa gider. |
| `--warmup 3000` | OpenLRM değeri; koşunun %9,7'si. |
| `--snapshot_every 5000` | `last_*.pt` **eziliyor**. C2'de kalite ortada tepe yapıp düştü (in-sample 24,33 → 22,00); arşiv olmadan en iyi model kaybolur. |
| `--unfreeze_last 4` + `enc_lr_scale 0.1` | Referansların üçü de encoder'ı eğitiyor. |
| `--w_lpips 0.25` | 32 objede ölçüldü (2.0 → 17,06 dB/%25; **0,25 → 18,83/%62**). |

⚠️ **`--lr` varsayılanı 4e-4 ve TEK ÖLÇÜLMEMİŞ hiperparametre.** OpenLRM'den
kopyalandı ama onun **çok-GPU global batch'i** kopyalanmadı; bizim efektif
batch'imiz 16. §5'te nöbet maddesi var.

---

## 4. Koşu sırasında ne üretilecek

| ne | nerede |
|---|---|
| adım logu | `dataset/lrm_logs/train_<zaman>.log` |
| **held-out val**, her 1000 adım | log + `dataset/lrm_logs/val_metrics.jsonl` |
| **in-sample val**, her 1000 adım *(YENİ)* | aynı satırın altında `IN-SAMPLE ... ACIK = +X.XX dB` |
| **görsel önizleme**, her 1000 adım | `dataset/lrm_val_previews/TAM_asama3/val_XXXXXX.png` |
| checkpoint (ezilen) | `dataset/lrm_ckpts_tam/last_TAM_asama3.pt` |
| **arşiv checkpoint**, her 5000 adım *(YENİ)* | `dataset/lrm_ckpts_tam/snap_TAM_asama3_stepXXXXXX.pt` |

**Tek sayfada okumak için** *(YENİ araç)*:
```bash
python scripts/kosu_raporu.py --tag TAM_asama3
# -> egriler.png (PSNR+taban+rakip, oran, top-1, çöküş dedektörü)
#    onizleme.png (aynı objelerin adım adım değişimi)
#    ozet.md (ilk/son/en iyi + ön-kayıtlı kapıların geçti/kaldı tablosu)
```

---

## 5. Nöbet — koşuyu ne zaman KES

**Önce belirsizlik bandı** (n=64'te ölçüldü): PSNR **±0,67 dB**, top-1 **±5,7 puan**.
Bu bandın içindeki değişim **gürültüdür**, sinyal değil.

| işaret | ne demek | ne yap |
|---|---|---|
| `DEJENERE! EMPTY_COLLAPSE` (`acc→0`) | sahne boşaldı, softplus gradyanı öldü | **KES.** Önce önizlemeye bak. |
| `acc ≈ 0.5` sabit + `mask` düşmüyor | dolu-küp çöküşü | **KES.** *(Otomatik dedektör YOK — `ACC_MAX` tanımlı değil, gözle bak.)* |
| `obj_std < 0.010` | objeler arası fark yok = blob | **KES.** |
| **adım 3000–8000'de `oran` 1.10 üstüne çıkıp KALIRSA** | tepe LR distile başlangıcı yıkıyor olabilir | **KES**, `--lr 1.5e-4 --resume` ile devam et |
| `ACIK` (in-sample − val) **küçük** ve ikisi de tabanda | genelleme sorunu DEĞİL — yetersiz eğitim/kapasite | kesme, not al (kod bunu otomatik uyarıyor) |
| `ACIK` **büyüyor**, val düz | klasik aşırı uyum | kesme; 12k'da beklenmiyor, olursa raporla |
| `psnr <= komsu_psnr` adım 15.000'de | retrieval yapıyor olabilir (Tatarchenko) | kesme, sonda değerlendir |

**Sağlıklı seyir referansı** (C2, 1024 obje, held-out): top-1 %25 → **%41** tırmandı,
PSNR 17,4'te (taban 17,36) sabit kaldı. Tam koşuda **top-1'in daha yükseğe, `oran`ın
1,0 altına** inmesi bekleniyor — 3. aşama bu kez 12.320'nin **11.296'sını (%92)
ilk kez** görüyor.

⚠️ Eski bir tavsiye vardı: *"acc≈0.5 → `--mask_fg_weight` 1–2 dene."*
Bu **hipotezdir, kanıtlanmadı** — kol A ile C2 iki değişkende farklıydı.

---

## 6. A/B sonucu: girdi kırpma REDDEDİLDİ

Denetim, girdi çerçevelemesinin conditioning sinyalinin ancak %25'ini verdiğini
ölçtü (kaplama 0,092, DINOv2'nin 256 patch'inden 24'ü obje). Sıkı kırpma bunu
**0,250 / 64 patch'e (2,72×)** çıkarıyor. 250 obje × 1500 adımda ölçüldü:

| adım 1500, held-out | tam kare | sıkı kırpma |
|---|---|---|
| **top-1** | **%35,9** | %23,4 |
| IoU | **0,482** | 0,447 |
| LPIPS | **0,263** | 0,279 |
| oran | **1,052** | 1,094 |
| *in-sample* PSNR | 22,48 | **22,72** |

**Kırpma ezberlemeyi kolaylaştırıp genellemeyi bozuyor.** Muhtemel mekanizma:
tam karede objenin görüntüdeki büyüklüğü **gerçek ölçeğinin doğrudan ipucu**;
kırpma bunu normalize edip yok ediyor, model ölçeği tahmin etmek zorunda kalıp
görülmemiş objede yanılıyor (IoU düşüşü bununla tutarlı).

🔴 **FAZ C GEREKSİNİMİ:** kullanıcının fotoğrafı **sıkı kırpılamaz.**
`fit` normalizasyonunun çerçeveleme konvansiyonu yeniden üretilmeli.

---

## 7. Denetimden sonra NE DEĞİŞTİ

Hepsi doğrulandı (`dosya:satır` + ölçüm) ve düzeltildi:

| bulgu | etkisi | durum |
|---|---|---|
| Öğretmen **öğrencinin** NeRF'iyle render ediliyordu | TAVAN çizgisi koşu boyunca kayardı (%10 kayma = 1,6 dB) | ✅ `render_view(nerf=...)` |
| **In-sample probe yoktu** | "az mı eğitildi / genellemiyor mu" ayırt edilemezdi | ✅ eklendi + otomatik teşhis |
| `eval_suite --split` **sessizce yok sayılıyordu** | held-out sanılan rapor train ölçüyordu | ✅ split önceliği |
| `eval_geometry` **`--split` yoktu** | §6'daki komut ÇÖKÜYORDU; geometri held-out'ta hiç ölçülemezdi | ✅ `--split`/`--uids_file` |
| F-score'un **iyi** kuyruğu **kötü** etiketiyle | geometri raporu ters okunuyordu | ✅ p10/p90 |
| **Obje-başına veri atılıyordu** | "hangi objeler çöktü" geri gelmezdi | ✅ `per_object` kaydediliyor |
| SSIM iki farklı parametreyle | eğitim eğrisi ↔ kapı raporu 0,053 sistematik fark | ✅ eşitlendi |
| `canonicalize` **gerçek veride bozuk** | ↓ | ✅ ortonormalleştirildi + 3 gerçek-veri testi |
| **Eğitim tekrar üretilemezdi** | iki A/B kolu aynı komutla farklı veri görüyordu | ✅ `set_epoch` + seed |
| Önizlemeler düz klasörde | farklı koşular birbirini eziyordu | ✅ koşu başına klasör |

**`meta.json` extrinsic'leri ortonormal DEĞİL** (12/12 obje, ölçek 0,31–893).
Ölçek eksenler arası **düzgün** olduğu ve ışın yönleri normalize edildiği için
render'lar, öğretmen fitleri ve **bu koşu etkilenmiyor** (bağımsız uzay-oyma
doğrulaması: yeniden izdüşüm IoU **0,990**). Ama şunları **geçersiz kılar**:
`normalize_cams` hakkındaki tüm sonuçlar, cross-attention kararı, ve kol D'nin
çöküşünün `mask_fg_weight`'e atfedilmesi. **`--normalize_cams` ve `--cross_attn`
yeniden ölçülmeden kullanılmamalı.**

**Belge düzeltmesi:** CLAUDE.md'deki *"distilasyon 0,71'de takıldı"* **ölü sayı**
(o koşu %27'de kesilmişti). Gerçek plato `val_rel = 0.3029` (`blok2_distill.log`).

---

## 8. Koşu bitince

```bash
# 1) Sayısal — in-sample ve held-out AYRI (karıştırma!)
python scripts/eval_suite.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --teacher dataset/lrm_ckpts/teacher_1024_tv.pt --n_obj 128 --res 128 --amp \
  --tag TAM_insample
python scripts/eval_suite.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --split test --no_teacher --n_obj 128 --res 128 --amp --tag TAM_test

# 2) Tek/çok görünüm (projenin kuzey yıldızı)
for N in 1 2 4; do
  python scripts/eval_suite.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
    --split test --no_teacher --n_input $N --n_obj 128 --res 128 --amp --tag TAM_test_in$N
done

# 3) Geometri — artık held-out'ta koşabiliyor
python scripts/eval_geometry.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --split test --no_teacher --n_obj 32 --tag TAM_geom

# 4) Tek sayfa rapor
python scripts/kosu_raporu.py --tag TAM_asama3

# 5) GÖRSEL — asıl çıktı bu
python scripts/extract_mesh.py --ckpt dataset/lrm_ckpts_tam/last_TAM_asama3.pt \
  --uid <TEST_UID> --auto_level --grid 256 --smooth 10 \
  --out dataset/mesh_tam --tag TAM
```

`--split test` bilinçli: `val`in ilk 64 objesi koşu boyunca nöbet kararlarında
kullanıldı (karar sızıntısı). 500'lük `test` seti dokunulmamış.
`--smooth 10`: Taubin — alan 64² triplane hücresi altında yapı taşımıyor
(gücün %95'i 3,5 hücre üstünde), yani **artefakt siliniyor, veri değil.**

---

## 9. Açık işler *(bu koşuyu engellemiyor)*

**Faz C öncesi zorunlu:**
1. **Gerçek foto değerlendirme seti YOK.** Tüm sayılar kendi render'ımızda.
2. **Kullanıcı fotosu çerçeveleme** — §6'daki bulgu; `fit` konvansiyonu yeniden üretilmeli.
3. **Mesh eşiği 4 GT silüetinden** seçiliyor; Faz C'de tek maske olacak ⇒ sayılar iyimser.
4. **Çıkarım süresi ölçülmedi** (foto → mesh).

**Sonraki eğitim turu için sıralı öneriler:**
5. **`TriplaneHead`'i OpenLRM sırasına çevir** — şu an `Linear(512→32)` sonra
   deconv; gerçek serbestlik derecesi 64² değil **32²×32**. Distilasyon
   `val_rel 0.303` ile mimari tabanına (**0.177**) doymuş sayılır ⇒ kalan pay
   ağırlıklı olarak burada. Zinciri yeniden kurmayı gerektirir.
6. **`n_samples` 48 → 96.** Işın adımı (0,0292) triplane hücresinin (0,0188)
   1,55 katı — Nyquist ihlali. Ölçülmüş maliyet %8. Referanslar 96/128.
7. **Split'te %3,4 yakın-kopya sızıntısı** — md5 imzası kuantalama sınırında
   körleşiyor. `test` setini kirletiyor. Sürekli betimleyici + `cos>0.97` ile
   yeniden bölme, ~15 dk.
8. **AdamW weight decay** LayerNorm/bias/`tp_tokens`'a da uygulanıyor; referanslar muaf tutar.
9. **`k=1` ağırlıklandırma** — bütçenin %75'i çok-görünümlü, hedef metrik tek görünüm.
10. **Girdi poz çeşitliliği yok**: eğitimde daima elevation +20°, 4 azimuth.
    Kullanıcı 0° veya 45°'den çekerse model bunu hiç görmedi.
11. `near/far/n_samples`'ı `render_cfg` künyesine ekle.
12. `--collapse_after 0` (distile init'ten başlanıyor; ilk 2 saat kör).
13. Antialias'sız downsample + straight-alpha resize — ikisi de mevcut
    performansın altında, 26+ dB'ye çıkınca tavan olur.
