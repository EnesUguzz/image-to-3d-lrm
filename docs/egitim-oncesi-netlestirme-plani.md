# Eğitim öncesi netleştirme planı — 2026-08-24

Amaç: 13k objelik veriyle uzun eğitime girmeden önce **hangi soruların ölçümle
kapandığını**, hangilerinin hâlâ açık olduğunu ve gece hangi koşuların bunları
kapatacağını tek yerde toplamak.

---

## A. KAPANMIŞ sorular (ölçüldü, karar verildi)

| # | Soru | Karar | Kanıt |
|---|---|---|---|
| A1 | "Her girdiye aynı blob" pipeline hatası mı? | Hayır — *ortalama obje* yerel minimumu | oracle 25.04 dB → regresyon 23.94 → sıfırdan 17.06 → distile başlangıç 24.99 |
| A2 | LPIPS ağırlığı | `w_lpips = 0.25` | 32 obje: 2.0 → top-1 %25; 0.25 → %62 |
| A3 | Öğretmen triplane önyüklemesi | **Bırakıldı** — ölçeğe bağlı | obje başına güncelleme = batch÷N; 2635'te 34, işe yaramıyor |
| A4 | `bound=0.6` objeyi kesiyor mu? | Hayır | 40 objede Blender'da ölçülen yarıçap 0.296–0.497 |
| A5 | Objaverse++ filtreleri doğru uygulandı mı? | Evet (bayraklar string, düzeltildi) | 14.594 obje, yeniden sayıldı |
| A6 | `fit` çerçeveleme kenardan taşıyor mu? | Hayır, %0.00 | 2.400 render alfa denetimi |
| A7 | EEVEE_NEXT ≟ CYCLES | Denk | 384 render çifti, PSNR ort 37.4 dB; 5× hız |
| A8 | `min_cov` eşiği | 0.03 → **0.010** | 0.03, 661 sağlam ince objeyi haksız eliyordu (gözle doğrulandı) |
| A9 | Beyaz bg / rastgele bg kompozit | Doğru | %0 obje beyazda kayboluyor |
| A10 | dataset RNG hatası | Düzeltildi + regresyon testi | her epoch aynı seed → 16 render'ın 12'si ölüydü |
| A11 | Aydınlatma referanstan düz, değiştirelim mi? | **Hayır, dokunulmuyor** | `TriplaneNeRF.forward(feats)` bakış yönü almıyor → model bakış-bağımsız görünüm dışında bir şeyi temsil edemez. Alt dolgu ışığı yeni alt görünümler için gerekli. Karanlık veri = bilinen "renk siyaha çökme" tuzağı. Bedeli: zayıf gölge-ile-şekil ipucu (tek-girdi yolunu etkiler). |
| A12 | Kanonik görünümlerin tümü gerekli mi? | Objelerin %30'unda 4 yan görünüm birbirinin neredeyse aynı (MSE<0.01), %6'sı neredeyse özdeş (MSE<0.001) → üst/alt gerçekten bilgi katar | 14.594 obje üzerinde ölçüldü |

---

## B. GECE ÖLÇÜLEN sorular

Zincir: `scripts/overnight.sh` → loglar `dataset/overnight_logs/`

### S1 — Yeni çerçeveleme conditioning'i bozuyor mu? *(yeniden-render kapısı)*
Aynı 32 uid, aynı bayraklar, tek değişken render dizini.
- **A kolu (eski `sphere`, `dataset/renders`)** — koşuldu: `15.80 dB, top-1 %3, mse 0.0325 = ortalama-baseline 0.0325` → **tam çöküş**.
- **B kolu (yeni `fit`+EEVEE)** — gece koşuyor.

> ⚠️ Bu A/B önce **hatalı** kurulmuştu: "yeni" kolu `--render_dir` almamıştı, yani
> eski dizini okuyordu ve A kolunun tekrarıydı. Durduruldu, doğru kuruldu.

### S2 — Alt görünüm gerçekten bilgi katıyor mu? *(sphere20 kapısı)*
`scripts/bench_view_coverage.py` — **yeni**. Encoder/transformer yok; her objeye
kendi serbest triplane'i (oracle). Ölçülen şey saf bilgi meselesi.

Adil kurulum: 24 obje, her iki koşulda da **12 fit görünümü** (sayı eşit),
her ikisi de aynı **held-out alt görünümlerde** (el −25…−70) ölçülür.
- `alt_dahil=0`: 12 görünümün hepsi el ≥ −10 (ring12 kapsaması)
- `alt_dahil=1`: içinde alt kutup (el −88) var

Fark büyükse → 14.594 objeyi sphere20 ile yeniden render etmeye değer (~4–5 saat).
Fark yoksa → mevcut veriyle devam, 5 saat kazanılır.

### S3–S5 — Tarife matrisi (32 obje, 2000 adım, tek değişken)
| etiket | değişken | test ettiği hipotez |
|---|---|---|
| `M_base` | — | referans |
| `M_enc4` | `--unfreeze_last 4` | *"sınır donuk encoder"* — kayıtlı hipotez |
| `M_encall` | `--train_encoder` | LRM/OpenLRM/TripoSR üçü de encoder'ı eğitiyor |
| `M_crop08` | `--crop 0.8` | ön-plana yanlı kırpma (32 objede +0.44 dB ölçülmüştü) |
| `M_batch8` | `--batch 8` | efektif batch |

Sıralama ölçütü: `mse / ortalama-obje-baseline` (düşük = iyi conditioning),
eşitlikte top-1. Kazanan otomatik seçilip S6'ya bayrak olarak geçer.

### S6 — Tam veride ilk dürüst genelleme eğrisi
13.009 train / 1.445 val, 16.000 adım, render 128, sayısal val her 500 adımda.

**Bu güne kadar hiç ölçmediğimiz şey buydu.** `train_lrm.py` sadece görsel
GT|tahmin ızgarası kaydediyordu; genelleme hakkında tek bir sayı üretmiyordu.

---

## C. Gece eklenen kod

| dosya | değişiklik | neden |
|---|---|---|
| `scripts/train_lrm.py` | `build_val_probe` + `val_metrics` | **en kritik boşluk**: sayısal val yoktu |
| " | `--train_encoder`, `--unfreeze_last` | S3'ü ölçekte tekrarlayabilmek için |
| " | `--val_n`, `--max_input` | |
| `scripts/lrm/dataset.py` | `max_input`, `force_n_input` | girdi tavanı `min(4, …)` sabitti; sphere20'de 6 kanonik var |
| `scripts/bench_view_coverage.py` | **yeni** | S2 |
| `scripts/render_object.py` | `canonical_indices` görünüm listesinden türetilir | `[0,1,2,3]` sabitti |
| `scripts/run_batch.py` | `is_done` beklenen görünüm sayısını şemadan alır | sabit 16 → sphere20'de yarım render "tamam" sayılırdı |
| `scripts/overnight.sh` | **yeni** | zincir |

Val metriğinin tasarımı: girdi = kanonik[0] (ön, az 0 el 20), hedef = kanonik[1]
ve [2] (yan + arka). Kanonik açılar tüm objelerde aynı olduğu için metrik objeler
arası karşılaştırılabilir; supervision görünümleri uid-seed'li rastgele olduğu
için onlarla bu yapılamazdı. Bu objeler eğitimde hiç görülmez.

Doğrulandı (duman testi, eğitilmemiş model): `top1 = %12.5 = şans`,
`oran = 1.000` → metrik "ortalama obje" çöküşünü doğru tespit ediyor.

---

## D. AÇIK kalan, gece ölçülmeyen sorular

| # | Soru | Neden geceye alınmadı |
|---|---|---|
| D1 | `normalize_cams` A/B | `bench_overfit` LRMDataset kullanmıyor, ayrı iş. Küçük veride kapalı kalması kayıtlı ölçüme dayanıyor (izole test: mask 400 adımda hâlâ 0.5) |
| D2 | Girdi sayısı dağılımı (1–4 mü 1–6 mı), tek-girdiye ağırlık | S2'nin sonucuna bağlı — sphere20 gelmezse konu yok |
| D3 | Encoder için ayrı (düşük) LR | Matris encoder'ı kazanırsa **sıradaki iş**. Şu an encoder de 4e-4 alıyor; ön-eğitilmiş ViT için yüksek |
| D4 | Triplane hacminin %34 kullanılması | `bound` daraltma denemesi; kapasite kazancı belirsiz |
| D5 | 50k'ya çıkma | Önce 13k'da genelleme eğrisini görmek lazım |
| D6 | Mesh çıkarma / web app | Faz C |

---

## E. Beklenen zaman çizelgesi

| blok | tahmini bitiş |
|---|---|
| 0) Çerçeveleme A/B (B kolu) | ~20:00 |
| 1) Duman testi | ~20:03 |
| 2) sphere20 render (48 obje) | ~20:06 |
| 3) Kapsama testi ×2 | ~21:00 |
| 4) Tarife matrisi (5 koşu) | ~22:40 |
| 5) Kazanan seçimi | anlık |
| 6) Tam veri eğitimi (16k adım) | ~02:20 |

Bir blok patlarsa zincir durmaz (duman testi hariç — o patlarsa son blok atlanır,
4 saat boşa gitmesin diye).

---

## F. Sabah bakılacak yer

```
dataset/overnight_logs/00_zincir.log     <- özet, her bloğun sonucu
dataset/overnight_logs/06_tam_egitim.log <- "[VAL step N] ..." satırları
dataset/lrm_logs/val_metrics.jsonl       <- val eğrisi (makine okunur)
dataset/lrm_bench/COV_alt0.png / COV_alt1.png  <- alt görünüm rekonstrüksiyonu, gözle
dataset/lrm_val_previews/val_*.png       <- val önizlemeleri
```
