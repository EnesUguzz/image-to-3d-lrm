# TAVANLAR — her aşamada gerçeğe ne kadar yaklaştık (2026-09-03)

Bu belge tek bir soruya cevap verir: **"gerçek" nerede, biz nerede kaldık?**
Her satır fiilen ölçülmüş bir log satırından gelir; kaynak dosya belirtilmiştir.
Ölçülmemiş hücrelere `—` yazılır, tahmin YAZILMAZ.

---

## 0. Gerçek fotoğrafla ne yapıyoruz

**Elimizde tek bir gerçek obje var:** `dataset/gercek_foto/kumanda4/` — bir TV
kumandası, 4 kanonik açıdan (ön/sağ/arka/sol) telefonla çekilmiş.

**Boru hattı:** `prep_photo.py` (arka plan silme + `fit` çerçeveleme + 256px)
→ `infer_photo.py` / `eval_photo4.py` → `extract_mesh.py`.

**Ölçüm fikri:** gerçek fotoda 3B GT yok. Ama aynı obje 4 açıdan çekildiği için
1'ini GİRDİ verip diğer 3'ünü HEDEF yapabiliyoruz.
**RAKİP = "girdiyi kopyala"** (yeni açı için girdi fotosunu aynen basmak).
Derinlikte yayılmış bir blob bu rakibi geçemez ⇒ ayırt edici.

### Gerçek fotoda TOPLAM iki ölçüm var, ikisi de KAPIYI GEÇEMEDİ

| checkpoint | girdi | yeni açı PSNR | RAKİP | yeni açı IoU | RAKİP | kapı |
|---|---|---|---|---|---|---|
| `TAM_asama3` | 1 foto | 17,46 | **22,23** | 0,380 | **0,525** | KALDI |
| `last_V3` (ürün) | 1 foto | 15,98 | **22,23** | 0,313 | **0,525** | KALDI |
| in-domain referans (test seti, 1 görünüm) | | *20,43* | | *0,662* | | |

Kaynak: `dataset/lrm_logs/v3_TABAN_gercek_foto.log`, `v3_asama23.out:303`.

> **Tek cümlelik özet: gerçek bir fotoğrafta, tek girdiyle, modelimiz "girdi
> fotoğrafını yeni açıya aynen kopyalamak"tan DAHA KÖTÜ.** Hem PSNR'da hem IoU'da.

### ⚠️ Ürünün asıl kullanım senaryosu HİÇ ÖLÇÜLMEDİ

`in4` (4 foto girdi) koşusunda **held-out açı kalmıyor** — 4 kanonik açının
hepsi girdi. O yüzden `TABAN_TAM_in4` satırları *yeniden üretim* kalitesidir,
genelleme değil. Yani **"4 gerçek foto → yeni açı"** için elimizde SIFIR sayı var.
Ölçebilmek için ya 5. bir açıdan foto çekilmeli ya da 3 foto girdi + 4.'sü hedef
yapılmalı. (`eval_photo4.py` bunu destekliyor; hiç koşulmadı.)

### ⚠️ Kayıtlarda doğrulanamayan bir sayı

`zincir_v3_asama23.sh:18` ve hafıza notu *"2 görünüm IoU 0,691"* diyor.
**Bu sayı hiçbir logda yok** (`grep -rn "YENI ACI ort"` → yalnız yukarıdaki 2 satır)
ve `0.691` değeri `val_metrics.jsonl`'deki tamamen alakasız bir `oran` metriğiyle
BİREBİR aynı. → **Doğrulanmamış kabul et**, karar dayanağı yapma.

---

## 1. KESKİNLİK MERDİVENİ (asıl şikâyetin konusu)

24 obje, ölçüm 256px, siluet 7px aşındırılmış, `bench_keskinlik.py`.
`kor` = ince bant (~1–3 px doku). `kor_orta` = 3×3−9×9 bandı (~3–9 px = çıta,
ayak, kol ölçeği — kullanıcının "sandalye kalınlaşmış" dediği bant).

| # | sistem | ne olduğu | kor | kor_orta |
|---|---|---|---|---|
| 0 | **GT** | gerçeğin kendisi | **1,000** | **1,000** |
| 1 | GT 128→256 | *denetim 128px olsaydı tavan* | 0,521 | 0,968 |
| 2 | GT 64→256 | *denetim 64px olsaydı tavan* | 0,190 | 0,767 |
| 3 | `TAVAN2_K256` | öğretmen, tp256², bölge denetim, 500 gün/obje | **0,302** | — |
| 4 | `TAVAN2_K128` | öğretmen, tp128², bölge denetim, 500 gün/obje | **0,269** | — |
| 5 | `TAVAN2_K64` | öğretmen, tp64², bölge denetim, 500 gün/obje | 0,158 | **0,418** |
| 6 | `teacher_1024_v4` | öğretmen, tp64², bölge denetim, **100 gün** (üretim bütçesi) | 0,151 | 0,361 |
| 7 | `teacher_1024_v3` | öğretmen, tp64², 64px tam kare, 100 gün — **SEVK EDİLEN** | 0,087 | 0,292 |
| 8 | `distilled_v3` | öğrenci, distilasyon sonrası (in-sample) | 0,027 | 0,199 |
| 9 | `last_V3` | **ÜRÜN** (3. aşama sonrası) | **0,008** | **0,024** |

Yan kollar: `K64_notv` 0,162 / 0,225 · `K64_kutu` 0,168 / 0,225.
Kaynak: `tavan2.out`, `zincir_02eyl.out`, `zincir3_keskinlik.log`, `ogretmen_v4_keskinlik.log`.

### Bu merdivenin okunuşu

- **Gerçeğe en çok yaklaştığımız nokta: `K256`, kor 0,302 = GT'nin %30'u.**
  Ama bu bir MODEL DEĞİL — obje başına serbest parametre fiti, yani *tavan*.
- **Ürün: 0,008 = GT'nin %0,8'i.** Yani ürettiğimiz doku, gerçek dokuyla
  pratikte korelasyonsuz. Görselde bu, kayanın üzerindeki kafes deseni.
- **Kayıp öğretmenden SONRA:** 0,292 → 0,199 → 0,024 (`kor_orta`).
  Eşleşmiş fark: distilasyon −0,093±0,078*, 3. aşama −0,268±0,092*.
- **Öğretmen bile kendi tavanının yarısında:** v4 0,361 ↔ 64px tavanı 0,767 (%47).

---

## 2. PSNR MERDİVENİ

⚠️ **PSNR'lar ölçüm çözünürlüğü aynı olmadan karşılaştırılamaz.**

| sistem | ölçüm | küme | PSNR |
|---|---|---|---|
| `teacher_1024_v3` | res **64** | in-sample | 34,08 dB |
| `teacher_1024_v4` | res **256** | in-sample | **32,03 dB** |
| öğretmen (`kanit_asama12`) | res 256, 8 train objesi | in-sample | 29,93 |
| `distilled_v3` | ″ aynı 8 obje | in-sample | 26,87 (−3,06) |
| `last_V3` 3. aşama | ″ aynı 8 obje | in-sample | 22,52 (−7,41) |
| `last_V3` test seti, **1 girdi**, yeni görünüm | res 128 | held-out | 20,22 |
| `last_V3` test seti, **2 girdi**, yeni görünüm | res 128 | held-out | 22,27 |
| `last_V3` test seti, **4 girdi**, yeni görünüm | res 128 | held-out | 22,71 |
| `last_V3` **GERÇEK FOTO**, 1 girdi, yeni açı | res 256 | held-out | **15,98** |

> v3'ün 34,08 dB'si v4'ün 32,03'ünden **iyi değil**: v3 64 piksellik bir hedefe
> karşı ölçüldü (16× daha az piksel), v4 256'ya karşı. Aynı sütunda değiller.

---

## 3. GEOMETRİ MERDİVENİ (`last_V3`, test seti)

| metrik | değer | not |
|---|---|---|
| GT hizalama doğrulaması (siluet IoU) | 0,903 | 1,0'a yakın olmalı ✓ |
| **F@1%** | **0,298** | p10 0,117 |
| F@2% | 0,541 | |
| F@5% | 0,845 | |
| Chamfer-L1 | 0,055 | p90 0,097 |
| NormalConsistency | 0,677 | |
| mesh hacim IoU (marching cubes) | **0,397** | tek bileşen, watertight |
| topoloji | %96,8 kapalı, Euler ort −1,9 | tünel dolduruyor |
| maske IoU, yeni görünüm (in1/in2/in4) | 0,652 / 0,754 / 0,772 | |

Kaynak: `eksik_evals_v3.out`, `geom_V3_geom.json`.

---

## 4. ⛔ DÜZELTME: "tp128 gündemden düştü" kararı DESTEKSİZ

`CLAUDE.md` (2026-09-03, ZİNCİR3 bölümü) şöyle diyor:

> *"tp128 GÜNDEMDEN DÜŞTÜ. Zincir kaybı (`kor_orta` −0,268) tp64→tp128'in
> vereceği kazançtan (+0,178) BÜYÜK."*

**İki hata var:**

1. **`+0,178` sayısı hiçbir logda yok.** `grep -rn kor_orta dataset/lrm_logs/`
   → `kor_orta` sütunu 2026-09-03'te eklendi; `TAVAN2_K128` ve `K256` ondan ÖNCE
   ölçüldü. **K128'in `kor_orta` değeri HİÇ ÖLÇÜLMEDİ.**
2. **Karşılaştırma iki FARKLI SÜTUNU yan yana koyuyor:** kayıp `kor_orta`'dan,
   kazanç `kor`'dan. `docs/KESKINLIK-TESHISI.md §8.3` bunu zaten yasaklıyor.

**K128'in ölçülmüş kazancı gerçek ve büyük:** `kor` 0,158 → **0,269 (+%70)**,
enerji %38 → %56, ve `TAVAN2_dalga1_kare.png` görselinde gözle görülür.
K256 daha da yukarıda (0,302) ama 128→256 kazancı sadece +0,033 ⇒ **doğru
duraklama noktası tp128.**

**Kararın hâlâ ayakta kalan kısmı:** öğretmen ne kazanırsa kazansın, mevcut
öğrenci onu atıyor (0,292 → 0,024). Bu, *"tp128 değersiz"* demek DEĞİL;
*"önce boruyu tamir et, sonra daha iyi su bas"* demek. Sıralama sorusu,
iptal sorusu değil.

---

## 5. Bir haftada fiilen KAZANILAN şeyler (yarın da doğru olacaklar)

| # | kazanç | kanıt |
|---|---|---|
| 1 | Blob çöküşü çözüldü | top-1 %1,6 (şans) → %92,2 |
| 2 | Ölü render seti tespit edildi + 6 scriptin varsayılanı düzeltildi | `AB_ESKI` %3 ↔ `AB_YENI` %75 |
| 3 | Küçültme defektleri (antialias yok + premultiply ters + kesirli kutu) | sahte HF %59→%37, kor+kaydır +%36 |
| 4 | `w_mask` iç optimumu 0,25 | tek o kol `taban`ı ve `rakip`i geçti |
| 5 | **Bölge kırpma denetimi** öğretmeni iyileştiriyor | v3 0,292 → v4 0,361 `kor_orta` |
| 6 | **Kafes artefaktının kök nedeni + doğrulanmış çözümü** | `blok_sinir` 3,96/3,99 → **0,77** |
| 7 | İlk dürüst genelleme sayısı | `heldout_rel` **0,935** (in-sample 0,377) |

## 6. Neden "dönüp duruyoruz" hissi doğru

**Yukarıdaki 7 kazancın 5'i ÖĞRETMENİ ölçtü.** Oysa ZİNCİR3'ün gösterdiği şey
kaybın öğretmenden SONRA olduğu. Yani bir hafta boyunca, %92'sini sızdıran bir
borunun GİRİŞİNİ cilaladık.

**Sıradaki iş bir test değil, bir İNŞA:** v4 denetimi + k4s2 head + tp128 tek
bir zincirde birleştirilip uçtan uca koşulmalı, ve sonuç **gerçek fotoğrafta**
(3 foto girdi + 4.'sü held-out) ölçülmeli — çünkü ürünün gerçek sayısı orada,
ve orada bugün "girdiyi kopyala"nın altındayız.
