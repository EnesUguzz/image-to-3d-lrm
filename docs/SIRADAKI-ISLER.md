# Sıradaki işler — öncelik sırası ve ön-kayıtlı eşikler

*2026-08-29, V3 koşusu başlatıldıktan sonra yazıldı. Bu belge "ne yapalım?"
sorusunun tek cevabı; deney seçimi buradan yapılır ki her seferinde sıfırdan
tartışmayalım.*

## 0. Bu koşunun (V3) cevapladığı ve cevaplamadığı sorular

**Cevaplayacak:**
- Rütbe darboğazı + NeRF derinliği + `n_samples` + görünüm kapsaması düzeltilince
  val PSNR/IoU nereye gider (eski taban: 19,49 dB / IoU 0,648 / top-1 %72).
- Gerçek fotoğrafta tek girdiyle genelleme (taban: `YENI ACI` 17,46 dB / IoU 0,380,
  rakip 22,23 / 0,525 — **kapı KALDI**). → `dataset/lrm_logs/v3_TABAN_gercek_foto.log`
- Dokunulmamış test split'inde (500 obje) ilk gerçek sayı.
- ACIK (in-sample − val) yeni kapasitede ne oluyor.

**Cevaplamayacak (bilerek dışarıda bırakıldı):**
- Encoder'ın tamamını eğitmek fayda eder mi (§1.1)
- Girdi görünüm havuzunun genişliği optimum mu (§1.2)
- `w_lpips` referans değeri 1.0 daha mı iyi (§1.3)
- `lr` efektif batch 16 için doğru mu (§1.4)

## 0.2 ✅ V3 SONUÇ TABLOSU (2026-08-31, tam)

Her iki checkpoint de **adım 30800**, test split'i 128 obje, res 128, `--no_teacher`.
`YENI GORUNUM` = held-out açı (genelleme); `GIRDI GORUNUM` = yeniden üretim.

| ölçüm | TAM_asama3 | **V3** | |
|---|---|---|---|
| val PSNR (kanonik) | 19,49 | 19,38 | ≈ eşit |
| test in1 yeni görünüm PSNR / IoU | 20,43 / 0,662 | 20,22 / 0,652 | ≈ eşit |
| test in2 yeni görünüm PSNR / IoU | 22,54 / 0,766 | 22,27 / 0,754 | ≈ eşit |
| test in4 yeni görünüm PSNR / IoU | 23,27 / 0,794 | 22,71 / 0,772 | hafif ↓ |
| **geometri F@1%** (32 test obj) | 0,260 | **0,298** | **+%15 ↑** |
| geometri F@2% / Chamfer-L1 | 0,526 / 0,056 | 0,541 / 0,055 | hafif ↑ |
| NormCons | 0,684 | 0,677 | ≈ eşit |
| mesh su-geçirmez % | %96,8 | %96,8 | eşit |
| **elevation [55,75) IoU** | 0,419 | **0,518** | **+%24 ↑** |
| gerçek foto in1 YENI ACI IoU | 0,380 | 0,313 | ↓ (**n=1**) |
| 8 train objesinde keskinlik | — | 22,52 dB | §3.5 ↓↓ |

**Okuma:** V3'ün kazancı **görünüm-açısı dayanıklılığı + ince geometri** (F@1%,
elevation); kaybı **keskinlik** (§3.5). Kanonik val ve test PSNR'ı hareketsiz.
Kapı 1 (çöküş) 4/4 geçti, Kapı 2 (kalite) 4/4 kaldı — yani model artık blob
basmıyor ama hedef kalite bandına girmedi. `set_epoch` düzeltmesi (16 görünümün
4'ü yerine 16'sı) beklendiği gibi **açı dayanıklılığına** yaradı, keskinliğe değil.

⚠️ **Gerçek foto düşüşü karar verdirmez** — set n=1 (tek kumanda). §4.

**En iyi checkpoint = `last_V3.pt`.** `kosu_raporu` "adım 24000'de tepe" uyarısı
verdi ama fark ihmal edilebilir: 24k↔30k PSNR 19,433↔19,382 (0,05 dB), IoU
0,6364↔0,6364 (aynı), LPIPS 0,1713↔0,1680 (30k daha iyi). Snapshot'a geçme.

## 0.1 ⚠️ V3 sayıları eski koşuyla NASIL karşılaştırılır

İki metrik ölçek değiştirdi; ham karşılaştırma **yanlış sonuç verir**.

### `val_rel` (distilasyon) — taban kaydı 0,195 → 0,234
Öğretmen keskinleşince (29,33 → 34,08 dB) "hedef gürültü payı" %19,5 → %23,4
çıktı. Bu metrik `var(hedef − 3x3 yumuşatılmış) / var(hedef)`; daha ince yapı
yakalayan öğretmende **tanım gereği** yükselir, bozulma değil. `distill_lrm`'in
%20 uyarısı bu yüzden ilk kez ateşlendi. Öğretmeni `--w_tv` ile yeniden oturtmak
(uyarının önerisi) yanlış olur — öğretmen zincirin tavanı, bilerek bulanıklaştırılmaz.

**Doğru karşılaştırma — tabandan fazlası:**

| | taban | plato | fazla |
|---|---|---|---|
| eski (`distilled_v2`) | 0,195 | 0,3029 | **0,108** |
| V3 eşdeğer kalite | 0,234 | **≈0,342** | 0,108 |

⇒ **V3'ün `val_rel`'i ~0,34 çıkarsa eski koşuyla EŞİT.** 0,34'ün altı gerçek
iyileşme; 0,30'u görüp "kötüleşmiş" demek hatalı olur.
(`rel_smooth` yumuşatılmış hedefe karşı ölçtüğü için gürültü bandını dışlar;
eski plato 0,1721 — bu daha doğrudan karşılaştırılabilir.)

### Öğretmen PSNR — in-sample → held-out
Eski 29,33 dB **eğitildiği 4 donmuş görünümde** ölçülmüştü (`set_epoch` hatası).
Yeni 34,08 dB sabit `EVAL_EPOCH`'ta. Yani gerçek fark gösterilenden **daha büyük**.

## 1. Deney kuyruğu — tek değişkenli A/B'ler

Hepsi V3 checkpoint'inden `--init_from` ile, aynı `bench_uids` kümesiyle,
**eşik koşudan ÖNCE yazılmış olarak**. Sıralama beklenen değer ÷ maliyet.

### 1.1 `--train_encoder` vs `--unfreeze_last 4`  🥇
**Neden ilk:** V3'te ölçülen ACIK küçükse (underfit sürüyorsa) kalan en büyük
donuk kapasite bu. Referansların üçü de (LRM, OpenLRM, TripoSR) encoder'ı eğitiyor.
**Neden V3'e konmadı:** `unfreeze_last 4` son 4 bloğun matrislerini %24 döndürmüş
ve tek başarılı gerçek-foto sonucumuz tam o konfigürasyonda alınmıştı;
`patch_embed` + blok 0–7 domain transferini taşıyor, ölçüm yok.
**Ön-kayıtlı eşik:** kabul için held-out val PSNR **+0,3 dB** VE gerçek-foto
`YENI ACI` IoU'da **düşüş yok**. İkincisi şart — asıl risk orada.
**Maliyet:** ~%2 hız, +0,3 GB VRAM.

### 1.2 `--mixed_p` taraması (0,25 / 0,5 / 0,75)
**Neden:** 0,5 ölçülmeden seçildi. Elevation eğrisi geniş havuzu destekliyor
(IoU 0,619 → 0,489 → 0,419) ama optimumun 0,5 olduğuna dair kanıt yok.
**Eşik:** gerçek-foto `YENI ACI` IoU'da **+0,03**.

### 1.3 `--w_lpips` 0,25 vs 1,0 (referans değeri)
**Neden:** 0,25 bulgusu 32 objede, "ortalama-obje çöküşü" rejiminde ölçüldü.
O rejim distilasyondan başlayınca **artık geçerli değil** — yani gerekçesi düştü.
**Eşik:** LPIPS'te **−0,01** ve PSNR'da **−0,3 dB'den fazla kayıp yok**.

### 1.4 `--lr` 4e-4 vs 2e-4
**Neden:** OpenLRM'den kopyalandı ama onun global batch'i 128 (16×8 GPU),
bizimki 16. sqrt-ölçekleme 1,41e-4 diyor. **Ama** ampirik kanıt düşürmeyi
desteklemiyor: cosine kuyruğu lr'yi 2,2e-4'ten 8,2e-7'ye indirdi, PSNR 0,02 dB
kazandı. Kararsızlık da hiç görülmedi. → **düşük öncelik**, sadece 1.1–1.3
sonuçsuz kalırsa.

### 1.5 `--normalize_cams` (referansta açık)
⚠️ **2. VE 3. aşamada birlikte** açılmalı. Sadece 3. aşamada açmak transformer'ın
Plücker koşullanmasını dağıtım dışına atar ve `--init_from`'u sessizce geçersiz kılar.
Yani ucuz değil: distilasyonu yeniden koşmak gerekir (~4 sa).

## 2. Veri — şimdilik KAPALI, yeniden açma koşulu var

**Karar:** 50k'ya çıkmak şu an gerekmiyor. Gerekçe ölçülmüş:
ACIK (in-sample − val) = +0,57 dB, prob kümesi zorluğu için kalibre edilince
(`taban_psnr` train[:64] 17,68 ↔ val[:64] 17,36) **+0,25 dB**. Model eğitim
verisini bile kuramıyor ⇒ ezberleme yok ⇒ daha fazla veri çare değil.

**Yeniden açma koşulu:** V3 sonrası **ACIK > 1,5 dB**. O zaman gerçekten
overfit'iz ve veri en ucuz çare olur.

**Hazır olan:** diskte kullanılmamış temiz score-3 obje var; score≥2 toplamı
24.571 (9.977'si henüz render edilmemiş, ~1,8 sa). 50k için ~25,5k yeni
indirme (~245 GB) gerekir.

## 3. Faz C — eğitimden bağımsız, şimdi yapılabilir

### 3.1 Doku pişirme (texture baking)  🥇 en ucuz büyük görsel kazanç
Mesh'e vertex rengi yerine **doku atlası** pişir: yüzey noktalarını NeRF'ten
örnekle, UV'ye yaz. Yeniden eğitim gerektirmez, keskinlik algısını doğrudan
artırır. Şu anki bulanıklığın bir kısmı vertex-renk enterpolasyonundan geliyor.

### 3.2 `extract_mesh.py --smooth N` (Taubin, hacim korumalı)
Yüzeydeki "köpük" için. Kaynağı ölçüldü: `nerf.py` konum kodlaması almıyor +
`grid_sample` bilineer 64² ⇒ alan bir hücre altında yapı taşıyamaz (gücün
%99,9'u 0,97 hücre üstünde). **Mimari değişikliği reddedildi** (triplane 128²:
+%5 süre, %95 bandı sadece −%6, ve öğretmen bankasını geçersiz kılar).

### 3.3 🔴 Kullanıcı fotosu SIKI KIRPILAMAZ
Ölçüldü: sıkı kırpma conditioning sinyalini 2,72× artırıyor ama held-out
top-1 %35,9 → %23,4, IoU 0,482 → 0,447. Mekanizma: tam karede objenin
görüntüdeki büyüklüğü *gerçek ölçeğinin ipucu*; kırpma bunu yok ediyor.
Faz C'de `fit` çerçeveleme konvansiyonu **yeniden üretilmeli**
(`TARGET_EXTENT = 0.549`, `scripts/prep_photo.py`).

### 3.4 Faz C'yi 4-foto üzerine kur
Ölçüldü: tek gerçek fotoda sorun domain değil **derinlik önseli**;
2 görünüm in-domain seviyesine çıkarıyor.

## 3.5 🔴 Keskinlik: kaynak distilasyon, 3. aşama DEĞİL

**Ölçüldü (2026-08-29, `scripts/kanit_asama12.py`).** 8 train objesinde
öğretmen 29,93 dB ↔ öğrenci 26,87 dB. Görselde: taneli kaya → düz gri kütle,
bank çıtaları → yok, makas → tanınmaz leke (−6,47 dB).
Objeler eğitim setinde ⇒ **kapasite sınırı, genelleme değil.**

**✅ ÖLÇÜLDÜ (2026-08-31) — 3. aşama bandı GERİ KAZANMIYOR, DAHA DA KAYBEDİYOR.**
Aynı 8 train objesi, aynı öğretmen, her sistem kendi decoder'ıyla
(`kanit_asama12.py --student_nerf`, 3. aşamada NeRF de eğitildiği için zorunlu):

| sistem | PSNR (8 train objesi) | öğretmene fark |
|---|---|---|
| öğretmen (aşama 1) | 29,93 | — |
| distile öğrenci (aşama 2) | **26,87** | −3,06 |
| render ince ayarı (aşama 3, `last_V3`) | **22,52** | **−7,41** |

Kanıt: `dataset/lrm_val_previews/KANIT_asama13_V3.png`. Görselde arı şeritlerini,
makas kollarını, iki ayrı deniz kabuğunu tamamen kaybetmiş; taneli kaya düz
yumurtaya inmiş. En kötü obje −11,68 dB.

**Yorum — bu bir çelişki değil, bir takas.** Aynı koşuda held-out val
16,76 → 19,38 dB yükseldi. 3. aşama, distilasyonun 1.024 objesindeki keskinliği
12.320 objeye yayarak takas ediyor. Yani keskinlik açığı artık **iki ayrı yerde**:
(a) distilasyonda doğuyor (−3,06), (b) 3. aşamada büyüyor (−4,35).
⚠️ Bu, 3. aşamayı kaldırma gerekçesi DEĞİL — held-out'ta ve elevation'da kazandıran
tek aşama o. Kapasite sınırı işareti.

Sıradaki adaylar:

- distilasyon kaybına **yüksek frekans terimi** ekle (şu an düz MSE; yumuşatılmış
  hedefe karşı zaten iyi, kayıp bandı ödüllendirmiyor)
- `TriplaneHead` üst-örnekleme: `ConvTranspose2d(k=2,s=2)` **örtüşmeyen** —
  komşu hücreler arası süreklilik yok. `k=4,s=2` denenebilir (ucuz).
- triplane 128² reddedilmişti (öğretmen bankasını geçersiz kılıyor) ama
  öğretmen zaten yeniden oturtulacaksa maliyeti düşer.

**Not:** `distill_lrm.py` hiç önizleme yazmıyor — bu yüzden sorun 24.000 adım
boyunca görünmedi. Önizleme eklenmeli.

## 4. Ölçüm altyapısındaki kalan boşluklar

- `fit_teacher` / `distill_lrm` **`render_cfg` künyesi yazmıyor** ⇒ üç aşamanın
  `n_samples` / `nerf_layers` tutarlılığını kod değil insan doğruluyor.
  `train_lrm` "KUNYESIZ" uyarısı basıp varsayılanı kabul ediyor. (Düşük risk,
  ama "Kol C" bir kez bu sınıftan hatayla 24 → 16,3 dB vermişti.)
- `[encoder] en cok degisen` log satırı **yanlış tensörü** gösteriyor:
  `blocks.11.ls1.gamma` delta'sı büyük ama yön değişimi 0,0005 (saf weight decay
  büzülmesi); gerçek öğrenme hiç raporlanmayan mlp/attn matrislerinde (%24 dönme).
- Gerçek foto değerlendirme seti **n=1** (kumanda). En az 20 obje gerekir ki
  `YENI ACI` sayısı gürültüden ayrılsın.
- `eval_photo4 --n_input 4` **held-out görünüm bırakmıyor** ⇒ o satırlar
  yeniden üretim kalitesi, genelleme değil. Genelleme için `--n_input 1` veya 2.
