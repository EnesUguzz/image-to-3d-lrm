# Keskinlik problemi — teşhis ve plan

**Tarih:** 2026-09-02 · **Yöntem:** systematic-debugging (Faz 1 kanıt → Faz 2 örüntü → Faz 3 hipotez)
**Soru:** "Öğretmende bile keskin görüntü alamıyoruz. Neden?"

> Bu belge keskinlik konusunda **tek referanstır**. Yeni bir keskinlik deneyi
> önermeden önce §6'daki ön-kayıtlı kapılara bak: cevabı zaten yazılı olan bir
> soruyu tekrar deneme.

---

## 1. Tek cümlelik cevap

Keskinlik bir **tarife sorunu değil, bir çözünürlük bütçesi sorunu.**
Boru hattında objenin kaç "örnek" genişliğinde temsil edildiğini aşama aşama
ölçtüğümüzde üç ayrı darboğaz çıkıyor ve **üretim tarifesinde üçü de aynı anda
bağlayıcı.** Bugüne kadarki her deney bunlardan yalnız birini gevşetti, diğer
ikisi bağlayıcı kaldı, sonuç kıpırdamadı — deneylerin "rastgele" hissettirmesinin
teknik sebebi bu.

Ayrıca veri yolunda, tam da keskinliğin yaşadığı bandı bozan **iki gerçek defekt**
vardı (§3, düzeltildi). Bu yüzden bugüne kadarki keskinlik ölçümlerinin çoğu
temsilin tavanını değil **bozuk hedefi** ölçüyordu.

---

## 2. Kök neden: çözünürlük bütçesi

Tek bir soruyu her aşamada sorduk: **obje bu aşamada kaç örnek genişliğinde?**
(100 train objesi, kanonik görünüm; objenin en uzun kenarı karenin **%54**'ü,
medyan %53,9, p90 %70,7.)

| aşama | obje genişliği | bağlayıcı mı? |
|---|---|---|
| diskteki master render (512px) | **276 px** | — (kaynak) |
| **encoder girdisi 224px / DINOv2 patch 14** | **8,6 patch** | ⚠️ **§8.1'e bak** |
| transformer token ızgarası 32² | 24 token | sarı |
| **triplane 64² (küp 1,2 dünya birimi)** | **48 hücre** | 🔴 öğretmen için |
| **üretim denetimi 64px tam kare** | **35 px** | 🔴 öğretmen için |
| denetim 128px | 69 px | |
| denetim 256px | 138 px | |
| ölçüm (`bench_keskinlik`) 256px | 138 px | — (hedef) |

Okunuşu:

- **Üretim öğretmeni objeyi 35 piksel genişliğinde gördü.** Keskinliği 138
  piksellik bir GT'ye karşı ölçüyoruz. Öğretmenin fit ettiği görüntüde makasın
  kolu ya da bankın çıtası **hiç yok** — kaybolmuş değil, hiç verilmemiş.
- Denetim (35) triplane'den (48) **daha kaba.** Yani üretim tarifesinde temsil,
  denetimin taşıyabileceğinden ince; fazla serbestlik gürültüyle doluyor.
  `last_V3`'ün "GT'nin %31'i kadar HF enerjisi ama korelasyon 0,007" tablosu
  (= kayanın üstündeki gözle görülür kafes deseni) tam olarak budur.
- Denetim 256'ya çıkarılsa bile **triplane 48 hücrede kalır:** 138 piksellik bir
  GT'yi 48 örnekle temsil etmeye çalışırsın. Öğretmenin tavanı buradadır.
- ⛔ **BU MADDE ÇÜRÜTÜLDÜ — §8.1'e bak.** "Encoder objeyi 8,6 patch görüyor,
  öğrenci duvarı budur" iddiası bir **hesaptı, ölçüm değil**; üç ayrı ölçüm ona
  karşı. Öğrencinin hatası bant sınırlı detay kaybı değil, *yanlış düşük frekans*
  (makas → blok, arı → yumru). Öğrenci tarafındaki dar boğaz **hâlâ bilinmiyor**;
  onu belirleyecek deney §8.6/1 (oracle koşullandırma).

### Ölçülmüş destek: bulanıklık *yanal*, derinlikten değil

Yeni teşhis (bugün, obje içi ön-plan ışınlarında ağırlık dağılımı):

| sistem | acc (obje içi) | yüzey kalınlığı (dünya) | = hücre | = px@256 |
|---|---|---|---|---|
| tvA — denetim 256px, 1000 güncelleme, tp64 | 0,997 | 0,0088 | 0,47 | **1,35** |
| tvC — denetim 256px, 1000 güncelleme, tp128 | 0,997 | 0,0081 | 0,86 | 1,24 |
| DET_A — denetim 64px, 100 güncelleme, tp64 | 0,988 | 0,0237 | 1,27 | **3,64** |
| DET_C — bölge128, 100 güncelleme, tp128 | 0,995 | 0,0182 | 1,94 | 2,79 |

Yeterli denetim verildiğinde yüzey **tam opak** (acc 0,997) ve derinlikte
**1,35 piksel** kalınlığında — yani keskin. "Sisli yüzey" bir kök neden değil,
denetim yoksunluğunun **belirtisi**. Bu, koca bir hipotez dalını kapatıyor.

---

## 3. Veri yolunda iki gerçek defekt — ölçüldü, DÜZELTİLDİ

`LRMDataset._load_rgba` 512'lik master'ı denetim çözünürlüğüne indirirken iki
hata yapıyordu. İkisi de yalnızca **yüksek frekans bandını** bozuyor: PSNR'de
neredeyse görünmez, keskinlikte her şey.

### 3a. Anti-aliasing yoktu

`F.interpolate(mode="bilinear")` PyTorch'ta varsayılan `antialias=False` ile
gelir: küçültmede çekirdek kaynak ölçeğine genişletilmez, **8× küçültmede 8×8
bloğun sadece 2×2'si örneklenir.** Ölçüldü (12 gerçek render):

| denetim res | ölçek | \|bilinear − area\| RMS | görüntünün TÜM HF enerjisinin |
|---|---|---|---|
| 64 | 8,0× | 0,0209 | **%72** |
| 128 | 4,0× | 0,0101 | %47 |
| 192 | 2,7× | 0,0115 | %73 |
| **256** | **2,0×** | **0,0000** | **%0** |
| 384 | 1,3× | 0,0081 | %72 |

Bu takma ad gürültüsü **poza bağlıdır** (alt-piksel fazı her görünümde farklı)
⇒ **görünümler arası tutarsız.** Hiçbir 3B temsil tutarsız bir sinyali fit
edemez; optimize edicinin verebileceği en iyi cevap **ortalamasıdır — yani
bulanık doku.**

⚠️ **Ölçümü gizleyen tesadüf:** tam 2× küçültmede (512→256) bilinear zaten kutu
ortalamasına eşittir. `bench_keskinlik` ölçümü **hep 256'da** yapılıyor ⇒ GT
temizdi, eğitim hedefi bozuktu. Hiçbir metrik bunu göremezdi.

### 3b. Premultiply sırası tersti

RGBA **düz (straight)** halde küçültülüyor, alpha ile çarpma sonra geliyordu.
Blender alpha=0 pikseline RGB=0 yazar (ölçüldü: 0,0028) ⇒ siluet kenarına siyah
sızar. Kompozitleme doğrusaldır; ortalama **premultiply uzayında** alınmalıdır.

| denetim res | kenar pikselleri RMS hata | obje içi |
|---|---|---|
| 64 | **0,1099** | 0,0264 |
| 128 | 0,0672 | 0,0140 |
| 256 | 0,0058 | 0,0001 |

res 64'te obje **35 piksel** genişliğindedir; 1–2 piksellik bir bank çıtası ya da
makas kolu **tamamen kenar pikselidir** ⇒ rengi siyaha çekilir, alpha'sı ezilir.
"İnce yapılar kayboluyor / kalınlaşıyor" tespitinin doğrudan mekanizması budur.

### Düzeltme

premultiply → **alan (box) ortalaması** → düz renge geri dön.
`_load_rgba` ve `_crop_input` (dormant ama aynı hata — bu projede "düzeltmeyi
tek çağrı noktasına uygulama" hatası daha önce `set_epoch`'ta yaşandı).
`tests/test_kucultme_dogrulugu.py`, 7 test. **Testler 197 → 204.**
Fit hızına maliyeti **%2,5**.

---

## 4. ⛔ Hangi eski ölçümler geçersiz

Defekt her denetim çözünürlüğünü **farklı oranda** bozuyor (res 64'te en çok).
Dolayısıyla **denetim çözünürlüğünü değiştiren her A/B'de defekt de birlikte
değişti** — tek değişken kuralı sessizce ihlal edildi.

| ölçüm | durum |
|---|---|
| `den_res` merdiveni (64→0,088 / 128→0,135 / 256→0,199) | ⛔ **karışık** — kazancın ne kadarı detay, ne kadarı "daha az takma ad" bilinmiyor |
| `bench_detay` A/B/C (0,078 / 0,147 / 0,173) | ⛔ üçü de bozuk hedefte eğitildi |
| `kesk_2x2` 2×2 taraması | ⛔ aynı sebep + yalnız enerji ölçüyordu |
| `TAVAN` probu (tvA/tvB/tvC, denetim 256px) | ⚠️ **en az etkilenen** (512→256 tam 2×) ama **n=4** |
| `teacher_1024_v3`, `distilled_v3`, `last_V3` | ⛔ hepsi res 64 denetimle eğitildi |

Yani **temsilin tavanına dair güvenilir tek verimiz `TAVAN` probu ve o da 4
objeyle alınmış.** Projenin kendi kuralı ≥24 obje. Üstelik `den_res` ile `TAVAN`
birbiriyle çelişiyor (`den_res`: tp128 < tp64; `TAVAN`: tp128 > tp64) ve
`den_res`'in tp128 kolu ayrıca **Nyquist ihlalliydi** (fit `n_samples 96`,
tp128 en az 150 ister).

**"Deneyler rastgele oluyor" hissinin teknik açıklaması budur:** birbiriyle
çelişen, düşük güçlü ve ortak bir defektle kirlenmiş ölçümler üzerine karar
verilmeye çalışıldı.

---

## 5. Ölçülüp ELENENLER — bunları tekrar deneme

| aday | eleyen kanıt |
|---|---|
| yüzey derinlikte yayılmış ("sis") | 256px denetimde acc 0,997, kalınlık 1,35 px ⇒ belirti, sebep değil |
| ~~NeRF MLP derinliği/kapasitesi~~ ⛔ **ELEME GEÇERSİZ, §8.6 sonu** | `kap_mlp` `kor`'u 0,010→0,085 (8,5×) oynatmış; eleme `kor+kaydır`/PSNR'a bakıyordu |
| renderer / kamera / sayısal hassasiyet | izole test: bf16 bağıl hata 0,0003; `n_samples` Nyquist'e bağlandı |
| veri kalitesi / miktarı | denetim raporu: taşma 0, bozuk 0; ACIK (in-sample − val) +0,25 dB ⇒ ezber yok |
| "triplane 128² fayda etmiyor" | ⛔ o sonuç `--res 64` ile alınmıştı (hücre 0,88 px = piksel-altı) — geçersizdi |

---

## 6. Plan

### Aşama 0 — veri yolu defektleri ✅ BİTTİ
Düzeltildi + 7 regresyon testi. **Bundan önceki hiçbir keskinlik sayısı
yenileriyle kıyaslanamaz.**

### Aşama 1 — düzeltmenin etkisi (tek değişkenli A/B) ✅ BİTTİ → §7

### Aşama 2 — ÖĞRETMEN TAVANI (tek, kesin deney) ⏳
Bugüne kadarki tüm çelişkili ölçümlerin yerine geçer.

- **Kurulum:** 24 obje (`train_list_v2` ilk 24 — `bench_keskinlik` bu sırayı şart
  koşuyor), obje başına **500 güncelleme**, `batch 2` ⇒ 6000 adım.
- **Denetim BAĞLAYICI DEĞİL:** `--region 128 --render_low 256 --render_high 512`.
  Işın maliyeti 128²'de sabit, detay tavanı 512 piksel, her kolda **aynı**.
- **Tek değişken = `tp_res`.** Kanal 32'de SABİT — `TAVAN` probunda tp128 ile
  kanal birlikte değişmişti (confound). `n_samples` Nyquist'ten zorunlu.

  | kol | tp_res | kanal | n_samples | w_tv |
  |---|---|---|---|---|
  | `K64` | 64 | 32 | 96 | 0,05 |
  | `K128` | 128 | 32 | 192 | 0,10 |
  | `K256` | 256 | 32 | 320 | 0,20 |
  | `K64_notv` | 64 | 32 | 96 | **0** |

  `w_tv` dünya-ölçeğinde sabit tutulacak şekilde `tp_res` ile ölçekleniyor
  (TV `mean|Δ|`; komşu farkı ∝ 1/res).
  `K64_notv` şart: `TAVAN` probunda yalnız `w_tv` değişince kor+kaydır
  0,230 ↔ 0,138 oynadı (n=4). Bu ya gerçek ya gürültü; bilmeden `tp_res` okunamaz.
- **Ölçüm:** `bench_keskinlik --n_obj 24 --res 256` + yüzey teşhisi + görsel.
- **Süre:** ~8 saat (K256 baskın). Gece koşusu.

#### ⚠️ Peşinen yazılan confound
`K256`'nın obje başına parametresi `K64`'ünkinin 16 katı, güncelleme sayısı aynı.
Kaybederse "kapasite yetmedi" değil **"bütçe yetmedi"** olabilir. Bu yüzden PSNR
eğrisinin son 1000 adımdaki eğimi de raporlanacak: `K64` düzken `K256` hâlâ
tırmanıyorsa sonuç eksik eğitimdir, tavan değil.

#### Ön-kayıtlı kapılar (`kor+kaydır`, 256px ölçüm)
Referanslar: **GT 128→256 = 0,520** ("128 piksellik görüntü kadar keskin"),
GT 64→256 = 0,231. Bugünkü üretim öğretmeni (`teacher_1024_v3`) = 0,145.

| kapı | koşul | SONRAKİ İŞ (şimdiden karar verildi) |
|---|---|---|
| **A** | herhangi bir kol **≥ 0,35** | Temsil suçlu değil. O kolu benimse, öğretmen bankasını o tarifeyle kur, öğrenci tarafına geç. **Başka mimari deneyi yok.** |
| **B** | `tp_res` ikiye katlandıkça **≥ +0,05** ve monoton | Çözünürlük merdiveni gerçek → Aşama 3. |
| **C** | tüm kollar **< 0,25** ve `tp_res`'te düz | Duvar temsilde değil. Hacim render'ı/MLP tarafına dön, **çözünürlük yolunu kapat.** |

### Aşama 3 — (yalnız kapı B geçerse) ölçeği taşıyacak altyapı
1. **Öğretmen bankası CPU'da sayfalanmalı.** `fit_teacher.py:65` tüm triplane'leri
   tek GPU tensöründe tutuyor: tp128 → 1024 obje 19,3 GB (sığmaz). Adımda yalnız
   `batch` obje gerekiyor; PCIe ~10 ms, adım ~600 ms ⇒ %2 maliyet.
   **Bu fizik değil, kod.**
2. **`TriplaneHead` üst-örnekleme.** Öğrenci 32² token ızgarasından `k=2,s=2`
   deconv ile 64² üretiyor. 128²/256² için **örtüşmeli progresif** yükseltme şart:
   `k=s` deconv örtüşmesizdir, her token kendi bloğunu bağımsız üretir ⇒ blok
   başına etkin serbestlik token boyutuyla sınırlı. (Bu, `TriplaneHead`
   docstring'indeki rütbe darboğazının ta kendisi, sadece daha büyüğü.)
3. Zinciri (fit → distill → render ince ayarı) yeni bütçeyle yeniden koş.

### Aşama 4 — ÖĞRENCİ DUVARI (ayrı problem, ayrı deney)
Öğretmen keskinleşse bile öğrenci onu **kopyalayamaz**: objeyi 8,6 DINOv2 patch'i
olarak görüyor.

- Aday: `INPUT_RES` 224 → 448 ⇒ obje 8,6 → **17,3 patch**. Encoder 4× FLOP;
  gövde `joint` ise token 4096 → 7168 (~3× dikkat), `cross_attn` ise doğrusal.
- ⛔ **Girdiyi sıkı kırpma denendi ve REDDEDİLDİ** (held-out top-1 %35,9→%23,4):
  objenin görüntüdeki büyüklüğü gerçek ölçeğin ipucu. Tekrar açma.
- **Ön koşul:** önce öğretmen keskinleşmeli; yoksa öğrencinin kopyalayacağı
  keskin hedef yok. Sıra: Aşama 2 → 3 → 4.

---

## 7. Sonuçlar

### Aşama 1 — küçültme düzeltmesinin etkisi (2026-09-02)

Üretim öğretmen tarifesinin **birebir aynı komutu**, tek fark kod düzeltmesi.
32 obje, 1600 adım, batch 2, n_sup 2, res 64, tp64×32, w_tv 0,05.
Ölçüm tek çağrıda ⇒ ikisi de **aynı GT'ye** karşı. Betik:
`scripts/ab_kucultme_duzeltmesi.sh`, log: `dataset/lrm_logs/ab_kucultme.out`.

| kol | HF enerji | GT oranı | kor | **kor+kaydır** | eval PSNR |
|---|---|---|---|---|---|
| `DET_A_taban` (hatalı veri yolu) | 0,01149 | %59 | 0,006 | 0,078 | 26,37 dB |
| `DET_A_duzeltilmis` (düzeltilmiş) | 0,00711 | **%37** | −0,001 | **0,106** | **27,12 dB** |

**Okunuşu:**
- **Sahte enerji üçte bir azaldı** (%59 → %37). Kaybolan şey doku değil, takma ad
  gürültüsünün ürettiği kafes deseniydi — `bench_keskinlik`'in "enerji tek başına
  yanıltır" uyarısının somut örneği.
- **kor+kaydır +%36** (0,078 → 0,106) ve **PSNR +0,75 dB**. Düzeltme gerçek ve
  bedavaya geliyor.
- **Ama keskin değil ve olamazdı:** bu tarifede denetim objeyi 35 piksel
  gösteriyor. **Bozuk hedefi düzeltmek, hiç verilmemiş detayı yaratmaz.**
  Bu sonuç §2'deki bütçe tablosunun doğrudan doğrulamasıdır.
- `kor` (kaydırmasız) her iki kolda ≈ 0: bu tarifede yüzey 1–3 piksel yanlış
  yerde (yüzey kalınlığı 3,64 px ile tutarlı).

Görsel: `dataset/lrm_bench/ABFIX_kare.png`

### Aşama 2 — öğretmen tavanı
*(koşulacak)*

---

## 8. Bağımsız denetim + kod incelemesi (2026-09-02)

İki bağımsız inceleme koşturuldu: (a) `code-review` skill'i diff üzerinde,
(b) soğuk başlangıçlı bir denetçi agent projenin tamamı üzerinde. **İkisi de
teşhisin bir bölümünü doğruladı, bir bölümünü çürüttü.**

### 8.1 ⛔ TEŞHİSTE DÜZELTME: "encoder duvarı" iddiası KANITSIZDI

§2'de encoder'ın 8,6 patch'i **🔴 öğrenci duvarı** diye işaretlenmişti.
Bu bir **hesap**, ölçüm değil — tek değişkenli hiçbir deneyle sınanmadı.
Delil olarak "öğretmen kor 0,075 ↔ öğrenci 0,007" gösterilmişti; bu geçersiz,
çünkü aradaki farkta **iki tam eğitim aşaması** var ve ikisinin kaybı ayrı ayrı
ölçülmüş (distilasyonda −3,06 dB, 3. aşamada −4,35 dB).

Üç ölçüm bu iddiaya **karşı**:

1. `kuyruk_31agu.out`: `--unfreeze_last 4` (20,37 dB / IoU 0,656) ↔
   `--train_encoder`, tüm ViT açık (20,52 / 0,661). Fark +0,15 dB, belirsizlik
   bandı ±0,67. *(Uyarı: bu deney encoder'ı EĞİTMEYİ ölçtü, patch sayısını
   değil — 224→448 sorusunu doğrudan çürütmez.)*
2. `--input_crop` patch yoğunluğunu 2,7× artırdı; held-out top-1 %35,9 → %23,4.
3. **En güçlüsü:** `KANIT_asama13_V3.png`'de öğrencinin hatası bant sınırlı detay
   kaybı DEĞİL. Bulanık bir arı hâlâ sarıdır ve şerit izi taşır; bulanık bir
   makas hâlâ iki ince çubuktur. Öğrencide arı **soluk bej yumru**, makas
   **tanınmaz blok**, iki deniz kabuğu **tek yumurta** — bu *yanlış düşük
   frekans*, yani temsil kurma başarısızlığı. Çözünürlük bütçesi bunu açıklamaz.

**Düzeltilmiş ifade:** encoder patch sayısı bir *bütçe rakamıdır*, ölçülmüş bir
duvar değildir. Öğretmen tarafındaki iki duvar (denetim, triplane) ölçülmüştür;
**öğrenci tarafındaki darboğaz hâlâ bilinmiyor.**

### 8.2 🔴 AŞAMA 3, YAZILDIĞI HÂLİYLE ÇALIŞAMAZ

Öğrencinin ürettiği triplane, transformer'ın `tp_tokens` ızgarasının
**deterministik** bir fonksiyonu (`transformer.py:78` → `3·32²×512 = 1.572.864`
sayı; `model.py:32` `TriplaneHead`'e `upsample` **hiç geçmiyor** ⇒ daima ×2 ⇒ 64²).

| şey | eleman sayısı |
|---|---|
| öğrenci gizil hâli (3×32²×512) | **1.572.864** |
| öğretmen tp64²×32 | 393.216 |
| öğretmen tp128²×32 | **1.572.864** ← tam doygunluk |
| öğretmen tp128²×64 | 3.145.728 ← ancak yarısı taşınabilir |

⇒ `tp_res`'i öğretmende büyütmek, **token ızgarası sabitken** öğrenciye bir bit
bilgi eklemez; aynı bilgiyi 4× ince ızgaraya yayar. Üstelik `distill_lrm.py`'de
`--tp_res` bayrağı **yok**; bir tp128 öğretmeni yüklenirse şekil hatası verir.

Bu, §6 Aşama 3'ün 2. maddesinin teşhisini de düzeltir: sorun "örtüşme/süreklilik"
değil, **serbestlik derecesi**. `k=4,s=2` örtüşmeli deconv da 32² token'dan türetir.
Referans ailenin kendi merdiveni bunu zaten söylüyor: OpenLRM small→base→large
geçişinde `triplane token 32→48→80` ve `girdi 224→336→448` **birlikte** büyüyor;
çıkış triplane çözünürlüğünü tek başına büyütmek referansta **hiç kullanılmayan**
bir eksen.

### 8.3 ⚠️ `kor` metriği büyük ölçüde bir `tp_res` detektörü

`hf_vek` 3×3 kutu-bulanıklık kalıntısı ⇒ 256px'te fiilen **1,5–3 piksel** periyotlu
yapıyı ölçer. Hücre boyutları: tp64 → 2,9 px, tp128 → 1,44 px. Yani tp64'ün
temsil edebildiği en ince yapı ölçüm bandının **dışında**, tp128'inki içinde.
Referans çizgileri bunu doğruluyor — neredeyse mükemmel bir çözünürlük cetveli:
hücre 2 px → 0,521 · 2,9 px → 0,158 · 4 px → 0,190 · 1,44 px → 0,269.

Metrik bozuk değil, ölçtüğünü doğru ölçüyor. Ama **"hangi darboğaz bağlayıcı"
sorusunu bu metrikle sormak, cevabı soruya gömmektir.** Izgaranın *taşıyabildiği*
bantta (5–20 px yapı) ne kadar doğru olduğumuzu ölçen hiçbir sayı yok — ve
öğrencinin çöküşü tam orada.

→ **Yapılacak:** `hf_vek` tek 3×3 yerine **çok ölçekli bant geçiren** olsun
(Laplacian piramidi: 2–4, 4–8, 8–16, 16–32 px bantları, her biri için ayrı `kor`).
"Kayıp hangi oktavda başlıyor" sorusunu ilk kez cevaplar.

Ayrıca `kor+kaydır` yanlılığı **sinyal zayıfladıkça büyüyor** — mükemmel hizalı GT
referanslarından ölçüldü: 128→256 için +0,000, 64→256 için +0,045.
⇒ Birincil karar `kor` üzerinden verilmeli; bugünkü `FAYDALI` sütunu en yanlı
sayıyı en görünür yere koyuyor.

### 8.4 Düzeltilen kod defektleri

| # | defekt | etki |
|---|---|---|
| B1 | `compat.load_lrm` ckpt render künyesini okumuyordu | 4 eval aracı `density_bias`/`bound`/`n_samples`'ı **sessizce** yanlış kurabilirdi |
| B1b | künye okunsa bile `nerf_layers != 4` dalında **atılıyordu** (kendi düzeltmemdeki hata) | eski 2-katmanlı ckpt'lerde çağırana yüklenenin TERSİ raporlanırdı |
| B2 | `distill_lrm` künye yazmıyordu | 3. aşamanın uyuşmazlık kapısı varsayıma düşüyordu |
| B3 | `bench_overfit` / `bench_triplane_fit` `--w_lpips` **2.0** | ana kapı aracı, ölçülerek terk edilmiş tarifeyi ölçüyordu |
| **C1** | **küçültme düzeltmesi 4 çağrı noktasına taşınmamıştı** (`eval_suite:68`, `bench_overfit:37`, `diag_retrieval:22`, `metrics:148`) | **birincil kapı aracı bozuk GT'ye karşı ölçüyordu** — aynı sınıf hatanın 3. tekrarı |
| C2 | `train_lrm` künyeyi yazıp **karşılaştırmıyordu** (`_cur` sadece 2 alan) | 48-örnekli ckpt 96 ile sessizce devam ederdi |
| C3 | `volume_render_importance` `chunk` almıyordu | `--n_fine` + `--ray_chunk` birlikte → OOM, uyarı yok |
| C4 | `kanit_asama12` öğretmen decoder'ını **sabit** kuruyordu | GT&#124;öğretmen&#124;öğrenci kanıt görseli sessizce geçersiz olabilirdi |
| C5 | `extract_mesh.photo_iou` paydası daima 4 | 2 fotoluk girdide gerçek 0,60 IoU rapora **0,30** diye giriyordu |
| C6 | `prep_photo` düz RGBA üzerinde LANCZOS | Faz C girdi yolunda **renkli** siluet halesi = dağılım uyuşmazlığı |
| C7 | 3 scriptte `INPUT_RES` yerine sabit 224 | 224→448 planı sessizce kırılırdı |
| C8 | `kosu_raporu` ölçülmemiş metriği `9.0000 ❌ KALDI` diye basıyordu | ölçülmemiş şeyi başarısızlık gibi okutuyordu |

**Tekrar engeli:** `lrm/imutil.py` tek kaynak + `tests/test_kucultme_dogrulugu.py`
içinde **`ast` denetimi** — `scripts/` altında anti-alias'sız bilinear/bicubic
`F.interpolate` yasak (gerçek büyütme `# OLCEK-DENETIMI: buyutme` ile işaretlenir).
Test, yazıldığı anda gözden kaçan bir tane daha yakaladı (`eval_suite:145`).
**Testler 204 → 213.**

### 8.5 Denetçiyle KATILMADIĞIM nokta

Denetçi, "sıfırdan render kaybı ölçekte çalışmıyor" sonucunun dayandığı taban
koşunun *hiç öğrenmediğini* iddia etti (`step 0 loss=0.7205 ↔ step 5980 loss=0.7540`).
`t1_taban.log` bunu **doğrulamıyor**: kayıp `3.9051 → 0.6709`, mask terimi
`0.5631 → 0.4452`. Koşu öğreniyor. *(Verdiği sayılar başka bir kolun olabilir.)*

Ama altındaki gözlem gerçek ve önemli: **val PSNR üç ölçümde de `15.11` dB —
virgülden sonra iki hane aynı**, `top1 = %1,6 = şans`, `oran = 1,282` (ortalama
blob'dan kötü), buna karşılık `obj_std = 0,105` (çöküş eşiğinin çok üstünde).
Yani çıktılar objeler arasında **farklı ama hedefleriyle ilişkisiz** — `guards.py`
bu durumu yakalamıyor, "saglikli" bastı. Ve kaybın ~%66'sı `mask` terimi.
**Tek değişkenli `w_mask` taraması hak ediyor** — ama gerekçesi "koşu hiç
öğrenmedi" değil, "kaybın üçte ikisi ve çıktı hedefiyle ilişkisiz".

`guards.py`'ye eklenecek iki tek satır: `ACC_MAX` (dolu-küp çöküşü; kol A 4000
adım `acc≈0.50`'de koştu) ve `FROZEN_OUTPUT`'un bir eval geç ateşlenmesi
(`train_lrm.py:725` geçmeden önce ekliyor).

### 8.6 Revize edilmiş sıradaki işler

K256 bitene kadar **hiçbir mimari değişiklik yapılmayacak.** Sonrasında sıra:

1. **ORACLE KOŞULLANDIRMA (~2 sa) — her şeyden önce.** `distill_lrm`'i encoder
   çıktısı yerine **obje başına serbest, öğrenilebilir token ızgarasıyla** koş
   (aynı boyut, aynı transformer, aynı head, aynı 1024 hedef, aynı adım sayısı).
   `rel` düşerse darboğaz **koşullandırma** (girdi çözünürlüğü haklı);
   düşmezse **transformer + head kapasitesi** (token ızgarası büyümeli).
   **Planın yarısını siler veya doğrular.** Bu ayrım bugün hiçbir yerde yok.
2. **`w_mask ∈ {1,0 · 0,25 · 0}` taraması (~3 sa)**, 1024 obje, `--density_bias 1.0`,
   diğer her şey sabit. `w_mask=0` ile sıfırdan render kaybı çalışıyorsa
   **3 aşamalı zincirin ve öğretmen VRAM duvarının tamamı gereksizleşir.**
3. **Gerçek foto değerlendirme setini n=1 → n=20'ye çıkar** (telefon, 4 kanonik
   açı, `prep_photo`). Ürünün kalitesini belirleyecek sayı `kor` sütunu değil bu.
4. `kor`'u çok ölçekli bant geçirene çevir (§8.3).
5. `tp_res` artışına **yalnızca** `triplane_res` token ızgarasıyla **birlikte** git;
   önce `TriplaneHead`'e `upsample` parametresini geçir (`model.py:32`) ve
   `distill_lrm` / `train_lrm`'e `--tp_res` / `--tp_ch` bayrakları ekle.

**Açık kalan çelişki (kayda geçsin):** §5 "NeRF MLP kapasitesi elendi" diyor ama
`kuyruk_31agu.out` kapasite taramasında **ızgara sabitken** `nerf 8×128` `kor`'u
0,010 → 0,085 (**8,5×**) oynatmış — `tavan2`'nin tp64→tp128 sıçramasından büyük.
O eleme `kor+kaydır` ve PSNR'a bakılarak yapılmıştı; ayrıca o fitler küçültme
düzeltmesinden **önce** alındı. "Elendi" denemez, yeniden ölçülmeli.
Aynı şekilde `bound 0,552` kalibrasyonu belgede "yapıldı" yazıyor ama
`defaults.BOUND` hâlâ **0.6** — bedava %28 hacim yoğunluğu masada duruyor.
