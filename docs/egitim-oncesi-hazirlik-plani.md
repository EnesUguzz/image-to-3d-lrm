# Eğitim Öncesi Hazırlık Planı — 2026-08-26

**Amaç:** Tam veri setiyle (24,6k → 50k) uzun eğitime girmeden önce her kararı
kapatmak, her aracı hazır etmek, her ölçümü güvenilir kılmak. Uzun koşu sırasında
"acaba şunu mu değiştirsek" sorusu kalmasın.

**Bu belge nasıl okunur:** Her iddianın yanında onu üreten ölçüm var. Ölçümü
olmayan hiçbir şey karar değildir. Önceki durum: [`PROJE-DEVIR-BELGESI.md`](PROJE-DEVIR-BELGESI.md).

---

## 0. Yönetici özeti

| soru | cevap |
|---|---|
| Sistem sağlam mı? | Render + temsil + araçlar **evet**. Ölçekte conditioning **hayır**. |
| Kök neden ne? | Tarifemiz OpenLRM'in kendi örnek config'inden **5 noktada** sapıyor ve 5'i de aleyhimize. Ayrıca **~10 epoch** yaptık, referans **60**. |
| Ne kadar hızlanabiliriz? | Ölçüldü: **3,0×** (192 ms/obje → 63,5 ms/obje). |
| Veri tavanı? | Diskteki 51.534 glb'de temiz score≥2 = **24.571 obje**. 50k için **~25k yeni indirme (~245 GB)** şart. |
| Ne kadar sürer? | Hazırlık **~5 iş günü**, tam eğitim **~27 saat** (24,6k × 60 epoch @ 15 obje/s). |
| En büyük risk? | Kapasite sınırı. **Blok 2 bunu 1 günde kesin olarak kapatıyor.** |

**Tek cümlelik strateji:** Referans tarifeyi (OpenLRM-small) *ablasyon yapmadan
olduğu gibi benimse*; deney bütçesini sadece bizim rejimimize özgü sorulara harca.
Bugüne kadarki döngünün sebebi, referansın zaten çözdüğü şeyleri 32 objelik
gürültülü bir tezgâhta yeniden türetmeye çalışmaktı.

---

## 1. Bu oturumda ÖLÇÜLEN yeni bulgular

### 1.1 🔴 `M_base` ve `M_enc4` dejenere koşulardı — tarife matrisi geçersiz

Devir belgesinde "`M_enc4`, `M_base` ile bit-bazında aynı, sebebi bulunamadı"
yazıyor. Gerçek sebep bulundu:

```
M_base : trainPSNR = 17.80 dB  (adım 500, 1000, 1500, 1999'da AYNI, 2 ondalık)
M_enc4 : trainPSNR = 17.80 dB  (aynı)
tahmin görüntüsü: pred_min = 1.000, pred_mean = 1.0000, std = 0.0000
```

**Her iki koşu da yoğunluğu sıfıra çökertip TAMAMEN BEYAZ çıktı üretti**
(`acc→0` ⇒ renderer `bg_color`'ı aynen basar). Bu bir *soğurucu durum*:
density sıfıra oturunca softplus gradyanı ölür, encoder'ı açmak da açmamak da
hiçbir şey değiştiremez. Loss değerleri aslında 5. ondalıkta farklı
(0.5465 vs 0.54651) — "bit-bazında aynı" tespiti de yanlıştı.

**Sonuçları:**

- "Kısmi encoder çözme cevapsız" kaydı → **soru hâlâ açık ama sebebi artık biliniyor**.
- `M_base` bir *taban tarife* değil, bir *çöküş*. Matrisin tüm sıralaması
  (`oran = 1.000` referansına göre) **geçersiz**. `overnight.sh`'ın kazanan
  seçici mantığı çöp veriyle çalıştı.
- Aynı tarife, aynı 2000 adım, farklı 32-obje kümesi (`AB_YENI`, `_ab_uids.json`)
  **çökmedi** (20,11 dB / top-1 %75). Yani çöküş *hem tarifeye hem alt kümeye*
  bağlı — optimizasyon manzarasında kararsız bir bölge.

**Aksiyon:** her koşuya **çöküş alarmı** ekle (`acc.mean() < 0.01` veya objeler
arası çıktı std < 0,005 ⇒ `DEJENERE` etiketi, sıralamaya sokma).

### 1.2 🟢 Hız: ölçülmüş 3,0× kazanç mevcut

Obje başına tam eğitim adımı (ileri + geri + loss), RTX 5080, gerçek model:

| konfigürasyon | ms/obje | kümülatif |
|---|---|---|
| **bugünkü** (fp32, 4 görünüm × 128²) | **192,3** | 1,00× |
| bf16 autocast (loss fp32) | 107,5 | 1,79× |
| bf16 + 4 görünüm × 64² | 66,9 | 2,87× |
| **bf16 + 3 görünüm × 64² (OpenLRM tarifesi)** | **63,5** | **3,03×** |

**Faydası ölçülüp ELENENLER** (bir daha deneme):

| fikir | ölçüm | karar |
|---|---|---|
| `torch.compile` (nerf + transformer) | 63,5 → 61,9 ms (%2) | ✗ değmez |
| obje-batch'leme (transformer B=1→8) | 116 → 156 ms/örnek (fp32); bf16'da fark yok | ✗ zaten compute-bound |
| ışın–AABB near/far kırpma | örneklerin %69,8'i zaten kutu içinde ⇒ teorik 1,4×, küçük terim | ✗ öncelik değil |
| `n_samples` 48 → 16 | 190 → 148 ms | ~ kaliteyle takas, kazanç küçük |

**Not:** `torch.compile` proje yolundaki **Ğ** yüzünden `UnicodeDecodeError`
veriyor. `TORCHINDUCTOR_CACHE_DIR=C:/tmp/inductor` ile çalışıyor — ama zaten
faydasız olduğu için gündemden düşüyor.

**Veri yükleme:** senkron, tek iş parçacığı, **28,5 ms/obje** (adımın bugün %14'ü).
Hızlanmadan sonra %45'i olur ⇒ `DataLoader(num_workers=6, persistent_workers=True,
prefetch_factor=4)` **şart**.

Bugünkü fiili hız **5,06 obje/s**. Hedef: **≥15 obje/s**.

### 1.3 🟡 Distilasyon sonucu eksik koşudan çıkarılmış

"1024 objede öğrenci bağıl hata 0,71'de takıldı ⇒ sınır kapasite" kaydı,
`distill.log`'a göre **adım 3200/12000'de** (programın %27'si, kosinüs LR hiç
sönmemiş) durmuş bir koşudan. O anda öğrenci std'si hâlâ **tırmanıyordu**
(0,132 → 0,150; hedef 0,339). Ayrıca `egitim_rel` (0,78) > `val_rel` (0,70) —
yani **aşırı öğrenme değil, YETERSİZ öğrenme**. Bu, "kapasite" hipotezi için
*işaret* ama *kanıt değil*. Tamamlanmış bir koşu gerekiyor (Blok 2).

### 1.4 🟢 Veri tavanı kesin olarak biliniyor

Diskteki 51.534 glb ∩ Objaverse++ 800k anotasyonu = 50.910 obje.
Temiz filtre = `is_multi_object`, `is_scene`, `is_transparent`, `is_single_color`
hiçbiri `"true"` değil:

| score | diskte toplam | temiz | durum |
|---|---|---|---|
| 3 | 18.409 | **14.594** | ✅ render edildi (14.597 klasör) |
| 2 | 22.638 | **9.977** | ⬜ render edilmedi (~1,8 saat, ~25 GB) |
| 1 | 8.306 | 1.241 | düşük kalite |
| 0 | 1.557 | 709 | düşük kalite |

- **Diskten çıkabilecek maksimum kaliteli veri: 24.571 obje (score 2+3).**
- Tüm Objaverse++ havuzunda temiz score≥2 = **382.176** obje ⇒ 50k'ya çıkmak
  teknik olarak mümkün, ama **~25k yeni glb indirmek gerekiyor (~245 GB)**.
- Disk: **792 GB boş** — yeterli. Render sonrası glb silinirse hiç sorun yok.

### 1.5 ⛔ Kopya alarmı YANLIŞ POZİTİFTİ — düzeltildi

İlk ölçüm (3000 obje, **sadece kanonik[0]** üzerinden 16×16 imza) %1,3 kopya
bildirdi; tam veride 34 küme / 246 fazladan obje çıktı. **Hepsi yanlış pozitif.**

En büyük kümeye (91 obje) gözle bakıldı: bunlar **plastik kılıflı koleksiyon
kartları**. Düz oldukları için kanonik ön (az 0°) ve arka (az 180°) açılarında
**kenardan** görünüyorlar — ince bir çizgi, hepsi birbirinin aynı. Yan açılarda
(90°/270°) kartların kendisi görünüyor ve tamamen farklılar:

```
kume 0 (91 obje):  on=0.0008  arka=0.0007   <- ayni gorunuyor
                   sag=0.0293  sol=0.0309   <- farkli objeler
rastgele kontrol:  on=0.0880  arka=0.0912  sag=0.1355  sol=0.1379
```

İmza **dört kanonik açının hepsinden** alınınca (hepsi tüm objelerde aynı kamera):
**0 kopya kümesi.** Yani val sızıntısı da yok.

> **Ders:** Düz/ince objeler kanonik ön-arka açılarda dejenere görünür. Bu veri
> setinde tek görünüme dayanan hiçbir imza güvenilir değil.

**Bunun yerine gerçek bir sorun bulundu:** girdi görünümü (kanonik[0]) neredeyse
boş olan **134 obje (%0,93)** var — o objelerde "ön foto" bir çizgi, hiçbir
fonksiyon doğru 3B veremez. Elendi.

### 1.6 🔴 `train_lrm.py` her koşu sonunda `last.pt`'yi KOŞULSUZ eziyordu

**Bu hata teoride değil, fiilen gerçekleşti (2026-08-26, Blok 0 sırasında):**
40 adımlık bir duman testi, `train()` fonksiyonunun son satırındaki koşulsuz
`save_checkpoint(last_ckpt, ...)` yüzünden gece koşusunun **16.000 adımlık
checkpoint'ini ezdi**; dosya geri getirilemedi. (`overnight.sh` bu tuzağı
biliyormuş — duman testinden sonra `rm -f dataset/lrm_ckpts/last.pt` yapıyor.
Yani tuzak kayıtlıydı ama kapatılmamıştı.)

Kayıp pratikte küçük: val eğrisi (`val_metrics.jsonl`, 31 satır) ve önizlemeler
duruyor, plan zaten o tarifeden devam etmiyor. Ama 27 saatlik bir koşuda aynı
hata felaket olurdu.

**Kapatıldı:** `save_checkpoint` artık daha KÜÇÜK adımlı bir kaydı, daha BÜYÜK
adımlı mevcut bir kaydın üzerine `force=True` olmadan yazmıyor; yazmadığında
`False` dönüyor ve uyarı logluyor. 3 regresyon testi eklendi
(`test_kisa_kosu_uzun_kosunun_checkpointini_EZEMEZ` vb.).

### 1.7 🟢 Tam veri denetimi yapıldı — render pipeline temiz çıktı

`scripts/audit_dataset.py` (yeni), **14.454 obje × 16 görünüm = 231.264 render**:

| kontrol | sonuç |
|---|---|
| eksik/bozuk `meta.json`, eksik/bozuk PNG | **0** |
| yanlış çözünürlük, boş render | **0** |
| kamera yarıçapı tutarsızlığı, intrinsic tutarsızlığı | **0** |
| **kenar taşması** | **0** ← `fit` çerçeveleme artık 100 objede değil, TÜM veride doğrulandı |
| tek renk objesi | 62 |
| karanlık obje | 18 |
| **toplam sorunlu** | **79 / 14.454 = %0,55** |

Kaplama medyanı 0,0903 (kayıtlı 0,0899 ile tutarlı).

**🔑 `bound` kalibrasyonu (E8 kapandı).** Obje yarıçapı silüetten ters-projeksiyonla
ölçüldü: medyan **0,453**, p99 **0,512**, maks **0,540**.
⇒ `bound` **0,6 → 0,552** çekilebilir. Kazanç: aynı triplane ızgarasında
**1,28× etkin hacim yoğunluğu**, bedava. (Maks 0,540 olduğu için 0,552 güvenli pay.)

**⛔ Kopya raporu geçersiz (bkz. §1.5).** Tek-görünüm imzası düz objelerde
yanılıyordu. Dört kanonik açıyla yeniden ölçüldü: **0 kopya, 0 val sızıntısı.**

---

### 1.9 🔴 G2 kapısı DÜŞTÜ (250 obje)

Kazanan tarifeyle (bf16 + `density_bias 1.0`), sabit `bench_uids_250`, 6000 adım:

```
adım    0   9,43 dB  top-1  0,4%  oran 1,015
adım 2000  16,62 dB  top-1  0,8%  oran 1,027
adım 4000  17,68 dB  top-1  7,6%  oran 0,888
adım 6000  18,23 dB  top-1 13,6%  oran 0,814   ← final
```

| ölçüt | ön-kayıtlı eşik | sonuç |
|---|---|---|
| top-1 | ≥ %60 | **%13,6** |
| oran | ≤ 0,50 | **0,814** |

**Eşik gevşetilmiyor.** Olumlu yanı: hiç çöküş bayrağı yok, eğri monoton, top-1
şansın **34 katı** — conditioning var, ama plandaki seviyede değil.

**⚠️ Kendi kurduğum karıştırıcı:** G2'ye obje başına **96 maruziyet** verdim
(6000 × 4 ÷ 250); 32-obje tezgâhı **250 maruziyet** almıştı. Yani "32'de %41,
250'de %13,6" karşılaştırması maruziyet açısından eşit değil. Eşik geçersiz
olmuyor (adım sayısı bilinerek yazıldı) ama tek başına "kapasite sınırı"
sonucunu da desteklemiyor. İki okuma:
(a) tarife ölçeğe transfer etmiyor, (b) kapıya yetersiz bütçe verildi.
**Blok 2 tam olarak bu ayrımı yapan deney** → oraya geçildi.

*Not: 60 epoch'luk gerçek eğitim rejimi obje başına 60 maruziyet demek, yani
G2'nin 96'sı gerçek rejime 32-obje tezgâhının 250'sinden DAHA yakın.*

### 1.10 ✅ Blok 5 GEÇTİ — mesh çıkıyor, seyreklik terimi gerekmiyor (E1)

Oracle triplane üzerinde, `scripts/extract_mesh.py`:

| obje | IoU (silüet↔GT) | üçgen | parça | watertight |
|---|---|---|---|---|
| index 1 | **0,914** | 39.986 | 1 | ✅ |
| index 12 | **0,916** | 29.332 | 1 | ✅ |
| index 0 (ince/düz) | 0,674 | 8.900 | 1 | ✅ |

- Dolu voksel oranı **%0,6–2,7** ⇒ yoğunluk alanı **seyrek ve yüzeysel**, sisli değil.
- Tek bağlı parça, eşik 0,0012 → 112 arasında (5 büyüklük mertebesi) korunuyor.
- **Sonuç: eğitim hedefine yoğunluk seyreklik/entropi terimi EKLEMEYE GEREK YOK.**

**İki tuzak çıktı ve kapatıldı:**
1. Eşiği tüm hacmin yüzdeliklerinden seçmek hepsini 0'a düşürüyordu (hacmin
   %99,4'ü boş) ⇒ gürültü de meshleniyordu. Eşikler artık **sıfırdan büyük**
   yoğunlukların dağılımından alınıyor.
2. `sample_triplane` `padding_mode="border"` kullanıyor ⇒ kutu kenarında sahte
   bir kabuk. Renderer'ın `inside` maskesi bunu gizliyor, marching cubes görüyor.
   Kenar voksel kırpılıyor.

**Eşik otomatikleşti** (`--auto_level`): GT silüet IoU'sunu maksimize ediyor.
Objeden objeye yoğunluk ölçeği çok değişiyor (aynı objede IoU 0,589 → 0,741 →
0,545), elle seçim Faz C'de imkânsızdı. Faz C'de kullanıcı fotosunun maskesi
zaten elimizde olacağı için aynı ölçüt oraya taşınır.

---

---

## 2. Referans tarifelerle karşılaştırma (kritik tablo)

OpenLRM'in kendi `configs/train-sample.yaml` dosyası mimarimizle **neredeyse
birebir aynı**: dim 512, 12 kat, 8 kafa, triplane 32→64 × 32 kanal, DINOv2
ViT-S/14, girdi 224, lr 4e-4, wd 0,05, betas 0,9/0,95, clip 1,0, tv 5e-4.
Yani mimari tartışması kapandı. Fark **tarifede**:

| # | parametre | bizde | OpenLRM-small | TripoSR | etki |
|---|---|---|---|---|---|
| 1 | **encoder** | **DONUK** | `encoder_freeze: false` | ön-eğitimli DINOv1 | 🔴 büyük |
| 2 | **denetim görüntüsü** | tam kare 128², coarse-to-fine | render 64–192 arası rastgele, **64² BÖLGE kırpma** | 512²'den **128² rastgele yama**, ön-plana yanlı | 🔴 büyük |
| 3 | **hassasiyet** | fp32 (bf16 yasaklı) | **bf16** | — | 🟠 1,8× hız |
| 4 | **kamera normalizasyonu** | **kapalı** | `normalize_camera: true` | — | 🟠 orta |
| 5 | **warmup** | 500 | **3000** | 2000 | 🟠 orta (çöküşe karşı) |
| 6 | epoch | ~10 (16k adım) | **60** | — | 🔴 büyük |
| 7 | efektif batch | 8 | 16/GPU (× N GPU) | — | 🟠 orta |
| 8 | `n_samples` | 48 | 96 | 128 | 🟡 küçük |
| 9 | LPIPS ağırlığı | 0,25 (32 objede ölçüldü) | 1,0 | 2,0 | 🟡 bizim rejime özgü |
| 10 | mask loss | fg-ağırlıklı L1, w=1,0 | **yok** | BCE, **λ=0,05** | 🟡 bizimki ~20× ağır |
| 11 | arka plan | sürekli rastgele renk | `choice([0.0, 0.5, 1.0])` | — | 🟢 denk |
| 12 | girdi görünümü | sadece 4 kanonik | 32 render'ın herhangi biri | — | 🟢 bilinçli (Faz C ile tutarlı) |
| 13 | supervision görünüm | 4 | 3 | — | 🟢 denk |
| 14 | NeRF MLP | 2 kat, ReLU | — | 10 kat, SiLU | 🟢 bizde ölçüldü: 2 kat daha iyi |

**Yorum:** 1, 2 ve 6 satırları tek başına bile ölçekte conditioning'i öldürmeye
yeter. Üçü birden bizde yanlış. "Kök neden bulunamadı" hissinin gerçek kaynağı bu.

**Kaynaklar**

- OpenLRM `configs/train-sample.yaml` + `model_card.md` — github.com/3DTopia/OpenLRM
- TripoSR (arXiv 2403.02151): triplane 64²×40, 16 kat × 1024 dim, mask BCE λ=0,05,
  LPIPS 2,0, **512²'den 128² ön-plana yanlı rastgele yama**, 128 örnek/ışın,
  AdamW 4e-4, cosine, 2000 warmup
- Objaverse++ (arXiv 2504.07334): **kaliteli ~50k, rastgele 100k'yı yeniyor**;
  "veri kalitesi yükseldikçe eğitim kaybı daha hızlı yakınsıyor" ⇒ score-3
  odaklı kürasyonumuz doğru yönde

---

## 3. SAĞLAM — dokunma

| # | ne | kanıt |
|---|---|---|
| S1 | Render pipeline (`fit` çerçeveleme + EEVEE_NEXT) | 2400 render alfa denetimi: taşma %0,00; motor A/B 384 çiftte PSNR 37,4 dB; 14.597 obje 0 hata |
| S2 | Kamera konvansiyonu (`c2w = inv(extrinsic)`, OpenGL) | oracle triplane 16 görünümde 25,04 dB — tutarsız kamerayla imkânsız |
| S3 | Temsil kapasitesi (triplane 64²×32, NeRF 2 kat) | O1 25,04 dB tavan; O2 (10 kat) 21,66; O4 (32²) 22,89 |
| S4 | Mimari seçimi (joint self-attention) | B_cross_attn 17,05 vs A 17,06 — fark yok, %80 daha az parametre |
| S5 | Sayısal val metriği (`psnr` / `top1` / `oran`) | eğitilmemiş modelde top1 = şans, oran = 1,000 |
| S6 | Teşhis araç seti | `bench_overfit`, `bench_triplane_fit`, `bench_view_coverage`, `diag_*` |
| S7 | Veri kürasyonu (Objaverse++ score-3 + alpha filtresi) | %99,04 geçme oranı; Objaverse++ makalesi kaliteyi doğruluyor |
| S8 | Alt/üst görünüm gerekmiyor (supervision tarafı) | COV_alt0 27,26 dB vs COV_alt1 27,74 dB |
| S9 | `dataset.py` RNG düzeltmesi + regresyon testleri | 63 test geçiyor |
| S10 | Aydınlatma kararı (düz, alt dolgulu) | NeRF bakış yönü almıyor ⇒ düz aydınlatma kapasiteyle eşleşiyor |

---

## 4. KUSURLU / YANLIŞ KAYDEDİLMİŞ — düzelt

| # | ne | gerçek durum | aksiyon |
|---|---|---|---|
| K1 | "M_enc4 geçersiz, sebep bulunamadı" | İki kol da **beyaz çöküşe** düştü (§1.1) | Matris sonuçlarını iptal et; çöküş alarmı ekle |
| K2 | "bf16 rengi öldürüyor ⇒ fp32 şart" | 2026-08-19 tarihli, **eşit olmayan koşu** (1450 vs 2450 adım), 4 bilinen bug düzeltilmeden önce. OpenLRM bf16 kullanıyor. Bedeli **1,8×** | Kontrollü yeniden test (aynı adım/seed, loss fp32'de hesaplanarak) |
| K3 | "Öğrenci 1024'te 0,71'de takıldı ⇒ kapasite" | Koşu **%27'de kesilmiş** (§1.3) | Tam programla tekrar (Blok 2) |
| K4 | `TriplaneNeRF`'in `density_bias` / `noise_std` çengelleri | Yazılmış ama **hiç bağlanmamış** — `LRM.__init__` daima varsayılan 0,0 geçiyor. Tam da yaşadığımız çöküşün panzehiri | CLI'ya bağla + A/B |
| K5 | `overnight.sh` kazanan seçici | Aktarılamayan bayrak kazanınca sessizce "(yok)" yazdı ⇒ 7 saatlik koşu taban tarifeyle gitti | Seçiciyi kaldır; tarife elle sabitlenip commit'lensin |
| K6 | Veri yükleme | `DataLoader` yok, senkron, tek thread | worker'lı DataLoader |
| K7 | Weight decay | `tp_tokens`, LayerNorm ve bias'lara da 0,05 uygulanıyor | Parametre grupları (norm/bias/embed → wd 0) |
| K8 | ~~`n_samples` varsayılanı~~ | CLI 48, `LRM.__init__` 64 — sessiz tutarsızlık | ✅ `lrm/defaults.py` |
| K9 | EMA / grad-norm / acc-mean logu | Yok | Ekle (EMA 0,999 tipik +0,2–0,5 dB, neredeyse bedava) |
| K10 | Encoder LR | Encoder çözülürse ana LR'yi (4e-4) alıyor — ön-eğitimli ViT için 10–40× fazla | Ayrı parametre grubu, `lr/10` |
| K11 | ~~Bazı `open()` çağrılarında `encoding` yok~~ | Proje yolunda **Ğ** var | ✅ `scripts/` altında 7 çağrı düzeltildi |
| K12 | ~~Kopya objeler split'e bölünmüş~~ | **Yanlış alarm** — tek-görünüm imzası düz objelerde yanılıyordu; 4 açıyla 0 kopya (§1.5) | ✅ kapandı |

---

## 5. HİÇ YAPILMAMIŞ ama eğitim öncesi ŞART

| # | eksik | neden eğitimden ÖNCE |
|---|---|---|
| E1 | ~~**Mesh çıkarma doğrulaması**~~ ✅ IoU 0,92 | Nihai ürün `.glb`. Yoğunluk alanımız "sisli" ise marching cubes çöp verir ve bu **eğitim hedefini değiştirir** (seyreklik regülarizasyonu gerekir). 40 saatlik koşudan sonra öğrenilecek şey değil. **Oracle triplane (25 dB) üzerinde bugün test edilebilir.** |
| E2 | **Gerçek foto değerlendirme seti** | Faz C'nin girdisi telefon fotoğrafı. 20–30 fotoluk mini set olmadan augmentation/aydınlatma politikası kör karar. |
| E3 | ~~**Tam veri denetim raporu**~~ ✅ | Taşma/kaplama sadece **100 objede** ölçüldü, 14.597'de değil. Kopya, boş render, kamera tutarlılığı hiç taranmadı. |
| E4 | **Dokunulmamış test seti** | Şu an train/val var; val hem tarife seçiminde hem raporlamada kullanılıyor ⇒ seçim yanlılığı. |
| E5 | **Resume doğrulaması** | `--resume` hiç "öldür ve devam et" testinden geçmedi. 27 saatlik koşuda bu tek başına projeyi batırabilir. |
| E6 | **Uzun koşu dayanıklılığı** | Windows uyku/güncelleme, sürücü TDR, termal. Otomatik yeniden başlatan gözetmen yok. |
| E7 | **Çöküş alarmı** | §1.1'deki beyaz çöküş 7 saat sessizce sürebilirdi. |
| E8 | ~~**`bound` kalibrasyonu**~~ ✅ 0,552 | Triplane hacminin sadece **%34'ü** kullanılıyor. `fit` normalizasyonundan sonra gerçek obje yarıçapı hiç ölçülmedi. |

---

## 6. PLAN

Bloklar sıralı; her blok bir **çıktı** ve bir **karar** üretir.
GPU gerektirmeyen işler (indirme, denetim) paralel yürür.

### Blok 0 — Ölçüm hijyeni ✅ **TAMAMLANDI (2026-08-26)**

Amaç: bundan sonraki hiçbir ölçüm sessizce yanlış olmasın.

**Üretilenler:**

| dosya | iş | doğrulama |
|---|---|---|
| `scripts/lrm/guards.py` | çöküş dedektörü (`EMPTY_COLLAPSE` / `MEAN_COLLAPSE` / `FROZEN_OUTPUT`) | 12 test; eşik gerçek çıktılarla kalibre edildi |
| `scripts/lrm/runstamp.py` | koşu künyesi + **ağırlık-değişti kontrolü** | 7 test |
| `scripts/lrm/defaults.py` | `N_SAMPLES` / `BOUND` / `NEAR` / `FAR` tek kaynak | K8 kapandı |
| `scripts/make_bench_uids.py` | sabit, **iç içe** tezgâh kümeleri | 32 ⊂ 250 ⊂ 1024 doğrulandı |
| `dataset/bench_uids_{32,250,1024}.json` | commit'lenen uid kümeleri | `.gitignore`'a istisna eklendi |
| `scripts/bench_speed.py` | hız tezgâhı | taban + hedef ölçüldü |

**Eşik kalibrasyonu** (`dataset/lrm_bench/*.png` üzerinde objeler arası piksel std):

| çökmüş | değer | | sağlam | değer |
|---|---|---|---|---|
| `M_base` | 0,0000 | | `val_015500` | 0,0382 |
| `M_enc4` | 0,0000 | | `AB_YENI` | 0,0520 |
| `AB_ESKI` | 0,0001 | | `M_crop08` | 0,0881 |

Boşluk ~400× ⇒ eşik **0,010**. `AB_ESKI`'nin de dedektöre takılması, "eski render
seti tam çöküş" kaydını bağımsız olarak doğruluyor.

**`M_enc4` gizemi tamamen kapandı:** yeni ağırlık-değişti kontrolüyle
`--unfreeze_last 2` koşulduğunda **30/30 encoder tensörü değişiyor**
(`ort |delta| = 1.5e-04`). Yani gradyan gerçekten akıyordu; koşunun etkisiz
görünmesi baştan sona beyaz-çöküş soğurucu durumundandı.

**Çöküş alarmı canlı doğrulandı:** eğitilmemiş modelde `obj_std = 0,0002`,
`oran = 1,000` ⇒ `MEAN_COLLAPSE`. Eğitilmemiş model *tanımı gereği* bu durumda
olduğu için gürültülü alarm yapmasın diye alarm `2 × warmup` adımından sonra
çalıyor (`--collapse_after`).

**Ölçülen hız (taban ↔ hedef, `bench_speed.py`):**

| tarife | ms/obje | obje/s (GPU) | VRAM |
|---|---|---|---|
| taban — fp32, 4 gör × 128², ns 48 | 196,7 | 5,1 | 7,25 GB |
| + bf16 | 107,1 | 9,3 | 5,27 GB |
| **hedef — bf16, 3 gör × 64², ns 48** | **69,0** | **14,5** | **2,72 GB** |
| hedef + ns 96 (OpenLRM değeri) | 74,6 | 13,4 | 3,02 GB |

`n_samples` 48 → 96 yükseltmesi sadece **%8** maliyetli — ucuz kalite kazancı.
Hedef tarifede VRAM 16 GB'ın altıda biri ⇒ **Blok 2b için bol yer var**
(daha büyük model / girdi 448).

> ⚠️ Bu ölçümler *sentetik girdiyle* GPU maliyetini verir. Gerçek koşuda senkron
> veri yükleme (17–25 ms/obje) üstüne biner; worker'lı `DataLoader` (Blok 1.1)
> bunu gizleyene kadar 14,5 değil ~10,9 obje/s görülür.

**Yapılan ek düzeltmeler:** `scripts/` altındaki `encoding`'siz 7 `open()` çağrısı
(K11), `save_checkpoint` ezme koruması (§1.6).

Aşağıdaki liste tamamlanan işin tanımıdır:

1. **Çöküş dedektörü** (`scripts/lrm/guards.py`)
   - `acc.mean() < 0.01` ⇒ `EMPTY_COLLAPSE`
   - objeler arası tahmin std < 0,005 ⇒ `MEAN_COLLAPSE`
   - N ardışık değerlendirmede PSNR aynı (±0,01) ⇒ `FROZEN_OUTPUT`
   - Etiket her `bench_*.json`'a ve her `[VAL]` satırına yazılır; dejenere koşu
     **sıralamaya girmez**.
2. **Sabit tezgâh kümeleri**, repoya commit: `dataset/bench_uids_32.json`
   (= `_ab_uids.json`, çökmediği doğrulanmış), `_250.json`, `_1024.json`.
   Kural: A/B kolları **daima** `--uids` ile sabitlenir.
3. **Koşu künyesi**: her sonuç json'ına `git_sha`, tam `argv`, `torch.__version__`,
   GPU adı, config hash. Aynı künye = aynı koşu.
4. **Ağırlık-değişti kontrolü**: encoder açıldığı iddia edilen her koşuda
   başlangıç–son ağırlık farkı loglanır (`M_enc4` dersi).
5. **Hız tezgâhı** (`scripts/bench_speed.py`): ms/obje ve obje/s'yi tek komutla
   raporlar; her hız değişikliği bununla doğrulanır.
6. `encoding="utf-8"` taraması; `n_samples` tek kaynağa indirilir.

**Çıktı:** güvenilir ölçüm altyapısı. **Karar:** yok (altyapı).

---

### Blok 1 — Hız (1 gün) — hedef ≥15 obje/s

Sırayla; her adımdan sonra `bench_speed.py` + 32-obje kalite kontrolü:

| # | değişiklik | beklenen | doğrulama |
|---|---|---|---|
| 1.1 | `DataLoader(num_workers=6, persistent_workers=True, prefetch_factor=4)` | 28,5 ms/obje CPU maliyeti gizlenir | ms/obje düşer, sonuç aynı |
| 1.2 | **bf16 autocast, loss fp32'de** (`rgb.float()`, `acc.float()` loss'a girmeden önce) | 1,8× | **K2'nin kontrollü testi:** aynı seed, aynı 2000 adım, `bench_uids_32` ⇒ PSNR/top-1 farkı < 0,3 dB ise benimse |
| 1.3 | **Bölge kırpma** (`region-crop`): supervision render çözünürlüğü `r ~ U[64,192]`, sonra **64² bölge** kırp, intrinsic'i ölçekle + kaydır; TripoSR gibi ön-plana yanlı seçim | 1,6× + çok daha yoğun ön-plan sinyali | `bench_overfit.sample_fg_crop` zaten var; `dataset.py` + `train_lrm.py`'ye taşı |
| 1.4 | `n_sup` 4 → 3 (OpenLRM) | %5 | — |
| 1.5 | coarse-to-fine'ı **kaldır** (bölge kırpma yerini alıyor) | — | val eğrisindeki 8500. adım sıçraması ortadan kalkar |

**Çıktı:** ≥15 obje/s. **Karar:** bf16 benimsendi mi (evet/hayır, ölçümle).

> ⚠️ 1.3 uygulanırken: `bench_overfit.py`'deki `--crop` bayrağı **merkez kırpma**,
> OpenLRM'inki **rastgele bölge**. İkisi aynı şey değil. Ölçülen kazanç
> (`M_crop08`, top-1 %38) merkez kırpmadandı; rastgele bölge daha genel ama
> ilk adımlarda daha gürültülü. **İkisini de destekle, Tier-2'de karşılaştır.**

---

### Blok 2 — KARAR KAPISI: image→triplane eşlemesi ölçekte öğrenilebiliyor mu? (1 gün)

Projenin en kritik ve en ucuz deneyi. Render kaybını tamamen devre dışı bırakır;
sadece "bu ağ, bu encoder ile N objenin triplane'ini üretebilir mi" sorusunu sorar.
Sinyal yoğunluğu render'a göre **~90×** (obje başına 393k skaler vs 4,4k).

**Neden belirleyici:** iki hipotezi kesin ayırır.

- Öğrenebiliyorsa ⇒ sorun **render kaybı optimizasyonu** ⇒ çözüm tarife (Blok 3).
- Öğrenemiyorsa ⇒ sorun **kapasite/encoder** ⇒ çözüm mimari (Blok 2b).

**Koşu:**

```bash
# Asama 1 - ogretmen triplane bankasi
# (mevcut teacher_init.pt ESKI/olu render setinde fit edilmisti, yeniden gerekiyor)
python scripts/fit_teacher.py --train_list dataset/train_list_opp_score3.json \
  --renders_dir dataset/renders_opp_score3 --n_obj 1024 --steps 25600 --batch 4 \
  --out dataset/lrm_ckpts/teacher_1024_new.pt          # ~1,5 sa, 1,6 GB

# Asama 2 - saf distilasyon, TAM program (onceki kosu %27'de kesilmisti)
python scripts/distill_lrm.py --teacher dataset/lrm_ckpts/teacher_1024_new.pt \
  --train_list dataset/train_list_opp_score3.json \
  --renders_dir dataset/renders_opp_score3 --steps 24000 --batch 8
# => obje basina ~190 maruziyet (dogrulanmis T2 deneyinde 150 yetmisti)
```

**Eşikler (koşudan ÖNCE yazıldı — sonradan değiştirilmez):**

| sonuç | yorum | sonraki adım |
|---|---|---|
| `train_rel < 0,25` **ve** `öğrenci_std > 0,8 × hedef_std` | ✅ eşleme öğrenilebiliyor | Blok 3 (tarife) |
| `0,25 ≤ train_rel < 0,50` | 🟡 sınırda | Blok 2b'nin sadece encoder ayağı |
| `train_rel ≥ 0,50` | 🔴 kapasite/encoder sınırı | Blok 2b tamamı |

**Blok 2b — kapasite merdiveni** (sadece kapı düşerse; her basamak ayrı ölçülür):

1. **Encoder'ı çöz** (`lr/10`) — en olası tek düzeltme; referansların hepsi yapıyor
2. **Girdi çözünürlüğü 224 → 448** (DINOv2 patch 14 ⇒ 256 → **1024 token**;
   TripoSR 512² kullanıyor. Encoder maliyeti 2,3 → ~9 ms, ihmal edilebilir)
3. `dim` 512 → 768 **veya** `depth` 12 → 16
4. Triplane kanal 32 → 40 (TripoSR)

**Blok 2'de bedava gelen ölçüm:** aynı yöntemle 4096 objeye çıkıp eğrinin eğimini
ölç ⇒ "veri büyüyünce eşleme zorlaşıyor mu" sorusu da kapanır.

---

### Blok 3 — Tarife: benimse + sadece bize özgü olanı ablasyonla (1,5 gün)

**Tier 1 — ABLASYON YAPMA, referanstan aynen al.** Bunlar OpenLRM/TripoSR'da
zaten doğrulanmış; 32 objelik gürültülü tezgâhta yeniden türetmek geçmişteki
döngünün ta kendisiydi.

- bf16 (Blok 1.2'de kontrol edildi)
- bölge kırpma ile denetim
- `normalize_camera = true`
- `warmup = 3000`
- encoder çözük + `encoder_lr = lr/10`
- `n_samples = 96`
- arka plan `choice([0.0, 0.5, 1.0])` + sürekli rastgele karışımı
- efektif batch 32 (mikro-batch python döngüsü olduğu için **bellek maliyeti yok**,
  sadece daha az optimizer adımı)
- weight-decay parametre grupları, EMA 0,999

**Tier 2 — GERÇEKTEN bize özgü, ablasyon şart.** Her biri `bench_uids_32` **ve**
`bench_uids_250` üzerinde, tek değişken, çöküş alarmı açık:

| etiket | değişken | hipotez | neden bize özgü |
|---|---|---|---|
| `T_lpips` | 0,25 vs 1,0 | 0,25 sadece 32 objede ölçüldü; ölçekte 1,0 doğru olabilir | referans rejimi 730k obje + batch 1024 |
| `T_mask` | fg-L1 w=1,0 vs **BCE λ=0,05** vs kapalı | bizimki referansa göre ~20× ağır | OpenLRM'de mask loss YOK, TripoSR'da 0,05 |
| `T_dens` | `density_bias ∈ {0; 0,5; 1,0}` × `noise_std ∈ {0; 0,1}` | beyaz çöküşün doğrudan panzehiri (§1.1, K4) | referanslarda bu çöküş görülmüyor (batch 1024) |
| `T_inres` | girdi 224 vs 448 | ViT-S'in token sayısı sınır mı | TripoSR 512² kullanıyor, biz 224 |
| `T_bound` | 0,6 vs 4c-8'de ölçülen sıkı değer | hacmin %34'ü kullanılıyor | `fit` normalizasyonu bize özgü |
| `T_crop` | merkez kırpma vs rastgele bölge | ölçülen kazanç merkez kırpmadandı | — |

6 soru × 2 ölçek ≈ 12 koşu × ~10 dk (hızlanma sonrası) ≈ **2 saat**.
Sonra **kazananların birleşimi** tek koşuda doğrulanır (birleşim ≠ parçaların toplamı).

**Çıktı:** `configs/tarife_v2.yaml` — commit'lenmiş, dondurulmuş tarife.
**Karar:** bu tarife ile ölçek merdivenine girilir.

---

### Blok 4 — Veri (paralel; GPU'suz kısmı BUGÜN başlar)

#### 4a. Diskteki score-2'yi render et (GPU, ~1,8 saat)

```bash
python scripts/build_subset.py --root <glb_root> --opp_score 2 --clean \
  --out dataset/subset_opp_score2.json
RENDER_ENGINE=EEVEE python scripts/run_batch.py \
  --subset dataset/subset_opp_score2.json \
  --output_dir dataset/renders_opp_score3 --workers 8
```

9.977 obje × 0,66 s ≈ **1,8 saat**, +25 GB. Sonuç: **24.571 objelik havuz.**

> `build_subset.py`'a `--opp_score` / `--clean` bayrakları eklenecek (şu an
> sadece LVIS kesişimi var). Objaverse++ bayrakları **string** — `str(x).lower() == "true"`.

#### 4b. 50k için indirme (ağ, GPU'suz — HEMEN başlat, arka planda koşsun)

- Hedef uid listesi: Objaverse++ temiz score≥2 (382k havuz) **eksi** diskteki
  24.571 ⇒ rastgele ~25.500 seç (score 3 öncelikli).
- `objaverse` paketi kurulu değil → `pip install objaverse`.
- Tahmin: ~9,6 MB/obje ⇒ **~245 GB**; 792 GB boş ⇒ sığar.
- **Render sonrası glb'yi sil** ⇒ kalıcı disk maliyeti sadece render (~65 GB).
- Süre: bant genişliğine bağlı (20 MB/s ⇒ ~3,5 saat; 5 MB/s ⇒ ~14 saat).
- Render: 25,5k × 0,66 s ≈ **4,7 saat** (GPU; Blok 3 bitince).

> **Sıralama gerekçesi:** İndirme ağ-bağımlı ve GPU'yu meşgul etmez ⇒ Blok 0–3
> ile paralel gider. Render GPU'yu meşgul eder ⇒ Blok 3 bittikten sonra.
> **Ama:** Objaverse++ makalesi "kaliteli 50k > rastgele 100k" diyor; bizim
> 24,6k'mız zaten kaliteli. **50k'ya çıkmak G3 kapısı geçilmeden anlamsız**
> — 1024 objeyi ezberleyemeyen model 50k'da iyileşmez. İndirme paralel başlasın,
> ama G3 geçilmeden render'a GPU harcanmasın.

#### 4c. Tam veri denetim raporu (`scripts/audit_dataset.py` — YENİ, CPU)

Tek komut, tek rapor (`dataset/audit_report.json` + `audit_report.md`):

| # | kontrol | eşik / çıktı |
|---|---|---|
| 1 | Bütünlük | her uid: `meta.json` + N PNG, açılabilir, 512² RGBA |
| 2 | Meta şeması | `num_views`, `canonical_indices`, her view'de `intrinsic` + `extrinsic` |
| 3 | Kamera tutarlılığı | ‖kamera merkezi‖ = radius (±1e-3); `inv(extrinsic)` tersinir; intrinsic tüm view'lerde aynı |
| 4 | Kamera doğruluğu | bbox merkezinin izdüşümü ≈ alpha silüet centroid'i (görünüm başına, tolerans 3 px) |
| 5 | Alpha kaplama | **16 görünümün tamamında** dağılım (min / medyan / maks); `<0,005` = boş |
| 6 | Kenar taşma | **TÜM veri** (şimdiye kadar sadece 100 obje ölçüldü) |
| 7 | Renk sağlığı | obje piksel ortalaması `<0,05` (karanlık) veya std `<0,02` (tek renk) |
| 8 | **Obje yarıçapı** | alpha silüetinden ters-projeksiyonla üst sınır ⇒ **`bound` kalibrasyonu (E8)** |
| 9 | **Kopya tespiti** | 32×32 gri + alpha imzası ⇒ küme raporu |
| 10 | Objaverse++ dağılımı | score / style / density / is_figure histogramları |
| 11 | **Split** | deterministik seed, **kopya kümeleri tek tarafta**, `train ∩ val ∩ test = ∅` |
| 12 | **Test seti** | 500 obje, **hiçbir tarife kararında kullanılmaz**, sadece final rapor |

**Çıktı:** `dataset/train_list_v2.json` (train / val / **test**) + denetim raporu.
**Karar:** `bound` değeri; elenecek obje listesi.

#### 4d. Gerçek foto değerlendirme seti (E2, manuel, ~1 saat)

20–30 telefon fotoğrafı (masa üstü objeler, farklı ışık). Arka plan silme
(`rembg` ya da manuel), ortala, 224'e getir → `dataset/real_eval/`.
Eğitim boyunca her checkpoint'te gözle bakılacak tek gerçek gösterge.

---

### Blok 5 — Faz C ön-doğrulaması: mesh çıkıyor mu? (0,5 gün) — **eğitimden ÖNCE**

**Neden şimdi:** Nihai ürün `.glb`. Yoğunluk alanı yüzey oluşturmuyorsa
(sisli/bulutsu), eğitim hedefine seyreklik regülarizasyonu eklemek gerekir —
bu **eğitim tarifesini değiştirir**. 27 saatlik koşudan sonra öğrenilecek şey değil.

Test **oracle triplane üzerinde** yapılır (25,04 dB, en iyi hâlimiz):

1. `pip install scikit-image` (trimesh zaten var, skimage yok)
2. `scripts/extract_mesh.py`: triplane → 128³ yoğunluk ızgarası → marching cubes
   (eşik taraması) → `trimesh` temizlik → `.glb`
3. Kabul ölçütleri:
   - kapalı, tek baskın bileşen (bağlı parça sayısı < 20)
   - üçgen sayısı 10k–200k
   - mesh'i yeniden render edip GT ile karşılaştır: **PSNR ≥ 20 dB**
     ⇒ yoğunluk alanı gerçekten yüzeysel
   - eşik dayanıklılığı: σ ∈ [10, 50] arasında hacim %30'dan az değişsin
4. Vertex rengi: triplane'den `rgb_head` örneklenir

**Karar:**

- ✅ geçerse: eğitim tarifesi olduğu gibi kalır
- ❌ geçmezse: kayba **yoğunluk seyreklik/entropi terimi** eklenir
  (λ ≈ 1e-3, `-mean(w·log w)` ya da Cauchy sparsity) ve Tier-2'ye bir ablasyon daha girer

---

### Blok 6 — Ölçek merdiveni: ÖN-KAYITLI kapılar (1,5 gün)

Eşikler **koşudan önce** yazılır ve değiştirilmez. Bir kapı düşerse yukarı
çıkılmaz — geriye dönülüp teşhis edilir. Bu, "sürekli döngü"nün panzehiri.

| kapı | veri | adım (≈epoch) | süre @15 obj/s | eşik | düşerse |
|---|---|---|---|---|---|
| **G1** | 32 | 2.000 | ~4 dk | train top-1 ≥ %90, oran ≤ 0,30 | tarifede hata var |
| **G2** | 250 | 6.000 | ~12 dk | train top-1 ≥ %60, oran ≤ 0,50 | kapasite → Blok 2b |
| **G3** | 2.000 | 20.000 | ~1,2 sa | **val** oran ≤ 0,75, top-1 ≥ 8× şans | kapasite/tarife |
| **G4** | 24.571 | 40.000 (≈52 ep) | ~24 sa | **val** oran ≤ 0,80, PSNR ≥ 18 dB, top-1 ≥ 10× şans | DUR, teşhis |
| **FINAL** | 24,6k / 50k | 60 epoch | 27 / 55 sa | **test** oran ≤ 0,60, PSNR ≥ 20 dB, mesh Blok 5 ölçütlerini geçiyor | — |

*(epoch = veri boyutu ÷ efektif batch; batch 32 varsayıldı)*

**Ara kontroller (her 500 adım):** `psnr`, `top1`, `oran`, **`acc_mean`**,
**objeler arası çıktı std**, grad-norm, LR, ms/obje. Çöküş alarmı tetiklenirse
koşu **otomatik durur**.

---

### Blok 7 — Uzun koşu altyapısı + GO/NO-GO (0,5 gün)

**Altyapı**

1. **Resume testi (E5):** 500 adım koş → süreci öldür → `--resume` → loss eğrisi
   sürekli mi, `step` doğru mu, optimizer/scheduler/EMA durumu geldi mi.
   **Bu test geçmeden uzun koşu başlamaz.**
2. **Gözetmen script'i:** süreç ölürse otomatik `--resume` ile yeniden başlat;
   3 ardışık başarısızlıkta dur ve bildir.
3. **Checkpoint politikası:** `last.pt` + `best.pt` (val oranına göre) + her 10k
   adımda arşiv. ~500 MB/dosya. `git_sha` + config hash gömülü.
4. **Windows:** uyku/hazırda bekletme kapalı, otomatik yeniden başlatan güncelleme
   kapalı, `nvidia-smi --query-gpu=temperature.gpu,power.draw` dakikalık log.
5. **Disk bütçesi:** render 37 + 25 (score2) + 65 (50k) = 127 GB; checkpoint ~5 GB;
   öğretmen bankası (kullanılırsa) 19 GB. Toplam < 160 GB / 792 GB boş ✅

**GO / NO-GO kontrol listesi** — hepsi ✅ olmadan tam eğitim başlamaz:

- [ ] Blok 2 kapısı geçildi (image→triplane öğrenilebiliyor)
- [ ] Blok 5 geçildi (oracle triplane'den kabul edilebilir mesh çıkıyor)
- [ ] G1, G2, G3 kapıları geçildi
- [ ] `tarife_v2.yaml` dondurulmuş ve commit'lenmiş
- [ ] Denetim raporu temiz; `train_list_v2.json` (train / val / **test**) üretildi
- [ ] Kopya kümeleri split'e bölünmemiş
- [ ] Hız ≥ 15 obje/s doğrulandı; tahmini süre hesaplandı
- [ ] Resume testi geçti; gözetmen çalışıyor
- [x] Checkpoint ezme koruması var (§1.6) — kısa koşu uzun koşuyu ezemiyor
- [x] Çöküş alarmı canlı koşuda tetiklendiği doğrulandı (Blok 0)
- [ ] Disk, VRAM, termal marjı doğrulandı
- [ ] Gerçek foto değerlendirme seti hazır
- [ ] 63+ test geçiyor

---

## 7. Senaryolar ve karşılık planları

| senaryo | olasılık | belirti | plan |
|---|---|---|---|
| **A — Tarife düzeltmeleri yetiyor** | ~%55 | Blok 2 kapısı geçilir, G3'te val oranı 0,75'in altına iner | Düz yol: 24,6k × 60 epoch ≈ 27 sa. Beklenen: val ~20–22 dB, tanınabilir şekil + doğru renk |
| **B — Kapasite sınırı** | ~%25 | Blok 2'de `train_rel ≥ 0,5` | Blok 2b merdiveni. Encoder çözme + girdi 448 en olası çözüm. Maliyet ~2–2,5× yavaşlama ⇒ 24,6k × 60 ep ≈ 60–70 sa. Alternatif: veriyi 13k'da tutup epoch'u koru |
| **C — Render kaybı ölçekte kırılıyor** | ~%15 | Blok 2 geçer ama G3/G4 düşer | 3 aşamalı eğitim: öğretmen bankası (24,6k, bf16 ile ~23 sa, fp16'da 19 GB) → distilasyon → render ince ayarı. Daha pahalı ama sinyal yoğunluğu 90× |
| **D — Hiçbiri yetmiyor** | ~%5 | G3 bile düşük | **Kapsamı daralt:** tek kategori ailesi (LVIS mobilya + araç ≈ 3–5k obje), tek girdi görünümü, 64² render. Öğrenme projesi hedefi (uçtan uca çalışan + anlaşılan pipeline) yine karşılanır; SOTA hedefi zaten yok |

**Ek risk kaydı**

| risk | etki | azaltma |
|---|---|---|
| bf16 gerçekten rengi bozuyorsa | 1,8× kayıp | Blok 1.2 kontrollü testi; düşerse fp32'de kalınır, süre 2× |
| İndirme çok yavaş | 50k gecikir | 24,6k ile devam; 50k ikinci koşuda |
| Uzun koşu ortasında donanım/OS kesintisi | saatler | Gözetmen + resume testi (Blok 7) |
| `fit` çerçeveleme yeni objelerde taşarsa | veri kirliliği | Denetim raporu 4c-6 tüm veride kontrol eder |
| 32-obje tezgâhı yanıltıyor | yanlış tarife | Her Tier-2 ablasyonu **iki ölçekte** (32 + 250), çöküş alarmıyla |
| Kopya sızıntısı | val iyimser | 4c-9 dedup + küme-bütün split |
| Kısa koşu uzun koşunun checkpoint'ini ezer | saatler/günler | ✅ kapatıldı (§1.6) + 3 regresyon testi |
| Uzun koşuda aşırı öğrenme | val bozulur | `best.pt` val oranına göre; EMA; erken durdurma değil ama kayıt |

---

## 8. Zaman çizelgesi

| blok | iş | süre | GPU? | paralel? |
|---|---|---|---|---|
| 0 | ~~Ölçüm hijyeni~~ ✅ **bitti** | 0,5 gün | hayır | — |
| 4b | **İndirmeyi başlat** | arka plan | hayır | ✅ hemen |
| 1 | Hız (3× hedef) | 1 gün | evet | — |
| 2 | Kapasite kapısı (distilasyon) | 1 gün | evet | — |
| 5 | Mesh ön-doğrulaması | 0,5 gün | evet (kısa) | ✅ Blok 2 ile |
| 3 | Tarife: benimse + 6 ablasyon | 1,5 gün | evet | — |
| 4a | score-2 render (+9.977 obje) | 1,8 sa | evet | — |
| 4c | Denetim raporu | 0,5 gün | hayır | ✅ Blok 3 ile |
| 4d | Gerçek foto seti | 1 sa | hayır | ✅ |
| 6 | G1 → G4 kapıları | 1,5 gün | evet | — |
| 7 | Uzun koşu altyapısı + GO/NO-GO | 0,5 gün | kısmen | ✅ |
| — | **TAM EĞİTİM** | 27 sa (24,6k) / 55 sa (50k) | evet | — |

**Toplam hazırlık ≈ 5 iş günü**, ardından tek bir uzun koşu.

---

## 9. Kapatılmış sorular — bir daha açma

| soru | cevap | kanıt |
|---|---|---|
| Yeniden render gerekli mi? | Hayır, `fit` + EEVEE seti doğru | AB_ESKI 15,80 dB/%3 vs AB_YENI 20,11 dB/%75 |
| sphere20 (alt/üst) supervision gerekli mi? | Hayır | COV_alt0 27,26 vs COV_alt1 27,74 dB |
| Cross-attention gerekli mi? | Hayır | 17,05 vs 17,06 dB, %80 daha fazla parametre |
| NeRF MLP derin olmalı mı? | Hayır, 2 kat | O1 25,04 vs O2 (10 kat) 21,66 dB |
| Triplane 32² yeter mi? | Hayır, 64² | O4 22,89 vs O1 25,04 dB |
| Aydınlatma değişmeli mi? | Hayır (Faz C'ye kadar) | NeRF bakış yönü almıyor; oracle tavanı 25,04 dB bunu doğruluyor |
| `bound = 0.6` objeyi kesiyor mu? | Hayır | 40 objede yarıçap 0,296–0,497 *(→ `fit` sonrası 4c-8'de yeniden doğrulanacak)* |
| Obje-batch'leme hız kazandırır mı? | Hayır | B=1 zaten compute-bound (§1.2) |
| `torch.compile` değer mi? | Hayır (%2) | §1.2 |
| Öğretmen numarası tek aşamada ölçekleniyor mu? | Hayır | obje başına güncelleme = batch ÷ N; 2635'te 34 |
| Girdi hep kanonik açı olmalı mı? | Evet — Faz C'de kullanıcıdan kamera pozu alamayız, "ön foto = azimuth 0" varsayımı zorunlu | tasarım kararı |

---

## 10. Sonraki oturum için ilk komutlar

```bash
# 1) Blok 0: hiz tezgahi + cokus dedektoru + sabit uid kumeleri
# 2) Blok 4b: indirmeyi arka planda baslat (GPU'suz)
# 3) Blok 1: DataLoader + bf16 + bolge kirpma
# 4) Blok 2: KARAR KAPISI
python scripts/fit_teacher.py --train_list dataset/train_list_opp_score3.json \
  --renders_dir dataset/renders_opp_score3 --n_obj 1024 --steps 25600 \
  --out dataset/lrm_ckpts/teacher_1024_new.pt
python scripts/distill_lrm.py --teacher dataset/lrm_ckpts/teacher_1024_new.pt \
  --train_list dataset/train_list_opp_score3.json \
  --renders_dir dataset/renders_opp_score3 --steps 24000 --batch 8
```
