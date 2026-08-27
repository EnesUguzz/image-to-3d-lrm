# Metrik Şartnamesi

*2026-08-27. Bu dosya, projedeki **her** kalite ölçümünün tek kaynağıdır.
Bir kapı bu dosyada tanımlı olmayan bir sayıyla karar veremez.*

---

## 0. Neden bu dosya var

Blok 2'nin karar kapısı `rel` (triplane özellik uzayında MSE) üzerine yazılmıştı.
`rel` **kalibre değildi**: triplane kanallarının görsel karşılığı yok, bir
kanaldaki 0,1 hata görünmez, başkasındaki aynı hata objeyi yok eder. Eşik
"0,25" bir ölçüme değil sezgiye dayanıyordu.

Ölçüldüğünde ortaya çıkan: aynı checkpoint `rel` cetveliyle "sınırda", render
cetveliyle "tabandan tavana yolun %63'ü + top-1 şansın 117 katı". İki cetvel
farklı karar verdiriyordu.

**Kural: eşik yazmadan önce metriği kalibre et.** Bir metrik ancak
**TAVAN / TABAN / RAKİP** üçgeni içinde okunabiliyorsa kalibredir.

---

## 1. Referanslar ne ölçüyor

| sistem | 2B (render) | 3B (geometri) | not |
|---|---|---|---|
| **LRM** | PSNR, SSIM, LPIPS, CLIP-sim | Chamfer | 100 GSO objesi × 20 görünüm. **Tüm ablasyonlarda aynı 4'lü** |
| **TripoSR** | PSNR/SSIM/LPIPS | **Chamfer + F-score@0,1/0,2/0,5** | 10K nokta; hizalama için rotasyon taraması + ICP |
| **Meshy 7** | *reddediyor* | proportion / distribution / surface detail | Hizalama: öteleme+dönme+**tek ölçek** (oran hatası düzeltilemesin diye) |
| **Tatarchenko+ (CVPR'19)** | — | **F-score@1%** | IoU ve Chamfer'ı eleştiriyor; retrieval baseline'ı zorunlu kılıyor |

İki alıntı bu projeyi doğrudan bağlıyor:

- Meshy 7: *"geometry alignment became a tracked signal inside every training
  cycle rather than an evaluation applied at the end."*
  → **Katman 1+2 her `val` adımında koşar**, sonda değil.
- Tatarchenko: tek-görünüm 3B ağları çoğu zaman rekonstrüksiyon değil
  **sınıflandırma** yapıyor; "ortalama şekil" baseline'ı SOTA'dan ayırt edilemiyor.
  → **RAKİP baseline'ı zorunlu.**

---

## 2. Katmanlar

Tanımların tamamı `scripts/lrm/metrics.py` içinde — `eval_suite.py` ve
`train_lrm.py` **aynı kodu** çağırır (ayrı yazılırlarsa sessizce ayrışırlar).
Regresyon testleri: `tests/test_metrics.py` (21 test).

### Katman 1 — Görüntü *(her `val` adımında)*

| metrik | yön | not |
|---|---|---|
| PSNR | ↑ | ortak dil; tek başına blob çöküşüne **kör** |
| SSIM | ↑ | torch uygulaması, skimage ile uyum test edilmiş (±0,02) |
| **LPIPS-AlexNet** | ↓ | **VGG değil.** Eğitimde LPIPS-VGG kayıp fonksiyonunun parçası; onunla ölçmek kendi kaybımızı metrik diye raporlamak olurdu |
| CLIP-sim (ViT-B/32) | ↑ | "hâlâ aynı nesne mi"; bulanıklığa dayanıklı. Eğitim içinde **kapalı** (VRAM), kapılarda açık |
| siluet IoU | ↑ | `acc` vs GT alpha; bedava |

**İki koşulda ölçülür:** girdi görünümü (yeniden üretim) **ve** yeni görünüm
(gerçek 3B). İkisi ayrışırsa model 3B değil 2B kopya yapıyordur.

### Katman 2 — Çöküş / conditioning *(her `val` adımında)*

| metrik | çöküş imzası |
|---|---|
| top-1 retrieval | şansa (1/N) yakın |
| ortalama-baseline oranı | **1,000** |
| objeler arası piksel std | < 0,010 (`guards.INTER_STD_MIN`) |
| `acc` ortalaması | → 0 (boş sahne; softplus gradyanı ölür) |

### Katman 3 — Geometri *(sadece kapılarda)*

`eval_geometry.py`, GT `.glb`'lere karşı. **F-score@1% birincil** (Tatarchenko),
Chamfer-L1/L2 ikincil (aykırı değere duyarlı), doğruluk/tamlık medyanı,
Normal Consistency, Volume IoU (temkinli — içi dolu objelerde şişer).

**Hizalama bizde serbest** — kameralar biliniyor, tahmin kanonik çerçevede.
Koşulu: GT mesh'in render ile **aynı** normalizasyondan geçmesi. Bu yüzden
`export_norm_mesh.py` dönüşümü trimesh'te yeniden yazmaz, **Blender'da
`render_object.py`'nin kendi fonksiyonlarını** çağırır.
**Geçerlilik kapısı:** GT mesh siluet IoU'su `--min_gt_iou`'nun altındaysa obje
atlanır ve raporda sayılır.

### Katman 4 — Çerçeve *(her raporda, istisnasız)*

| | nedir | nasıl |
|---|---|---|
| **TAVAN** | temsilin verebileceği en iyi | öğretmen triplane'inin render'ı |
| **TABAN** | çöküş çizgisi | tüm GT'lerin piksel ortalaması |
| **RAKİP** | yenilmesi gereken | girdisi en çok benzeyen **başka** objenin GT'si |

Tek bir PSNR sayısı **yorumlanamaz**. Üçlü olmadan rapor yazılmaz.

---

## 3. Dağılım, ortalama değil

Ortalama PSNR, objelerin %20'sinin çöktüğünü gizler. Her metrik
**ortalama + medyan + kötü uç dilim** olarak raporlanır:
yüksek-iyi metriklerde **p10**, düşük-iyi metriklerde (LPIPS, Chamfer) **p90**.

---

## 4. Bilinen kısıtlar *(rapor ederken söylenecek)*

1. **Distilasyon in-sample'dır.** Öğretmen bankası eğitim objeleri için var;
   distilasyonun `val`i diye geçen sayı da eğitim içi. Genelleme = G3'ün işi.
2. **Mesh eşiği GT siluetinden seçiliyor.** Faz C'de elimizde tek maske olacak
   (kullanıcı fotosunun arka planı silinmiş). Geometri sayıları bu yüzden
   **bir miktar iyimser**; tek-maske varyantı Faz C öncesi ölçülmeli. → açık iş
3. **`silhouette_iou` nokta-splat tabanlı.** İnce objelerde kenar etkisiyle
   doğru hizalamada bile ~0,89 verir. Gerçek hizalama hatası ~0,1–0,4 görünür.
4. **Kanonik çerçeve nesne-merkezli.** Her objede azimuth 0 = "ön". Tatarchenko
   bunun ağa bedava semantik ipucu verdiğini, viewer-centered çerçevenin daha
   dürüst olduğunu söylüyor. LRM'in `normalize_camera`'sı bu yüzden açık;
   bizde kapalıydı → Blok 3 Tier 1'de açılıyor.
5. **Tavan da mütevazı.** Öğretmenin kendi F@1%'i 0,486. Mükemmel bir öğrenci
   bile bunu aşamaz — triplane 64²×32 + 2 katlı NeRF sınırı. Faz C kalite
   beklentisi buna göre kurulmalı: kaba şekil ve renk doğru, ince detay değil.

---

## 5. Kapı yazma kuralı

Yeni bir kapı yazarken üç soru **koşudan önce** cevaplanır:

1. **Kalibre mi?** Bu metriğin TAVAN'ı ve TABAN'ı bu koşuda ölçülüyor mu?
2. **Hangi küme?** In-sample mı, held-out mu — ve raporda adı doğru mu?
   (`distill_lrm`'in `val_rel`'i in-sample'dır, adı yanıltıcıydı.)
3. **Komutta var mı?** Ölçüm, koşuyu başlatan komut satırında fiilen çağrılıyor
   mu — yoksa planda sadece bir cümle mi?
   (Ölçek eğrisi planda yazılıydı, komuta girmediği için buharlaşmıştı.)
