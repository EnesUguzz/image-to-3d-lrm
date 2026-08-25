# LRM "blob" teşhisi — 2026-08-21

Model, girdi ne olursa olsun aynı bej/kahverengi kutuyu üretiyordu. Bu belge,
kök nedeni **ölçümle** daraltan deney zincirini kaydeder. Her satır tekrar
üretilebilir bir koşuya dayanır; tahmin yok.

## 0. Belirti

15000 adım (≈45 epoch, 2635 obje) sonunda `val_014000.png` ve `diag_train2635.png`:
görülen **train** objeleri bile aynı blob. Loss 1.24 → 0.49 düşmüştü, yani
"eğitim ilerliyor ama çıktı girdiden bağımsız".

## 1. Çökme türü: koşullu ortalama (`scripts/diag_retrieval.py`)

12 train objesi aynı kameradan render edilip karışıklık matrisi çıkarıldı:

| ölçüm | değer |
|---|---|
| doğru objeye MSE (köşegen) | 0.0169 |
| **tüm tahminlerin ortalamasının** her GT'ye MSE | **0.0168** |
| top-1 retrieval | %25 (şans %8.3) |

Modelin objeye özel tahmini, sabit bir "ortalama obje" basmaktan **daha iyi değil**.
Veri setindeki objelerin ortalama rengi `[0.469, 0.418, 0.387]` — modelin bastığı
ton tam olarak bu.

## 2. Veri azlığı değil

Görülen train objeleri de blob. 39.6M parametreli transformer 2635 objeyi 45 epoch'ta
ezberlemeliydi. Ezberleyemiyorsa sorun veride değil.

## 3. Hızlı tezgâh: 32 obje ezberleme kapısı (`scripts/bench_overfit.py`)

Ölçütler: train PSNR, top-1 retrieval, ve **mse'nin "ortalama tahmin" baseline'ına oranı**.

## 4. Eleme zinciri

| # | koşu | ne test etti | sonuç | karar |
|---|---|---|---|---|
| A | joint self-attn (mevcut) | mimarinin ezberleme gücü | 17.06 dB, top1 %25 | ✗ ezberleyemiyor |
| B | cross-attention + kamera modLN | "triplane token'ları görüntüyü okumak zorunda kalsın" | 17.05 dB, top1 %22 | **fayda yok** |
| O1 | oracle: her objeye kendi serbest triplane'i | temsil tavanı | **25.04 dB** | renderer/NeRF/loss/veri sağlam |
| O2 | oracle + 10 katmanlı NeRF MLP | derin MLP gerekli mi? | 21.66 dB | **daha kötü**, 2 katman doğru |
| O3 | O1 tekrarı (hedefler kaydedildi) | tekrarlanabilirlik | 24.96 dB | O1 doğrulandı |
| O4 | oracle, triplane 32² | TriplaneHead deconv darboğazı | 22.89 dB | 2.15 dB maliyet, ana boşluk değil |
| K1 | LRM, tek obje (makas, kaplama %1.7) | tezgâh kontrolü | 20.07 dB, **boş beyaz** | ince objede "boş üret" çöküşü |
| K2 | LRM, tek obje (kaplama %24.6) | tezgâh kontrolü | 20.30 dB ve yükseliyor | tezgâh sağlam |
| **T1** | **aynı encoder+transformer+head, triplane'e doğrudan regresyon** | eşleme öğrenilebilir mi? | **23.94 dB** | **✅ kök neden burada** |

### T1 neden belirleyici

Birebir aynı ağ — donuk DINOv2, aynı joint self-attention gövdesi, aynı TriplaneHead —
oracle triplane'lerine doğrudan regresyonla eğitildiğinde **23.94 dB** üretiyor
(tavan 24.96). Aynı ağ render kaybıyla eğitildiğinde **17.06 dB**.

Elenenler: donuk encoder, transformer mimarisi, TriplaneHead, NeRF MLP, triplane
çözünürlüğü, renderer, kamera konvansiyonu, veri kalitesi, veri miktarı.

**Geriye kalan tek şey: render üzerinden verilen denetim sinyali.**

## 5. Denetim sinyali neden yetersiz

### 5a. Kaybın %85'i arka plan hakkında

2635 koşusunun son adım değerleriyle (`loss=0.4381 mse=0.0176 mask=0.0726 lpips=0.1737`):

| terim | toplam kaybın payı | içindeki obje payı | net obje katkısı |
|---|---|---|---|
| mse (fg_weight=5) | %4 | %37 | %1.5 |
| mask (fg_weight=5) | %17 | %37 | %6.3 |
| lpips (**ağırlıksız**) | %79 | %9 | %7.1 |
| | | **toplam** | **≈%15** |

### 5b. "Boş üret" bedava doğru çıkıyor

Renderer: `rgb = Σw·rgb + (1−acc)·bg_color`. Model `acc=0` basarsa çıktı **tam olarak
bg_color** olur — rastgele arka planı, girdiyi hiç kullanmadan kusursuz tutturur.
Rastgele-bg numarası yalnızca "sabit renk bas" kısayolunu kapatır, "boş bas"ı değil.
K1 (kaplama %1.7) tam olarak buna çöktü.

### 5c. Objeler karede çok küçük

200 objelik örnek: alpha kaplama ortalama **%9.4**, medyan %8.6; objelerin %22.5'i
karenin %5'inden azını kaplıyor. Render normalizasyonu objeyi bbox **yarı-köşegeni**
(= sınırlayıcı küre) 0.5 olacak şekilde ölçekliyor; küp bir obje bu kürede ancak
0.577 kenar alıyor.

### 5d. Sinyal yoğunluğu farkı

Distilasyon adım başına obje başına 3×32×64×64 = **393k** hedef skaler veriyor.
Render ise 4 görünüm × 4096 ışın × %9 kaplama × 3 kanal ≈ **4.4k** skaler.
Yaklaşık **90 kat** fark. Render yolu doğası gereği sinyal-aç.

## 6. Düzeltilen somut hatalar

- `dataset.py`: `random.Random(base_seed*1_000_003 + idx)` her epoch aynı seed'i
  veriyordu → her objenin girdi görünümü, supervision görünümleri ve augmentation'ı
  sonsuza dek sabitti (16 render'ın 12'si hiç kullanılmıyordu, augmentation sahteydi).
  Düzeltildi + regresyon testleri (`test_dataset.py`).

## 7. Hedef mi bozuk, optimizasyon mu? (`E1`)

T2 = distilasyonla eğitilmiş ağırlıklar (23.50 dB). Bu ağırlıklardan başlayıp
**render kaybıyla** devam edildi (`--init_from` + `--init_nerf`):

| adım | train PSNR | top-1 | render kaybı |
|---|---|---|---|
| 0 | **24.10 dB** | %100 | **0.2317** |
| 100 | 23.51 dB | %100 | 0.1478 |
| 300 | 22.56 dB | %100 | 0.1807 |
| 400 | 22.80 dB | %100 | 0.2034 |

Kritik nokta: iyi çözümün render kaybı **0.2317**; A'nın 2000 adım sonunda ulaştığı
blob çözümünün kaybı **0.4658**. Yani **render hedefi iyi çözümü doğru şekilde tercih
ediyor** ve o çözüm kararlı (blob'a çökmüyor).

> **Sonuç: kayıp fonksiyonu bozuk değil. Model, sıfırdan başlayınca "ortalama obje"
> yerel minimumuna düşüp orada kalıyor.** Klasik bir optimizasyon-manzarası problemi.

## 8. Denenen düzeltmeler (32 obje, aynı 2000 adımlık program)

Ölçüt: **top-1 retrieval** (conditioning gücü) ve **mse / ortalama-baseline oranı**.

| varyant | değişken | PSNR | top-1 | oran |
|---|---|---|---|---|
| A | — (baseline, w_lpips=2.0) | 17.06 | %25 | 0.832 |
| B | cross-attention + modLN | 17.05 | %22 | 0.885 |
| C2 | fg_weight 5 → 20 | 17.59 | %38 | 0.701 |
| C1 | w_lpips → 0.0 | 18.09 | %66 | 0.629 |
| R4 | w_lpips 0.5 + fg 20 | 16.69 | %28 | 0.864 |
| **R3** | **w_lpips → 0.25** | **18.83** | **%62** | **0.556** |

### LPIPS ağırlığı

`w_lpips=2.0` LRM'in değeri, ama o 730k obje + batch 1024 rejimi için. Bizim
rejimde LPIPS baskın olunca ("makul genel doku"yu ödüllendirdiği için) model
ortalama-obje havzasında kalıyor. **0.25 optimum** — 2.0'ın 8'de biri.
`train_lrm.py` varsayılanı buna çekildi.

### Kararsızlık uyarısı

`w_lpips=0` + yüksek LR kararsız (R1/R2 ıraksadı): MSE-only yüksek frekanslı
detay isterken LPIPS'in yumuşatması yok. 0.25 hem conditioning'i açıyor hem
kararlılığı koruyor.

### Ama yetmiyor

R3'ün çıktısı artık objeye özgü (kara kedi koyu, şeftali turuncu) ama şekiller
bulanık. Loss ayarı bizi "aynı blob" → "girdiye bağlı blob" noktasına taşıdı,
kanıt kapısına değil.

## 9. Çözüm: öğretmen triplane + öğrenci LRM (`scripts/bench_teacher.py`)

E1'in mantığı: iyi havzaya bir kez girildiğinde render kaybı çözümü koruyor ve
iyileştiriyor. Serbest triplane (O1) aynı render kaybıyla 25 dB'ye çıkıyor çünkü
araya transformer girmiyor. O halde ikisini **aynı koşuda** çalıştır:

- **öğretmen**: obje başına serbest triplane, render kaybı → hızla doğru çözüme gider
- **öğrenci**: LRM; öğretmenin triplane'ine distilasyon (öğretmen varyansına
  normalize) + kendi render kaybı
- öğretmen eğitimin %75'inde kapanır, öğrenci saf render kaybıyla devam eder

| adım | TCH1 (w_distill=1) | TCH2 (w_distill=10) | A (baseline) |
|---|---|---|---|
| 250 | 17.81 dB / %34 | 18.19 dB / %34 | 12.9 dB / %3 |
| 500 | 22.17 dB / **%100** | 22.55 dB / %94 | 15.5 dB / %3 |
| 2000 (final) | — | — | 17.06 dB / %25 |

250. adımda baseline'ın 2000 adımlık finalini geçiyor. `w_distill` 1 ile 10 arası
fark yok → yöntem parametre seçimine duyarsız.

### Ölçekleme kısıtı

Öğretmen triplane'i obje başına 1.57 MB (3×32×64×64 fp32). 2635 obje = 4.1 GB,
Adam durumlarıyla 12.4 GB → 16 GB karta modelle birlikte sığmaz. Çözüm: öğretmen
fazını bir **alt kümede** koş (512 obje ≈ 2.4 GB), sonra tüm veri setinde render
kaybıyla devam et.

## 10. Faz A — render ayarları (2026-08-24)

**Düzeltme:** 2026-08-21'de "bbox çerçeveleme 1.87× kazanç, taşma yok" diye
kaydedilmişti. 6 objelik örneklem yetersizmiş: 100 objede `bbox` modunda
render'ların **%5'i kenardan taşıyor**. Kazanç kırpmadan geliyormuş.

| mod | kaplama | taşan render | taşan obje |
|---|---|---|---|
| sphere (eski) | 0.0843 | %0.00 | 0/100 |
| bbox | 0.1695 | **%5.00** | 13/100 |
| **fit (yeni varsayılan)** | **0.0899** | **%0.00** | **0/100** |

`fit`: bbox normalize edilir, sonra objenin 16 kamera görünümündeki gerçek izdüşüm
genişliği ölçülüp kareye tam oturtulur (birkaç yinelemeli; her yinelemede yeniden
merkezlenir — `obj.scale` objenin kendi orijinine göre ölçeklediği için merkezleme
atlanırsa sahne kayıp model gereksiz küçülür).

Sonuç: doğru çerçevelemede kazanç ~%7. 16 görünümlü kamera halkasında en kötü
izdüşüm zaten sınırlayıcı küreye yakın.

**Asıl hız kazancı motorda:** CYCLES → EEVEE_NEXT, 384 render çiftinde alpha farkı
0.00001 / PSNR 37.4 dB. 14.594 obje: 14.9 saat → **~2.7 saat** (EEVEE + 8 worker).
