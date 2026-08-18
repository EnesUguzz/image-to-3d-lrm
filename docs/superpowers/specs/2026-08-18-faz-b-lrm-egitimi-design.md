# Faz B — LRM Eğitimi (Tasarım / Spec)

**Tarih:** 2026-08-18
**Durum:** Onaylandı, uygulama planına geçiliyor
**Bağlam:** Kişisel öğrenme projesi. Amaç SOTA değil; uçtan uca çalışan + anlaşılan,
16GB VRAM'e sığan, deneme-yanılmayla iyileştirilecek bir multi-view LRM.

İlgili: [Kavram sözlüğü](../../faz-b-kavramlar.md) · [Faz A spec](2026-08-09-faz-a-render-pipeline-design.md)

---

## 1. Hedef

Tek bir transformer-tabanlı **LRM** eğitmek:
kullanıcı **1-4 kanonik açı foto** verir → model **triplane** üretir → **NeRF volume
rendering** ile yeni açılar render edilir → (Faz C'de marching cubes ile mesh çıkarılır).

- Tek model hem 1 hem 4 görüntüyü kabul eder (eğitimde rastgele 1-4 girdi görünüm).
- Referans mimari: OpenLRM-small + multi-view yolu (MeshLRM/M-LRM tarzı: çok-görünüm
  image token'ları + triplane token'ları tek self-attention transformer'da birleşir).
- **Optimum değil, mantıklı bir başlangıç.** Eğitim sonrası kontrol edip iterasyona girilecek.

## 2. Kapsam

**Dahil:** dataloader, girdi augmentation, LRM modeli (encoder→transformer→triplane→NeRF),
volume renderer, loss, eğitim döngüsü, checkpoint/resume, overfit sağlık testi, val görsel izleme.

**Hariç (sonraki fazlar/iterasyonlar):** mesh çıkarımı + web app (Faz C); 512 finetune;
Gaussian splatting; daha büyük dataset (Objaverse++/XL); dağıtık/çoklu-GPU eğitim.

## 3. Girdi Verisi (Faz A çıktısı)

- `dataset/train_list.json` → `{"train": [...2635 uid], "val": [...292 uid]}`
- Her uid: `dataset/renders/<uid>/000..015.png` (512×512 RGBA) + `meta.json`
  (intrinsic + her görünümün extrinsic'i, kanonik indexler, azimuth/elevation).
- Kanonik görünümler: index 000-003 (azimuth 0/90/180/270°, elevation ~20°).
- Supervision görünümleri: 004-015 (halka + rastgele elevation).

## 4. Mimari

### 4.1 Veri akışı
1. Girdi: 1-4 kanonik foto (512→**224** resize) + her birinin kamera pozu.
2. Her foto → **donuk DINOv2 ViT-S/14** → patch token'ları (224/14 = 16×16 = 256 token/görünüm).
3. Her fotonun kamera pozundan **Plücker ışın haritası** (6 kanal, patch çözünürlüğünde)
   üretilir ve token'lara eklenir/kaynaştırılır → model hangi token'ın hangi açıdan
   olduğunu bilir (1-4 görünümü doğal ayırt eder).
4. **Öğrenilebilir triplane token'ları** (3 × 32 × 32 = 3072 adet) foto token'larıyla
   **birlikte** tek transformer'a girer (self-attention). Attention fotolardan 3B'ye bilgi taşır.
5. Transformer çıkışının triplane token'ları → 3 düzlem (32×32, C kanal) →
   transpose-conv ile **64×64**'e büyütülür (C = 32 kanal).
6. **NeRF MLP:** uzaydaki nokta 3 düzleme yansıtılır, bilinear örneklenir, 3 özellik
   birleştirilir → küçük MLP (2-3 katman, 64 gizli) → **yoğunluk (density) + renk (RGB)**.
7. **Volume rendering:** supervision kameralarından ışın gönder, ışın boyunca örnekle,
   entegre et → tahmin RGB + alpha.

### 4.2 Boyutlandırma ("Tiny" — 16GB güvenli ilk koşu)
| Parça | Seçim | Gerekçe |
|---|---|---|
| Encoder | DINOv2 **ViT-S/14 donuk** (~21M, eğitilmez) | 3k obje az; hazır encoder şart. İnternetten 1 kez iner |
| Encoder girdi | 224×224 | ViT-S/14 doğal çözünürlük |
| Transformer | dim **512**, 12 katman, 8 head (~40M eğitilebilir) | OpenLRM-small ölçeği; 16GB'a rahat |
| Triplane | 3 düzlem, 64×64, **32 kanal** | Bellek/kalite dengesi |
| NeRF MLP | 2-3 katman, 64 gizli | Hafif; triplane özelliği zaten zengin |
| Render | **128×128**, ışın başına **64 örnek** | Volume rendering VRAM'in asıl yükü |
| Supervision | adım başına **4 görünüm** | Cross-view tutarlılık için yeterli |

### 4.3 3B temsil kararı
Triplane-NeRF (Gaussian splatting DEĞİL): CLAUDE.md kararı + Faz C'de marching cubes ile
temiz mesh çıkarımına uygun.

## 5. Dataloader

- `train_list.json` okunur; her obje için 16 png + meta.json lazy yüklenir.
- **Her örnek (bir obje) için:**
  - Girdi: 4 kanonikten **rastgele k ∈ {1,2,3,4}** görünüm (k rastgele → 1-4 esnekliği öğrenilir).
  - Supervision: kalan görünümlerden **4** görünüm (girdi olarak seçilmeyenlerden;
    kanonik + supervision karışabilir).
  - Döndürür: girdi görüntüleri (224, augment'li) + girdi pozları; supervision görüntüleri
    (128, temiz) + supervision pozları + alpha maskeleri.
- Resize: encoder 224, supervision hedef 128. Alpha maske olarak korunur.
- Val loader: augmentation kapalı, sabit seed (tekrarlanabilir izleme).

## 6. Augmentation (SADECE girdiye; hedefler temiz)

Kullanıcı fotosu robustluğu için girdi fotolarına uygulanır, supervision hedeflerine ASLA:
- Şeffaf arka planı **rastgele düz renk / gradyan / doku** üzerine bindirme
- Hafif Gauss blur
- JPEG sıkıştırma artefaktı
- Parlaklık / kontrast / renk jitter
- Küçük kırpma + ölçek kayması (kullanıcı ortalaması kusurlu olur)

## 7. Loss

Her supervision görünümü için tahmin vs gerçek:
- **L2 / MSE** (RGB) — temel piksel benzerliği
- **LPIPS** (algısal) — keskinlik
- **Mask / alpha loss** (tahmin opaklık vs GT alpha) — silüet doğruluğu

`loss = w_mse·MSE + w_lpips·LPIPS + w_mask·MASK` (ağırlıklar config'te; başlangıç
w_mse=1.0, w_lpips=1.0, w_mask=0.5 — iterasyonla ayarlanır).

## 8. Eğitim Döngüsü

- **Optimizer:** AdamW, lr ~4e-4, **warmup** (birkaç yüz adım) + **cosine decay**.
- **VRAM stratejisi:** bf16 autocast + gradient checkpointing (transformer) +
  gradient accumulation (mikro-batch 1-2 obje → efektif daha büyük). FlashAttention (varsa).
- **Checkpoint:** her N adımda kaydet; `--resume` ile son checkpoint'ten devam
  (uzun eğitim + PC riski). Optimizer/lr-scheduler state dahil.
- **Val izleme:** her M adımda sabit birkaç val objesini render edip **PNG grid kaydet**
  (Faz A montaj mantığı — gözle ilerleme).
- **Log:** adım, loss bileşenleri, lr, VRAM, hız (obje/sn) → zaman damgalı dosya + konsol.

## 9. Overfit Sağlık Testi (ilk kapı — tam eğitimden ÖNCE)

Tam eğitim başlamadan, bilerek **1-2 objede** ezberletme koşusu:
- Beklenti: loss ~0'a iner, tahmin render'lar GT'ye birebir oturur.
- Geçerse: pipeline (dataloader→model→renderer→loss→backprop) sağlam, model öğrenebiliyor.
- Geçmezse: bug var → tam eğitime girmeden düzeltilir (10 saatlik boş eğitimi önler).
- Bu test kod tabanında kalıcı bir script/komut olarak bulunur.

## 10. Donanım / Ortam

- RTX 5080, 16GB, Blackwell (sm_120) → **CUDA 12.8+ ve güncel PyTorch** kurulmalı
  (eski wheel'ler sm_120 desteklemez). torch henüz KURULU DEĞİL — plan ilk adımı bu.
- Windows 11, PowerShell + Bash. Python 3.10.8.
- Ek bağımlılıklar: torch (cu128), torchvision, `lpips`, DINOv2 (torch.hub ya da timm),
  numpy, Pillow, tqdm. (Not: proje yolu Ğ içerir → tüm dosya IO utf-8.)

## 11. Kod Yapısı (öngörülen)

```
scripts/  (Faz A ile aynı klasör)
  lrm/
    __init__.py
    cameras.py        # Plücker ışın haritası, poz yardımcıları (Faz A camera_poses ile uyumlu)
    dataset.py        # LRMDataset + collate (girdi/supervision seçimi, resize, maske)
    augment.py        # girdi-only augmentation
    encoder.py        # donuk DINOv2 sarmalayıcı
    transformer.py    # triplane token'ları + image token'ları, self-attention bloklar
    triplane.py       # token → 3 düzlem, transpose-conv upsample, nokta örnekleme
    nerf.py           # triplane feature → density+RGB MLP
    renderer.py       # volume rendering (ray sample + integrate), bf16/deferred
    model.py          # uçtan uca LRM (forward: girdiler+pozlar → render)
    losses.py         # MSE + LPIPS + mask
  train_lrm.py        # eğitim döngüsü, checkpoint/resume, val izleme, CLI
  overfit_lrm.py      # 1-2 obje overfit sağlık testi
tests/
  test_cameras.py, test_dataset.py, test_augment.py, test_triplane.py,
  test_renderer.py, test_losses.py, test_model_shapes.py
dataset/
  lrm_ckpts/          # checkpoint'ler
  lrm_val_previews/   # val render PNG grid'leri
```

## 12. Test Stratejisi (TDD)

Ağır GPU eğitimini uçtan uca test edemeyiz; bunun yerine **parça parça** test:
- **cameras:** Plücker ışın şekli/yönü bilinen poz için doğru mu.
- **dataset:** girdi sayısı 1-4 aralığında, supervision girdiyle çakışmıyor, şekiller doğru,
  augmentation sadece girdiye uygulanıyor.
- **triplane:** token→düzlem reshape ve nokta örnekleme şekilleri; bilinen noktada beklenen örnek.
- **renderer:** boş sahne → alpha 0; tek opak nokta → beklenen birikim (küçük sentetik test).
- **losses:** aynı görüntüde loss ~0; maske loss silüet farkında artar.
- **model_shapes:** uçtan uca forward CPU'da küçük boyutlarla çalışır, çıktı şekli doğru.
- **overfit (entegrasyon):** 1-2 objede loss düşüşü (GPU, elle/CI-dışı sağlık kapısı).

## 13. Başarı Ölçütü (bu faz için)

- Overfit testi geçer (loss ~0, render oturur).
- Tam eğitim çöker/patlamaz; loss düşer; val preview'ları zamanla tanınır objelere döner.
- 16GB'a sığar (OOM yok).
- Sonuç "SOTA" olmak zorunda değil — **çalışan + anlaşılan** pipeline + gözle makul 3B tahmin.
  Sonrası: iterasyon (boyut, loss ağırlıkları, augmentation, çözünürlük, daha çok veri).

## 14. Riskler / Açık Noktalar

- **VRAM:** 128px render + 64 örnek + 4 supervision ağır gelebilir → render çözünürlüğü/örnek
  sayısı/supervision sayısı ilk ayar düğmeleri. Gerekirse 64px'e düş.
- **Blackwell/PyTorch:** cu128 wheel kurulumu ilk engel; kurulum doğrulanmadan koda geçilmez.
- **DINOv2 indirme:** ilk çalıştırmada model ağırlığı iner (internet gerekir).
- **Convergence:** 3k obje az; hedef mükemmel değil, "öğreniyor mu" sinyali. Yetmezse veri büyütülür.
