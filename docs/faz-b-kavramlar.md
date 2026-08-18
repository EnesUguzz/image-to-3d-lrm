# Faz B — Kavram Sözlüğü (LRM eğitimi)

Bu dosya Faz B'de geçen terimleri **sade** açıklar. Amaç: kod okurken ya da
konuşurken "bu neydi?" dememek. Sıra, verinin model içindeki yolculuğunu izler.

---

## 1) Girdi ve Encoder (fotoyu sayıya çevirme)

- **LRM (Large Reconstruction Model):** "Büyük Yeniden-İnşa Modeli". 1 veya birkaç
  fotoğrafı alıp **tek seferde** 3B temsil üreten transformer. Difüzyon gibi adım adım
  gürültü temizlemez; doğrudan foto → 3B tahmin eder.

- **Encoder (kodlayıcı):** Fotoğrafı, modelin işleyebileceği **sayı vektörlerine**
  (özellik/feature) çeviren ağ. "Ham piksel" yerine "anlamlı özet" üretir.

- **ViT (Vision Transformer):** Görüntüyü küçük karelere (**patch**) bölüp her kareyi bir
  "kelime" gibi işleyen transformer türü encoder.

- **DINOv2:** Meta'nın devasa veriyle önceden eğitilmiş, çok iyi görsel özellik çıkaran
  hazır bir ViT. Biz sıfırdan eğitmeyiz, **hazır** kullanırız.

- **Frozen / donuk:** Ağırlıkları **eğitim boyunca değiştirilmeyen** parça. DINOv2'yi
  donuk tutarız: sadece özellik çıkarır, öğrenmez. (3k obje onu eğitmeye yetmez zaten.)

- **Patch:** Fotoğrafın küçük bir karesi (ör. 14×14 piksel). ViT her patch'i bir token yapar.

- **Token:** Modelin içinde dolaşan tek bir "birim vektör". Bir patch = bir görüntü token'ı.

- **Embedding:** Bir şeyi (patch, konum, kamera) sabit uzunlukta bir **sayı vektörüyle**
  temsil etme. "Gömme" diye de geçer.

---

## 2) Kamera (fotonun hangi açıdan çekildiği)

- **Kamera pozu (pose):** Kameranın 3B uzaydaki yeri ve baktığı yön.
  - **Extrinsic:** Kameranın konumu + dönüşü (dış parametreler).
  - **Intrinsic:** Lens/sensör bilgisi — odak, görüş açısı (iç parametreler).
  - Bunları Faz A'da her görünüm için `meta.json`'a yazdık.

- **Plücker koordinatları / ışın (ray):** Her pikselden çıkan ışını 6 sayıyla anlatan
  kompakt gösterim (yön + konum). Modele "bu token şu açıdan bakıyor" bilgisini verir.
  Sayesinde 1-4 fotoyu ayırt eder ve nereden bakıldığını bilir.

---

## 3) 3B Temsil (triplane + NeRF)

- **Triplane (üç-düzlem):** 3B bir objeyi **3 adet 2B özellik haritasıyla** (XY, XZ, YZ
  düzlemleri) temsil etme numarası. Tam 3B ızgaradan (voxel) çok daha az bellek yer.
  Uzaydaki bir noktanın özelliğini bulmak için nokta 3 düzleme yansıtılıp okunur.

- **NeRF (Neural Radiance Field):** "Sinirsel Işıma Alanı". Uzaydaki her (x,y,z) noktası
  için **yoğunluk** (orada madde var mı) ve **renk** üreten küçük bir ağ (MLP).
  Triplane'den okunan özellik bu MLP'ye girer.

- **MLP:** Basit katmanlı sinir ağı (çok-katmanlı algılayıcı). Burada: özellik → yoğunluk+renk.

- **Yoğunluk (density):** Bir noktanın ne kadar "dolu/opak" olduğu. Yüzey buradan çıkar.

---

## 4) Render (3B temsilden 2B görüntü üretme)

- **Volume rendering (hacimsel render):** Bir kameradan her piksel için uzaya bir **ışın**
  gönderip, ışın boyunca birçok nokta örnekleyip (yoğunluk+renk) bunları birleştirerek
  o pikselin rengini hesaplama. NeRF'i "fotoğrafa" çeviren adım.

- **Ray marching / örnek (sample):** Işın boyunca alınan nokta sayısı. Çok örnek =
  daha kaliteli ama daha yavaş/bellek yoğun. (Biz ~64 ile başlıyoruz.)

- **Neden render ediyoruz?** Elimizde 3B "doğru cevap" yok; elimizde **2B render fotoları**
  var (Faz A). Model 3B tahmin eder, biz onu tekrar 2B'ye render edip gerçek fotoyla
  karşılaştırırız. Fark = hata = öğrenme sinyali.

---

## 5) Eğitim (modelin öğrenmesi)

- **Input view (girdi görünüm):** Modele **verdiğimiz** foto(lar) — 1-4 kanonik açı.

- **Supervision view (denetim görünüm):** Modelin görmediği, ama tahminini **karşılaştırdığımız**
  cevap-anahtarı fotoları. Model bunları render etmeye çalışır; fark üzerinden öğrenir.
  (Faz A'daki 12 ek görünümün asıl işi bu.)

- **Loss (kayıp/hata):** Tahmin ile gerçek arasındaki farkın sayısal ölçüsü. Küçüldükçe iyi.
  - **L2 / MSE:** Piksel piksel kare farkı. Temel benzerlik.
  - **LPIPS:** "Algısal" fark — insan gözüne göre benzerlik; sonucu keskinleştirir.
  - **Mask / alpha loss:** Objenin silüeti (dolu/boş sınırı) doğru mu.

- **Gradient (gradyan):** Her ağırlığı "hatayı azaltmak için ne yöne itmeli" bilgisi.
- **Backprop (geri yayılım):** Loss'tan geriye doğru tüm ağırlıkların gradyanını hesaplama.
- **Optimizer (AdamW):** Gradyanı kullanıp ağırlıkları güncelleyen algoritma.
- **Learning rate (öğrenme oranı):** Her adımda ne kadar büyük güncelleme yapılacağı.
- **Warmup + cosine decay:** lr'yi başta yavaş artırıp sonra yumuşakça düşürme takvimi.

- **Batch:** Bir güncellemede işlenen obje sayısı.
- **Step (adım):** Bir güncelleme (bir batch işlenip ağırlıklar güncellenir).
- **Epoch:** Tüm veri setinin bir kez baştan sona geçmesi.
- **Gradient accumulation (gradyan biriktirme):** VRAM küçükse, birkaç küçük batch'in
  gradyanını toplayıp tek büyük batch gibi güncelleme. Küçük GPU'da büyük batch etkisi.

- **Overfit (ezberleme):** Modelin eğitim verisini ezberleyip yenisinde başarısız olması.
  - **Validation (doğrulama):** Eğitimde kullanılmayan objelerle ara sınav. Ezber kontrolü.
  - **Overfit testi (bizim sağlık kontrolü):** Bilerek 1-2 objede ezberletiriz; loss ~0'a
    inip render'lar birebir oturuyorsa "model öğrenebiliyor" demektir. Tam eğitimden ÖNCE
    bu geçilmeli — pipeline sağlam mı diye.

---

## 6) VRAM/hız hileleri (16GB'a sığdırma)

- **Mixed precision (bf16):** Sayıları 32-bit yerine 16-bit tutup belleği ~yarıya indirme.
- **Gradient checkpointing:** Ara sonuçları saklamak yerine geri yayılımda yeniden hesaplama;
  bellekten kazanır, biraz yavaşlatır.
- **FlashAttention:** Attention'ı bellek-verimli hesaplayan hızlı çekirdek.
- **Deferred backprop (render'da):** Render'ın geri yayılımını parça parça yaparak bellek pikini düşürme.

---

## 7) Çıktı (Faz C'ye köprü)

- **Marching cubes:** NeRF'in yoğunluk alanından **mesh** (üçgen yüzey) çıkaran algoritma.
- **Mesh:** Üçgenlerden oluşan yüzey modeli — `.glb` olarak kaydedilip three.js'te gösterilir.

---

> Not: Bu proje kişisel öğrenme amaçlı. Terimler kafanı karıştırırsa çekinmeden sor;
> ilerledikçe bu dosyaya ekleme yaparız.
