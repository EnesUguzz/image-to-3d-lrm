"""Tek kaynak: birden fazla yerde tekrarlanan ve SESSIZCE AYRISABILEN sabitler.

NEDEN VAR:
- `n_samples`: `LRM.__init__` varsayilani 64, tum CLI'lar 48 geciyordu. Hicbir
  kosu 64 kullanmadi ama bayraksiz kurulan bir model (ornegin yeni bir teshis
  script'i) sessizce baska bir modeli olcerdi.
- `near/far/bound`: `fit_teacher.py` hacim render'ini `0.8, 2.2` diye ELLE
  cagiriyordu. Bu degerler degisseydi ogretmen triplane'leri ogrencininkinden
  farkli bir uzayda oturur, distilasyon sessizce bozulurdu.

Deger DEGISTIRMEK bir tarife karari; burada sadece TEK YERE toplaniyor.
Guncel degerler tum ölçülmüş koşularda fiilen kullanilan degerlerdir.
"""

# Isin basi ornek sayisi. Referans: OpenLRM 96 (config.json
# `rendering_samples_per_ray`, small VE base ayni), TripoSR 128.
#
# 2026-08-29: 48 -> 96. Gerekce olculdu: isin adimi (FAR-NEAR)/N = 1.4/48 =
# 0.0292 dunya birimi, triplane hucresi 2*BOUND/64 = 0.0188 => ISIN ADIMI
# HUCREDEN BUYUKTU (Nyquist ihlali): triplane'in tasiyabildigi detay
# ornekleme sirasinda atlaniyordu. 96'da adim 0.0146 < hucre.
# Olculmus maliyet: %8 sure (bench_speed).
N_SAMPLES = 96

# Obje hacmi: |x|,|y|,|z| <= BOUND kupu disi tanim geregi BOS.
BOUND = 0.6

# Isin ornekleme araligi (kamera yaricapi 1.4866, obje yaricapi <= ~0.5).
NEAR = 0.8
FAR = 2.2

# Encoder giris cozunurlugu (intrinsic bu olcekte gelir).
#
# ENCODER DUVARI (2026-09-03): cozunurluk butcesi tablosundaki uc duvardan
# HIC gevsetilmemis olani bu. 224 / DINOv2 patch 14 => 16x16 = 256 patch;
# obje karenin %54'unu kapladigi icin objeye dusen ~8.6 patch. Yani model
# girdi fotografinda dokuyu GORMUYOR. 448'de 32x32 = 1024 patch, obje 17.2.
#
# Maliyet 4x DEGIL: transformer goruntu + triplane token'larini BIRLESTIRIP
# self-attention yapiyor (transformer.py:108), dizi 256+3072=3328 ->
# 1024+3072=4096 ⇒ attention (4096/3328)^2 = 1.51x. Encoder'in kendisi 4x
# ama donuk ve ViT-S.
#
# ⚠️ BU DEGERI DEGISTIREN KOSU, CHECKPOINT KUNYESINE `input_res` YAZAR ve
# `compat.load_lrm` onu geri kurar. Ikisi ayrisirsa Plucker haritasi gercek
# girdiyle SESSIZCE uyusmaz (model.py:52 `scale_intrinsics(K, INPUT_RES, side)`).
import os as _os
INPUT_RES = int(_os.environ.get("GIRDI_RES", "224"))
