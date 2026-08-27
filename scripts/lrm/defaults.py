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

# Isin basi ornek sayisi. Referans: OpenLRM 96, TripoSR 128 (bkz. hazirlik plani
# Tier 1 -- yukseltme Blok 3'un karari, burada kayit altinda).
N_SAMPLES = 48

# Obje hacmi: |x|,|y|,|z| <= BOUND kupu disi tanim geregi BOS.
BOUND = 0.6

# Isin ornekleme araligi (kamera yaricapi 1.4866, obje yaricapi <= ~0.5).
NEAR = 0.8
FAR = 2.2

# Encoder giris cozunurlugu (intrinsic bu olcekte gelir).
INPUT_RES = 224
