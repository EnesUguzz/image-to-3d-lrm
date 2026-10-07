#!/usr/bin/env bash
# E-R -- YENIDEN ORNEKLEYICI DUZELTMESININ TEK DEGISKENLI A/B'si (2026-09-02)
#
# NEDEN:
# `imutil.kucult` / `kucult_rgba` `F.interpolate(mode="area")` kullaniyor.
# `area` = `adaptive_avg_pool2d` ve TAM SAYI OLMAYAN olcek oranlarinda dogru
# alan filtresi DEGIL: kutu sinirlari tam sayiya yuvarlanir, siniri kesen
# piksel bir bin'e tamamen yazilir. Olculdu (512 -> out, rastgele goruntu):
#
#     out 128 (4.000x)  |area - gercek| RMS 0.00000   <- tam sayi: DOGRU
#     out 256 (2.000x)                     0.00000   <- tam sayi: DOGRU
#     out 300 (1.707x)                     0.07866   (ref std 0.136)
#     out 384 (1.333x)                     0.08688   (ref std 0.168)
#     out 448 (1.143x)                     0.10839   (ref std 0.180)
#
# TAVAN2 ogretmen tavani deneyi `--render_low 256 --render_high 512` ile kostu
# ve `crop.sample_render_res` bu araliktan TAM SAYI cekiyor => 257 olasiligin
# sadece 2'si (256, 512) temiz oran veriyor. Bagimsiz denetimin gercek
# render'larda olctugu hata/HF enerjisi ~0.59 -- 2026-09-02 sabahi duzeltilen
# `antialias=False` defektinden (%47-72) BUYUK, ve tam ayni bantta.
#
# Mekanizma da ayni: bin sinirlari goruntu koordinatlarinda sabit, obje
# izdusumu goruumler arasi kayiyor => poza bagli, tutarsiz HF gurultusu =>
# hicbir 3B temsil bunu fit edemez => optimize edicinin cevabi ortalama =
# BULANIK DOKU. Yani tam olarak teshis etmeye calistigimiz semptom.
#
# NE OLDUGUNU DEGIL, NE KADAR ONEMLI OLDUGUNU olcuyoruz: dort kol da AYNI
# bozuk hedefi gordugu icin TAVAN2'nin SIRALAMASI muhtemelen ayakta. Ama
# on-kayitli Kapi A esigi (kor+kaydir >= 0.35) 0.013 kala kalmisti ve ince
# izgarali kollar gurultuyu daha iyi fit edebilir => tp_res etkisinin
# BUYUKLUGU yukari yanli olabilir. Bu betik onu olcer.
#
# TEK DEGISKEN: `OLCEK_YONTEMI=kutu` ortam degiskeni. Kod yolu ayni, bayraklar
# ayni, uid'ler ayni (`np.random.default_rng(0)`), tohum ayni.
#
# NOT: `bilinear + antialias=True` KESIRLIDE 2-5x iyi ama TAM SAYIDA bozuyor
# (out 128: 0.03977, out 256: 0.07421). Bagimsiz denetim onu onerdi; olctum,
# tek yonlu dogruydu. Kullanilan duzeltme kesirli sinirli gercek kutu
# filtresi (`imutil._kutu_kucult`), ikisinde de dogru (RMS ~2e-08).
#
# ON-KAYITLI KAPI (kosudan ONCE yazildi), birincil sutun `kor` (KAYDIRMASIZ --
# `docs/KESKINLIK-TESHISI.md` §8.3: kaydirma bonusu en yanli sayidir ve
# 2026-09-02'de tam o sutundan yanlis karar verildi):
#
#   kor  >= +0.03  (0.158 -> >=0.188)
#       -> TAVAN2 tablosu KIRLI. tp_res karari ASKIDA kalir; dalga 1+2
#          duzeltilmis veri yolunda tekrar kosar (~3 sa). Ogretmen bankasi /
#          TriplaneHead upsample / --tp_res islerinin HICBIRINE baslanmaz.
#   kor  -0.015 .. +0.015
#       -> Merdiven ayakta. tp128 benimsenir, AMA head genislemesiyle:
#          tp128^2x32 = 1.572.864 = ogrencinin gizil tavani (3 x 32^2 x 512),
#          yani TAM DOYGUNLUK / sifir pay. Oyle birakilmaz.
#   kor  <= -0.03
#       -> `area`'nin gurultusu sinyal saniliyormus. Ayri ve daha kotu haber;
#          yine tekrar kosar.
#
# Yan cikti olarak PSNR de kiyaslanir (`ab_kucultme_duzeltmesi.sh` deseninde
# onceki kucultme duzeltmesi +0.75 dB vermisti).
set -u
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
O=dataset/lrm_bench
LG=dataset/lrm_logs
N=24
ADIM=${ADIM:-6000}
R0="$O/TAVAN2_K64.pt"            # dalga 1'den, DISKTE -- yeniden kosulmaz
R1="$O/TAVAN2_K64_kutu.pt"
ts(){ date '+%F %T'; }

if [ ! -f "$R0" ]; then
  echo "[$(ts)] !! $R0 yok -- taban kol olmadan A/B yapilamaz"; exit 1
fi

if [ -f "$R1" ]; then
  echo "[$(ts)] R1 ATLANDI (var: $R1)"
else
  echo "[$(ts)] === R1: K64, OLCEK_YONTEMI=kutu === (R0 = $R0, diskte)"
  # BAYRAKLAR bench_ogretmen_tavani.sh'teki `kol K64 64 96 0.05` ile BIREBIR.
  # Tek bilincli sapma --ray_chunk: 32768 verilirse `renderer.py:75`'teki
  # `isin_sayisi <= chunk` yuzunden parcalama SESSIZCE KAPANIYOR
  # (isin = n_sup 2 x region 128^2 = tam 32768). R0 da parcasiz kostu ve
  # parcalama sonucu bit-bazinda degistirmez (tests/test_render_chunk.py),
  # yalnizca bellegi/hizi etkiler. 0 vererek R0'in rejimini aynen tutuyoruz.
  OLCEK_YONTEMI=kutu python -u scripts/fit_teacher.py \
    --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $ADIM --batch 2 --n_sup 2 \
    --region 128 --render_low 256 --render_high 512 \
    --tp_res 64 --tp_ch 32 --nerf_layers 4 --nerf_hidden 64 \
    --n_samples 96 --w_tv 0.05 --ray_chunk 0 \
    --eval_res 256 --ckpt_every 1000 \
    --out "$R1" 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|detach()" \
    | tee "$LG/er_R1_kutu.log"
  RC=${PIPESTATUS[0]}
  if [ "$RC" != "0" ]; then
    echo "[$(ts)] !! R1 HATA (cikis $RC) -- OLCUM YAPILMIYOR."; exit 1
  fi
  echo "[$(ts)] R1 bitti"
fi

# TAMAMLANDI MI, KODLA DOGRULA -- "bitti" echo'suna guvenme (2026-09-01'de
# oldurulen bir surecin ardindan tam o echo basildi ve logu yaniltici yapti).
python - "$R1" << 'PY'
import sys, torch
tk = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not tk.get("done"):
    sys.exit(f"R1 TAMAMLANMAMIS (adim {tk.get('step')}, done={tk.get('done')})")
print(f"R1 dogrulandi: adim {tk['step']}, done=True")
PY
[ $? -ne 0 ] && { echo "[$(ts)] !! dogrulama basarisiz"; exit 1; }

# OLCUM TEK CAGRIDA: ayni GT, ayni objeler, ayni kod yolu => eslesmis.
# Olcum HER IKI kol icin de VARSAYILAN yolla (`area`) yapilir; degisken
# EGITIM HEDEFI, olcum araci degil. 256px zaten tam 2x, orada `area` dogru.
echo "[$(ts)] === KESKINLIK OLCUMU (E-R, 2 kol, $N obje, 256px) ==="
python -u scripts/bench_keskinlik.py --train_list "$LIST" --renders_dir "$REND" \
  --n_obj $N --res 256 --n_gorsel 6 --out "$O/ER_kutu" \
  --teachers "$R0" "$R1" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
  | tee "$LG/er_olcum.log"

echo
echo "[$(ts)] ================= E-R KAPI ================="
python - << 'PY'
import re
kor = {}
for L in open("dataset/lrm_logs/er_olcum.log", encoding="utf-8", errors="replace"):
    for ad in ("TAVAN2_K64_kutu", "TAVAN2_K64"):
        if L.startswith(ad + " ") or L.startswith(ad + "  ["):
            s = re.findall(r"[-0-9.]+", L.split("]")[-1])
            if len(s) >= 4 and ad not in kor:
                kor[ad] = (float(s[2]), float(s[3]))   # kor, kor+kaydir
            break
r0, r1 = kor.get("TAVAN2_K64"), kor.get("TAVAN2_K64_kutu")
if not (r0 and r1):
    print(f"!! kollar ayristirilamadi: {kor} -- er_olcum.log'a ELLE BAK")
else:
    d = r1[0] - r0[0]
    print(f"R0 (area) kor {r0[0]:.3f}  kor+kaydir {r0[1]:.3f}")
    print(f"R1 (kutu) kor {r1[0]:.3f}  kor+kaydir {r1[1]:.3f}")
    print(f"DELTA kor = {d:+.3f}   (birincil sutun, KAYDIRMASIZ)")
    print("-" * 46)
    if d >= 0.03:
        print("KAPI: TAVAN2 TABLOSU KIRLI. tp_res karari ASKIDA.")
        print("      Dalga 1+2 duzeltilmis veri yolunda tekrar kosar (~3 sa).")
        print("      Ogretmen bankasi / TriplaneHead / --tp_res islerine BASLAMA.")
    elif d <= -0.03:
        print("KAPI: `area`'nin gurultusu sinyal saniliyormus. Yine tekrar kos.")
    elif abs(d) <= 0.015:
        print("KAPI: merdiven AYAKTA. tp128 benimsenir -- AMA head genisletilerek")
        print("      (tp128^2x32 = 1.572.864 = ogrencinin gizil tavani, sifir pay).")
    else:
        print("KAPI: ARA BOLGE (0.015 < |d| < 0.03). On-kayitli karar YOK.")
        print("      Esik disi kalan sonucu 'kismen dogrulandi' diye okuma;")
        print("      n=24'te hata payi olmadan bu buyukluk zaten okunamaz.")
print()
print("GORSELE BAK: dataset/lrm_bench/ER_kutu_kare.png")
PY
echo "[$(ts)] E-R BITTI"
