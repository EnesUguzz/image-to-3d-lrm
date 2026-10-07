#!/usr/bin/env bash
# ORACLE KOSULLANDIRMA -- "darbogaz encoder mi, transformer+head mi?"
#
# NEDEN (docs/KESKINLIK-TESHISI.md §8.6/1):
# Ogrenci ogretmenin triplane'ini uretemiyor (`rel` platosu). Planin yarisi
# H1'e (girdi cozunurlugu 224->448), yarisi H2'ye (token izgarasi 32->48)
# yatirim yapiyor => biri kesinlikle bosa gidecek. Bu deney ayirir.
#
# Encoder ciktisi yerine obje basina SERBEST ogrenilebilir token'lar konur:
# "bu token butcesindeki HERHANGI bir encoder'in verebilecegi en iyi sinyal".
#
#   O0_taban  : normal (donuk DINOv2 @224)
#   O1_serbest: 256 serbest token x 384 = 98.304 sayi/obje (hedefin 0.25 kati)
#   O2_bol    : 1024 serbest token x 384 = 393.216 sayi/obje (hedefin 1.00 kati)
#
# O2 SART: O1 takili kalirsa "transformer eslemiyor" ile "98k token 393k'yi
# tasiyamiyor" ayrismaz. O2 token darbogazini tamamen kaldirir.
#
# ⚠️ OKUMA UYARISI -- GURULTU TABANI:
# `teacher_1024_v3` triplane varyansinin %24,2'si yuksek frekans gurultusu
# (betik basta basiyor). Bu, `rel` icin ~0.24'luk OGRENILEMEZ bir taban demek.
# Kaydedilen plato 0.3029 => hareket alani sadece ~0.06. O yuzden:
#   - birincil okuma `rel_smooth` (yumusatilmis hedefe gore = ogrenilebilir bilesen)
#   - `val_rel` tabana GORE okunur: kazanc = (0.30 - rel) / (0.30 - 0.24)
# Taban tum kollari ESIT etkiliyor, yani KOLLAR ARASI karsilastirma gecerli.
#
# ON-KAYITLI KAPI (kosudan ONCE yazildi):
#   O1 mevcut acigin >=%40'ini kapatirsa  -> darbogaz KOSULLANDIRMA
#         => girdi cozunurlugu 336/448 hakli, once o yapilir
#   O1 ~ O0  ama O2 belirgin daha iyi      -> darbogaz TOKEN BUTCESI
#         => triplane_res 32->48; girdi cozunurlugu bosa gider
#   O2 ~ O1 ~ O0 (ucu de plato)            -> darbogaz TRANSFORMER/HEAD KAPASITESI
#         => ikisi de bosa gider; genislik/derinlik ya da head yeniden tasarim
set -u
cd "$(dirname "$0")/.."

T=dataset/lrm_ckpts/teacher_1024_v3.pt
LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
O=dataset/lrm_bench
LG=dataset/lrm_logs
N=256          # O2 bellegi: 256 x 1024 x 384 x 4B = 0,40 GB (+Adam ~1,6 GB)
STEPS=3000     # 3000 x 8 / 256 = 94 maruziyet/obje = uretim V3 zinciriyle AYNI
BATCH=8
ts(){ date '+%F %T'; }

kol(){  # kol <ad> <oracle_token_sayisi>
  local AD=$1 ORC=$2
  local OUT="$O/ORC_$AD.pt" LOG="$LG/orc_$AD.log"
  if [ -f "$OUT" ]; then echo "[$(ts)] $AD ATLANDI (var)"; return 0; fi
  echo "[$(ts)] === $AD === oracle=$ORC"
  python -u scripts/distill_lrm.py \
    --teacher "$T" --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $STEPS --batch $BATCH --log_every 100 \
    --force_n_input 1 --oracle $ORC --out "$OUT" 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|xFormers\|Using cache" \
    | tee "$LOG"
  echo "[$(ts)] $AD bitti"
}

# force_n_input 1 UC KOLDA DA: serbest token (1,N,384) sekilli, taban kol da
# ayni girdi rejimini gormezse kollar kiyaslanamaz.
kol O0_taban    0
kol O1_serbest  256
kol O2_bol      1024

echo
echo "[$(ts)] ================= ORACLE OZETI ================="
python - << 'PY'
import glob, os, re
satirlar = []
for ad in ("O0_taban", "O1_serbest", "O2_bol"):
    p = f"dataset/lrm_logs/orc_{ad}.log"
    if not os.path.exists(p):
        satirlar.append((ad, None, None)); continue
    son = None
    for L in open(p, encoding="utf-8", errors="replace"):
        if "val_rel=" in L:
            son = L
    if son is None:
        satirlar.append((ad, None, None)); continue
    vr = re.search(r"val_rel=([0-9.]+)", son)
    rs = re.search(r"rel_smooth=([0-9.]+)", son)
    satirlar.append((ad, float(vr.group(1)) if vr else None,
                     float(rs.group(1)) if rs else None))
TABAN = 0.242      # teacher_1024_v3 gurultu payi (betik basta basiyor)
print(f"{'kol':<14} {'val_rel':>9} {'rel_smooth':>11} {'ogrenilebilir kismin %':>23}")
print("-" * 62)
o0 = next((v for a, v, _ in satirlar if a == "O0_taban" and v), None)
for ad, vr, rs in satirlar:
    if vr is None:
        print(f"{ad:<14} {'KOSMADI':>9}"); continue
    pay = ""
    if o0 and o0 > TABAN:
        pay = f"{100 * (o0 - vr) / (o0 - TABAN):+.0f}% (O0'a gore)"
    print(f"{ad:<14} {vr:>9.4f} {rs if rs else float('nan'):>11.4f} {pay:>23}")
print("-" * 62)
print(f"ogrenilemez taban (hedef gurultusu) = {TABAN:.3f}")
print("KAPI: O1 acigin >=%40'ini kapatirsa darbogaz KOSULLANDIRMA;")
print("      O1~O0 ama O2 iyiyse TOKEN BUTCESI; ucu de plato ise KAPASITE.")
PY
echo "[$(ts)] ORACLE BITTI"
