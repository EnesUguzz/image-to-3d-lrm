#!/usr/bin/env bash
# S3 DENETIM A/B -- BAGIMSIZ OLCUM (guvenlik agi, 2026-09-03)
#
# NEDEN AYRI BIR BETIK:
# `s3_denetim_ab.sh` olcumu kendi icinde yapiyor, AMA checkpoint listesi SABIT.
# B kolu coker ya da yarim kalirsa `bench_keskinlik` eksik dosyada patlar ve
# A'nin gorseli DE uretilmez -- yani 90 dakikalik saglam bir kol da bosa gider.
# Bu betik SADECE DISKTE OLAN checkpoint'leri olcer. Her zaman guvenle kosar.
#
# Ayrica: kosan bir bash betigi DUZENLENEMEZ (bash dosyayi bayt konumundan
# artimli okur, yurutmeyi bozar -- bu projede belgelenmis tuzak). Zincir
# koserken bir duzeltme gerekirse cozum onu duzenlemek degil, bunu kosmaktir.
#
# KULLANIM: bash scripts/s3_olcum.sh
set -u
cd "$(dirname "$0")/.."

CK=dataset/lrm_ckpts_s3
LG=dataset/lrm_logs
DIS=dataset/lrm_ckpts/distilled_v3.pt
ts(){ date '+%F %T'; }

# --- SADECE VAR OLANLARI TOPLA -------------------------------------------
# Sira ANLAMLI: baslangic -> A yorungesi -> B yorungesi -> uretim referansi.
# `bench_keskinlik` ESLESMIS FARK'i ilk ogrenciyi referans alarak basar,
# o yuzden distilled_v3 (BASLANGIC NOKTASI) daima ilk sirada olmali.
SIRA=("$DIS"
      "$CK/snap_S3_A_step001000.pt" "$CK/snap_S3_A_step002000.pt" "$CK/last_S3_A.pt"
      "$CK/snap_S3_B_step001000.pt" "$CK/snap_S3_B_step002000.pt" "$CK/last_S3_B.pt"
      dataset/lrm_ckpts_v3/last_V3.pt)
VAR=(); YOK=()
for f in "${SIRA[@]}"; do
  if [ -f "$f" ]; then VAR+=("$f"); else YOK+=("$f"); fi
done
echo "[$(ts)] olculecek: ${#VAR[@]} checkpoint"
for f in "${VAR[@]}"; do echo "    VAR  $f"; done
if [ ${#YOK[@]} -gt 0 ]; then
  echo "  -- henuz YOK (atlaniyor; egitim bitmemis olabilir):"
  for f in "${YOK[@]}"; do echo "    yok  $f"; done
fi
if [ ${#VAR[@]} -lt 2 ]; then
  echo "[$(ts)] !! en az 2 checkpoint gerekli, olcum yapilmiyor"; exit 1
fi

# GORSEL kollari: sadece son durumlar (sutun sayisi az olsun, gozle okunabilsin)
GORSEL=("$DIS")
[ -f "$CK/last_S3_A.pt" ] && GORSEL+=("$CK/last_S3_A.pt")
[ -f "$CK/last_S3_B.pt" ] && GORSEL+=("$CK/last_S3_B.pt")

echo
echo "[$(ts)] ===== 1/3 SAYISAL (yorunge dahil) ====="
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 0 \
  --ray_chunk 8192 --out "$LG/S3AB_sayisal" \
  --teachers dataset/lrm_ckpts/teacher_1024_v4.pt \
  --students "${VAR[@]}" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|Using cache" \
  | tee "$LG/s3ab_sayisal.log"

echo
echo "[$(ts)] ===== 2/3 GORSEL ====="
# ⚠️ CIKTI ADI `<out>_kare.png`. `--help` metni `_tam.png` ve `_zoom.png`
# diyor ama BU BAYAT (2026-09-03'te duman testiyle dogrulandi). Gercek dosya:
#   dataset/lrm_bench/S3DENETIM_kare.png
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 6 --zoom_px 96 \
  --ray_chunk 8192 --out dataset/lrm_bench/S3DENETIM \
  --students "${GORSEL[@]}" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|Using cache" \
  | tee "$LG/s3ab_gorsel.log"

echo
echo "[$(ts)] ===== 3/3 KAFES ====="
python -u scripts/diag_kafes.py --n_obj 8 --device cuda \
  --teachers dataset/lrm_ckpts/teacher_1024_v4.pt \
  --students "${GORSEL[@]}" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|Using cache" \
  | tee "$LG/s3ab_kafes.log"

echo
echo "[$(ts)] ================= KAPI (kosudan ONCE yazildi) ================="
echo "SAGLAMA once : kor_orta(last_S3_A) < 0.17 olmali."
echo "               A varsayilan tarife; 30800 adimda 0.024'e duser. 3000'de"
echo "               0.199'dan inmediyse deney COK KISA => etiket BELIRSIZ,"
echo "               'denetim sucsuz' DEGIL."
echo "GO           : kor_orta(B) - kor_orta(A) >= +0.05 VE kor_orta(B) >= 0.12"
echo "NO-GO        : ikisi de < 0.08 VE |B-A| < 0.027"
echo "BELIRSIZ     : digeri (tohum gurultu bandi +-0.027, DIS-HEAD'den)"
echo
echo "REFERANS     : distilled_v3 = BASLANGIC (0.199) | last_V3 = A'nin 30800 adimi (0.024)"
echo "               ogretmen v4 0.361 | GT 64px tavani 0.767"
echo
echo "GORSEL       : dataset/lrm_bench/S3DENETIM_kare.png"
echo "[$(ts)] OLCUM BITTI"
