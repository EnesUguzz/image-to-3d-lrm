#!/usr/bin/env bash
# DENETIM DETAYI deneyi: ogretmenin keskinlik tavani denetim cozunurlugune mi bagli?
#
# 3 kol, TEK DEGISKENLI zincir:
#   A -> B : yalniz DENETIM detayi degisir (tam kare 64 -> 128 yama <- U[192,384])
#   B -> C : denetim ayni, TEMSIL buyur (tp 64^2x32/n96 -> 128^2x64/n192)
# n_samples serbest DEGIL: tp_res'ten Nyquist ile zorunlu (tp64>=75, tp128>=150).
#
# Esit butce: her kolda obje basina ayni guncelleme (steps*batch/n_obj = 100),
# uretim ogretmeniyle ayni.
#
# 2026-09-01 DERSI: ilk surumde --train_list/--renders_dir VERILMEDI ve uc kol da
# fit_teacher'in o zamanki varsayilani olan OLU sette egitildi => deney cope gitti.
# Artik ACIKCA veriliyor. Varsayilana guvenme.
set -u
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
N=32; STEPS=1600; BATCH=2; NSUP=2
OUT=dataset/lrm_bench
ts(){ date '+%Y-%m-%d %H:%M:%S'; }

kol(){
  local ad="$1"; shift
  if [ -f "$OUT/DET_$ad.pt" ]; then echo "[$(ts)] $ad ATLANDI (var)"; return; fi
  echo "[$(ts)] === $ad === $*"
  # --line-buffered SART: yoksa kol bitene kadar tek satir gorunmez (iki kez yasandi).
  python -u scripts/fit_teacher.py \
    --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $STEPS --batch $BATCH --n_sup $NSUP \
    --w_tv 0.05 --eval_res 128 --ray_chunk 32768 \
    --out "$OUT/DET_$ad.pt" "$@" 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|detach()"
  echo "[$(ts)] $ad BITTI"
}

kol A_taban  --res 64  --tp_res 64  --tp_ch 32 --n_samples 96
kol B_detay  --region 128 --render_low 192 --render_high 384 --tp_res 64  --tp_ch 32 --n_samples 96
kol C_hepsi  --region 128 --render_low 192 --render_high 384 --tp_res 128 --tp_ch 64 --n_samples 192

echo "[$(ts)] === KESKINLIK OLCUMU ==="
python -u scripts/bench_keskinlik.py --train_list "$LIST" --renders_dir "$REND" \
  --n_obj 12 --res 256 \
  --teachers "$OUT/DET_A_taban.pt" "$OUT/DET_B_detay.pt" "$OUT/DET_C_hepsi.pt" \
  --out "$OUT/DETAY" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn"
echo "[$(ts)] HEPSI BITTI"
