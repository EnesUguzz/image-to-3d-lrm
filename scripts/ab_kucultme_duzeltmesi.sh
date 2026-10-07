#!/usr/bin/env bash
# TEK DEGISKENLI A/B: kucultme duzeltmesinin (anti-alias + premultiply sirasi)
# uretim ogretmen tarifesindeki etkisi.
#
# A kolu = dataset/lrm_bench/DET_A_taban.pt  (2026-09-01, HATALI veri yolu)
# B kolu = burada fit edilir, KOMUT SATIRI BIREBIR AYNI, tek fark kod duzeltmesi.
# Olcum tek cagrida yapilir => ikisi de AYNI GT'ye karsi olculur.
set -u
cd "$(dirname "$0")/.."
LIST=dataset/train_list_v2.json; REND=dataset/renders_opp_score3; O=dataset/lrm_bench
ts(){ date '+%F %T'; }
echo "[$(ts)] B_duzeltilmis fit basliyor (A ile birebir ayni komut)"
python -u scripts/fit_teacher.py \
  --train_list "$LIST" --renders_dir "$REND" \
  --n_obj 32 --steps 1600 --batch 2 --n_sup 2 \
  --w_tv 0.05 --eval_res 128 --ray_chunk 32768 \
  --res 64 --tp_res 64 --tp_ch 32 --n_samples 96 \
  --out "$O/DET_A_duzeltilmis.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|detach()"
echo "[$(ts)] olcum (ikisi ayni GT'ye karsi)"
python -u scripts/bench_keskinlik.py --train_list "$LIST" --renders_dir "$REND" \
  --n_obj 12 --res 256 --n_gorsel 4 \
  --teachers "$O/DET_A_taban.pt" "$O/DET_A_duzeltilmis.pt" \
  --out "$O/ABFIX" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn"
echo "[$(ts)] AB BITTI"
