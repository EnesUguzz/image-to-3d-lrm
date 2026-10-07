#!/usr/bin/env bash
# 2x2: denetim cozunurlugu x triplane izgarasi. Oracle (encoder/transformer YOK)
# => olculen sey saf TEMSIL TAVANI.
#
# Neden bu tarama: 2026-08-31 olcumu ogretmenin GT dokusunun %45'ini tuttugunu,
# tek-goruntu 64px tavanininsa %15 oldugunu gosterdi => denetim cozunurlugu
# baglayici duvar DEGIL. Kalan supheli: triplane hucresi (64^2 => 256px
# render'da 3.5 piksel). Onceki "128^2 fayda etmiyor" olcumu GECERSIZDI:
# bench_triplane_fit varsayilani --res 64 idi, orada hucre zaten 0.88 piksel
# (piksel-alti) -- izgarayi buyutmek tanimi geregi hicbir sey veremezdi.
set -e
cd "$(dirname "$0")/.."
O=dataset/lrm_bench; mkdir -p "$O"
N=12; ST=1000; B=4
ts(){ echo "[$(date '+%F %T')] $*"; }

for CFG in "64 64" "64 128" "128 64" "128 128"; do
  set -- $CFG; RES=$1; TP=$2
  OUT="$O/kesk_res${RES}_tp${TP}.pt"
  if [ -f "$OUT" ]; then ts "ATLANDI (var): $OUT"; continue; fi
  ts "fit: denetim ${RES}px, triplane ${TP}^2"
  python scripts/fit_teacher.py --train_list dataset/train_list_v2.json \
    --renders_dir dataset/renders_opp_score3 \
    --n_obj $N --steps $ST --batch $B --res $RES --tp_res $TP --out "$OUT"
done

ts "olcum (256px)"
python scripts/bench_keskinlik.py --n_obj $N --res 256 \
  --teachers "$O/kesk_res64_tp64.pt" "$O/kesk_res64_tp128.pt" \
             "$O/kesk_res128_tp64.pt" "$O/kesk_res128_tp128.pt" \
             dataset/lrm_ckpts/teacher_1024_v3.pt \
  --students dataset/lrm_ckpts/distilled_v3.pt dataset/lrm_ckpts_v3/last_V3.pt
ts "BITTI"
