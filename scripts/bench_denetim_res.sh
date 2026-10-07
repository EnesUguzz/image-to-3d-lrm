#!/usr/bin/env bash
# TEK DEGISKEN: ogretmene gosterilen denetim cozunurlugu (64 / 128 / 256).
#
# SORU (kullanici, 2026-08-31): "ogretmen o objeyi zaten gormus, genelleme
# yapmiyor -- cozunurluk neden bu kadar dussun ki?" Dogru sezgi: dusmesinin
# sebebi model degil TARIFE. fit_teacher varsayilani --res 64, yani diskteki
# 512px render'lar 64px'e kucultulup veriliyor; sonra 256px'te olcup
# "bulanik" diyoruz. Bu merdiven bunu dogrudan gosterir.
#
# Hepsi n_sup 1 (res 256 aksi halde OOM) => ONCEKI 2x2 TARAMASIYLA
# KARSILASTIRILAMAZ, sadece kendi icinde.
set -e
cd "$(dirname "$0")/.."
O=dataset/lrm_bench; mkdir -p "$O"
N=12; ST=2000; B=4
ts(){ echo "[$(date '+%F %T')] $*"; }

for CFG in "64 128" "128 128" "256 128" "256 64"; do
  set -- $CFG; RES=$1; TP=$2
  OUT="$O/den_res${RES}_tp${TP}.pt"
  if [ -f "$OUT" ]; then ts "ATLANDI (var): $OUT"; continue; fi
  ts "fit: denetim ${RES}px, triplane ${TP}^2"
  python scripts/fit_teacher.py --train_list dataset/train_list_v2.json \
    --renders_dir dataset/renders_opp_score3 \
    --n_obj $N --steps $ST --batch $B --n_sup 1 --res $RES --tp_res $TP --out "$OUT"
done

ts "olcum (256px)"
python scripts/bench_keskinlik.py --n_obj $N --res 256 --n_gorsel 6 \
  --out "$O/denetim_res" \
  --teachers "$O/den_res64_tp128.pt" "$O/den_res128_tp128.pt" \
             "$O/den_res256_tp128.pt" "$O/den_res256_tp64.pt" \
             dataset/lrm_ckpts/teacher_1024_v3.pt \
  --students dataset/lrm_ckpts_v3/last_V3.pt
ts "BITTI"
