#!/usr/bin/env bash
# V3 zincirinde CAGRILMAYAN 4 olcum. Ckpt = last_V3.pt (adim 30800),
# TAM_asama3 ile ayni adim -> in2/in4/geometri kiyasi adil.
set -e
cd "$(dirname "$0")/.."
CKPT=dataset/lrm_ckpts_v3/last_V3.pt
UID_TEST=004d02243a5b4117afc4baa45eb1eba0
ts(){ echo "[$(date '+%F %T')] $*"; }

for N in 2 4; do
  ts "1.$N) eval_suite test n_input=$N"
  python scripts/eval_suite.py --ckpt "$CKPT" --split test --no_teacher \
    --n_input $N --n_obj 128 --res 128 --amp --tag V3_test_in$N
done

ts "2) eval_geometry test"
python scripts/eval_geometry.py --ckpt "$CKPT" --split test --no_teacher \
  --n_obj 32 --tag V3_geom

ts "3) kanit_asama12 (ogretmen vs 3. asama ogrenci, kendi NeRF'iyle)"
python scripts/kanit_asama12.py --teacher dataset/lrm_ckpts/teacher_1024_v3.pt \
  --student "$CKPT" --student_nerf \
  --out dataset/lrm_val_previews/KANIT_asama13_V3.png

ts "4) extract_mesh (test objesi, 2 girdi)"
python scripts/extract_mesh.py --ckpt "$CKPT" --uid "$UID_TEST" \
  --auto_level --grid 256 --smooth 10 --n_input 2 \
  --out dataset/mesh_out --tag V3_test

ts "EKSIK EVALLER BITTI"
