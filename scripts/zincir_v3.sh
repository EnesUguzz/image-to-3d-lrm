#!/usr/bin/env bash
# ASAMA 1 -- 3 asamali zincir, TEK betik, sirali, oturumdan BAGIMSIZ.
#
# NEDEN TEK BETIK: her asama ayri baslatilirsa aralarda bekleyen biri gerekir.
# Boyle sirali kosar; bir asama coderse zincir durur (set -e).
#
# NEDEN ZINCIR ZORUNLU (CLAUDE.md): sifirdan render kaybi olcekte CALISMIYOR --
# 1024 objede iki kez olculdu, kol A 6000 adimda 8,8 dB'de kaldi (taban 17,36).
#
# 2026-08-29 degisiklikleri: TriplaneHead rutbe darbogazi kaldirildi, NeRF
# decoder 2->4 katman, n_samples 48->96, girdi havuzu 4 kanonik -> 16 goruum.
set -euo pipefail
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json          # SIZINTISIZ split (v1 degil!)
REND=dataset/renders_opp_score3
CK=dataset/lrm_ckpts
LOG=dataset/lrm_logs
mkdir -p "$LOG"
TEACHER=$CK/teacher_1024_v3.pt
DISTILL=$CK/distilled_v3.pt

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] ZINCIR BASLADI"

# ---------- ASAMA 1/3: ogretmen triplane bankasi ----------
if [ -f "$TEACHER" ]; then
  echo "[$(ts)] 1/3 ATLANDI (zaten var): $TEACHER"
else
  echo "[$(ts)] 1/3 fit_teacher basliyor (1024 obje, 25600 adim)"
  python scripts/fit_teacher.py --train_list "$LIST" --renders_dir "$REND" \
    --n_obj 1024 --steps 25600 --batch 4 --w_tv 0.05 --out "$TEACHER"
  echo "[$(ts)] 1/3 BITTI"
fi

# ---------- ASAMA 2/3: distilasyon ----------
if [ -f "$DISTILL" ]; then
  echo "[$(ts)] 2/3 ATLANDI (zaten var): $DISTILL"
else
  echo "[$(ts)] 2/3 distill_lrm basliyor (24000 adim)"
  python scripts/distill_lrm.py --teacher "$TEACHER" --train_list "$LIST" \
    --renders_dir "$REND" --steps 24000 --batch 8 --amp --out "$DISTILL"
  echo "[$(ts)] 2/3 BITTI"
fi

# ---------- ASAMA 3/3: render ince ayari (TUM veri) ----------
echo "[$(ts)] 3/3 train_lrm basliyor"
python scripts/train_lrm.py \
  --train_list "$LIST" --renders_dir "$REND" \
  --steps 30800 --micro_batch 8 --grad_accum 2 --workers 6 --amp \
  --teacher_subset 0 --unfreeze_last 4 --enc_lr_scale 0.1 \
  --density_bias 0.0 --bound 0.6 --warmup 3000 --w_lpips 0.25 \
  --input_crop 0.0 --input_pool mixed --mixed_p 0.5 \
  --val_every 1000 --ckpt_every 2000 --snapshot_every 5000 --val_n 64 \
  --ckpt_dir dataset/lrm_ckpts_v3 --tag V3 \
  --init_from "$DISTILL"
echo "[$(ts)] 3/3 BITTI"

# ---------- bitis degerlendirmesi ----------
echo "[$(ts)] degerlendirme"
CKPT=dataset/lrm_ckpts_v3/last_V3.pt
python scripts/eval_elevation.py --ckpt "$CKPT" --n_obj 40 --tag V3 || true
python scripts/eval_suite.py --ckpt "$CKPT" --split test --no_teacher \
  --n_obj 128 --res 128 --amp --tag V3_test || true
python scripts/kosu_raporu.py --tag V3 || true
echo "[$(ts)] ZINCIR BITTI"
