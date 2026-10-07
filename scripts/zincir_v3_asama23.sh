#!/usr/bin/env bash
# ASAMA 2+3 -- 1. asama (fit_teacher) AYRI baslatildi ve bitmeli.
#
# NEDEN AYRI BETIK: bash betikleri bayt konumundan artimli okur; kosan bir
# betigi duzenlemek yurutmeyi bozar. Asamalari ayirinca her asamanin tarifesi
# bir onceki koserken guvenle degistirilebiliyor.
#
# 2026-08-29 kararlari (iki bagimsiz uzman incelemesi sonrasi):
#  KABUL (hepsi tespit edilmis defekt):
#   - set_epoch: fit_teacher + distill_lrm'de cagrilmiyordu -> RNG donuktu,
#     her obje 16 goruumun SABIT 4'unden ogreniliyordu. Duzeltildi.
#   - norm/bias'lara weight decay YOK (referans pratigi); olculdu: wd'li
#     DINOv2 norm.weight'i 0.9659'a buzuluyordu, yon degisimi ~1e-5 = saf curume.
#   - --resume: onceki tam kosu bir kez oldu; set -e ile zincir komple biterdi.
#   - eval_photo4: zincirde gercek-foto olcumu HIC yoktu.
#  RED (ikisi de deney, defekt degil -- bu kosuda 4 mimari degisiklik zaten var):
#   - --train_encoder: unfreeze_last 4 son 4 blogu %24 dondurmus ve tek basarili
#     gercek-foto sonucumuz (2 goruum IoU 0.691) TAM o konfigurasyonda alindi.
#     patch_embed + blok 0-7 domain transferini tasiyor, olcum yok. Sonraki A/B.
#   - --mixed_p 0.25: elev egrisi IoU 0.619 (+20) -> 0.489 ([40,55)) -> 0.419
#     ([55,75)); kanonik-disi 12 goruumun %55,5'i telefon-fotosu bandinda.
#     Yariya indirmek Faz C acigini kapatan sinyali yariya indirirdi.
set -euo pipefail
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
CK=dataset/lrm_ckpts
TEACHER=$CK/teacher_1024_v3.pt
DISTILL=$CK/distilled_v3.pt

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] ASAMA 2+3 BASLADI"

# --- 1. asamanin GERCEKTEN bittigini dogrula (yarim dosyaya guvenme) ---
python - "$TEACHER" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert ck.get("done"), f"ogretmen TAMAMLANMAMIS (adim {ck.get('step')}) -- durduruldu"
print(f"ogretmen OK: adim {ck['step']}, {len(ck['uids'])} obje")
PY

if [ -f "$DISTILL" ]; then
  echo "[$(ts)] 2/3 ATLANDI (zaten var): $DISTILL"
else
  echo "[$(ts)] 2/3 distill_lrm basliyor (24000 adim)"
  python scripts/distill_lrm.py --teacher "$TEACHER" --train_list "$LIST" \
    --renders_dir "$REND" --steps 24000 --batch 8 --amp --out "$DISTILL"
  echo "[$(ts)] 2/3 BITTI"
fi

# 3. asama ORTAK bayraklari. micro_batch x grad_accum = efektif 16 SABIT.
# n_samples 48->96 ve NeRF 2->4 kat VRAM'i buyuttu ama OLCULEMEDI (olcum
# penceresi yoktu: 1. asama GPU'yu doldurmustu). Tahmin yerine OOM'u yapisal
# olarak kaldiriyoruz: ilk deneme dusunce AYNI efektif batch'le, daha kucuk
# mikro-batch + gradient checkpointing ile --resume edilir. ckpt_every=2000
# oldugu icin kayip en fazla 2000 adim.
ORTAK=(--train_list "$LIST" --renders_dir "$REND"
       --steps 30800 --workers 6 --amp
       --teacher_subset 0 --unfreeze_last 4 --enc_lr_scale 0.1
       --density_bias 0.0 --bound 0.6 --warmup 3000 --w_lpips 0.25
       --input_crop 0.0 --input_pool mixed --mixed_p 0.5
       --val_every 1000 --ckpt_every 2000 --snapshot_every 5000 --val_n 64
       --ckpt_dir dataset/lrm_ckpts_v3 --tag V3 --resume
       --init_from "$DISTILL")

echo "[$(ts)] 3/3 train_lrm basliyor (micro_batch 8 x accum 2)"
set +e
python scripts/train_lrm.py "${ORTAK[@]}" --micro_batch 8 --grad_accum 2
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "[$(ts)] !! 3/3 kod $RC ile dustu -- micro_batch 4 x accum 4 + grad_ckpt ile yeniden"
  python scripts/train_lrm.py "${ORTAK[@]}" --micro_batch 4 --grad_accum 4 --grad_ckpt
fi
echo "[$(ts)] 3/3 BITTI"

# ---------- bitis degerlendirmesi ----------
echo "[$(ts)] degerlendirme"
CKPT=dataset/lrm_ckpts_v3/last_V3.pt
python scripts/eval_elevation.py --ckpt "$CKPT" --n_obj 40 --tag V3 || true
python scripts/eval_suite.py --ckpt "$CKPT" --split test --no_teacher \
  --n_obj 128 --res 128 --amp --tag V3_test || true
# GERCEK FOTO -- iki uzmanin tartismasi da bunun uzerineydi, olculmuyordu
python scripts/eval_photo4.py --ckpt "$CKPT" --dir dataset/gercek_foto/kumanda4/hazir \
  --n_input 4 --tag V3_kumanda4 --out dataset/gercek_foto/kumanda4 || true
python scripts/eval_photo4.py --ckpt "$CKPT" --dir dataset/gercek_foto/kumanda4/hazir \
  --n_input 1 --tag V3_kumanda1 --out dataset/gercek_foto/kumanda4 || true
python scripts/kosu_raporu.py --tag V3 || true
echo "[$(ts)] ASAMA 2+3 BITTI"
