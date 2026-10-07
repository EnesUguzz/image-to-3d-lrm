#!/usr/bin/env bash
# 2026-08-31 gunduz kuyrugu. Kullanici aksam donecek; GPU bos kalmasin.
# Her is TEK DEGISKENLI ve esigi kosmadan ONCE yazilmis (docs/SIRADAKI-ISLER.md).
set -e
cd "$(dirname "$0")/.."
ts(){ echo "[$(date '+%F %T')] $*"; }
O=dataset/lrm_bench; LG=dataset/lrm_logs
LIST=dataset/train_list_v2.json; REND=dataset/renders_opp_score3

# ---- 0) once kosan merdiven bitsin (tek GPU, cakismasin) ----
ts "merdiven bekleniyor..."
while ! grep -q "BITTI" "$LG/den_res.out" 2>/dev/null; do sleep 60; done
ts "merdiven bitti, kuyruk basliyor"

# ---- 1) KAPASITE TARAMASI (oracle: encoder/transformer YOK) ----
# Soru: ogretmenin keskinlik tavanini ne sinirliyor?
# Merdiven denetim cozunurlugunu, 2x2 izgarayi test etti. Test EDILMEMISLER:
#   bound (kalibre edildi 0.552, HIC UYGULANMADI -- bedava 1.28x hacim)
#   NeRF MLP kapasitesi (4x64; TripoSR 10 katman)
#   triplane kanal sayisi (32; hucre BASINA kapasite)
# Hepsi denetim 128px'te, tek degisken. Esik: kor+kaydir'da +0.010.
for CFG in "taban 32 4 64 0.6" "bound 32 4 64 0.552" "mlp 32 8 128 0.6" "ch 64 4 64 0.6"; do
  set -- $CFG; AD=$1; CH=$2; LY=$3; HD=$4; BN=$5
  OUT="$O/kap_$AD.pt"
  if [ -f "$OUT" ]; then ts "ATLANDI: $OUT"; continue; fi
  ts "kapasite/$AD  (ch $CH, nerf ${LY}x${HD}, bound $BN)"
  python scripts/fit_teacher.py --train_list "$LIST" --renders_dir "$REND" \
    --n_obj 12 --steps 2000 --batch 4 --n_sup 1 --res 128 --tp_res 64 \
    --tp_ch $CH --nerf_layers $LY --nerf_hidden $HD --bound $BN --out "$OUT"
done
ts "kapasite olcumu"
python scripts/bench_keskinlik.py --n_obj 12 --res 256 --n_gorsel 6 \
  --out "$O/kapasite" \
  --teachers "$O/kap_taban.pt" "$O/kap_bound.pt" "$O/kap_mlp.pt" "$O/kap_ch.pt"

# ---- 2) ENCODER A/B (docs/SIRADAKI-ISLER.md §1.1, kuyrugun 1. maddesi) ----
# last_V3'ten devam, 10000 adim, TEK DEGISKEN: unfreeze_last 4 -> train_encoder.
#
# GUC UYARISI (kosmadan once yazildi): 10000 adim = obje basina 13 maruziyet,
# V3'un 40'ina karsi. Devam kosusu oldugu icin sifirdan kosunun 150 esigi
# gecerli degil, ama yine de SINIRLI. => NULL SONUC "fark yok" DEMEK DEGIL,
# "bu butcede fark gorunmedi" demektir. Ayrilma val egrisinde (her 500 adim)
# erken gorunmezse, karar bir sonraki tam kosuya birakilir.
#
# --lr 1e-4 benim secimim, olculmedi (V3 4e-4 ile basliyordu ama o sifirdan
# degil distilasyondan devamdi). Iki kol da AYNI lr'yi kullandigi icin A/B
# gecerli; sadece mutlak seviye bu secime bagli.
# Kontrol kolu da devam kosusu olmali; V3'un kendisiyle kiyas gecersiz olurdu
# (devam etmek basli basina bir degisiklik).
ORTAK=(--train_list "$LIST" --renders_dir "$REND"
       --steps 10000 --warmup 500 --lr 1e-4 --seed 0
       --micro_batch 8 --grad_accum 2 --workers 6 --amp
       --teacher_subset 0 --enc_lr_scale 0.1
       --density_bias 0.0 --bound 0.6 --w_lpips 0.25
       --input_crop 0.0 --input_pool mixed --mixed_p 0.5
       --val_every 500 --ckpt_every 2000 --snapshot_every 4000 --val_n 64
       --init_from dataset/lrm_ckpts_v3/last_V3.pt)

for KOL in K E; do
  if [ "$KOL" = "K" ]; then VAR=(--unfreeze_last 4); else VAR=(--train_encoder); fi
  if [ -f "dataset/lrm_ckpts_ab/last_AB$KOL.pt" ]; then ts "ATLANDI: kol $KOL"; continue; fi
  ts "encoder A/B kol $KOL: ${VAR[*]}"
  python scripts/train_lrm.py "${ORTAK[@]}" "${VAR[@]}" \
    --ckpt_dir dataset/lrm_ckpts_ab --tag AB$KOL
done

ts "A/B degerlendirmesi"
for KOL in K E; do
  C="dataset/lrm_ckpts_ab/last_AB$KOL.pt"
  [ -f "$C" ] || continue
  echo "########## kol $KOL ##########"
  python scripts/eval_suite.py --ckpt "$C" --split test --no_teacher \
    --n_input 1 --n_obj 128 --res 128 --amp --tag AB${KOL}_test || true
  python scripts/eval_elevation.py --ckpt "$C" --n_obj 40 --tag AB$KOL || true
  python scripts/eval_photo4.py --ckpt "$C" --n_input 1 --tag AB${KOL}_foto1 \
    --out dataset/gercek_foto/kumanda4 || true
  python scripts/kosu_raporu.py --tag AB$KOL || true
done
ts "KUYRUK BITTI"
