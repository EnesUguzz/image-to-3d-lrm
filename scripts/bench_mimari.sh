#!/usr/bin/env bash
# MIMARI TARAMASI -- tavan probundan SONRA calisir.
#
# Taban = tvB_notv (res 256 denetim, tp 64^2 x32, nerf 4x64, 96 ornek,
# w_tv 0, 4 obje, obje basina 1000 guncelleme). Her kol TEK DEGISKEN.
# Taban zaten fit edildigi icin yeniden kosulmuyor.
#
# Iki kol bu gece yazilan YENI yetenekleri test ediyor:
#   pos_enc : NeRF girdisi simdiye kadar sadece interpolasyonlu triplane
#             ozelligiydi; bilineer => parcali-dogrusal => hucre altinda yapi
#             uretecek mekanizma YOK. Literaturde frekans kodlamasi tam olarak
#             "dusuk uzamsal cozunurlugu telafi" icin kullaniliyor.
#   n_fine  : isin [0.8,2.2]'de ilerliyor ama obje bunun kucuk bir kismi;
#             orneklerin cogu bosluga gidiyor. NeRF hiyerarsik ornekleme.
set -e
cd "$(dirname "$0")/.."
O=dataset/lrm_bench; LG=dataset/lrm_logs
LIST=dataset/train_list_v2.json; REND=dataset/renders_opp_score3
N=4; ST=2000
ts(){ echo "[$(date '+%F %T')] $*"; }

ts "tavan probu bekleniyor..."
while ! grep -q "TAVAN PROBU BITTI" "$LG/tavan.out" 2>/dev/null; do sleep 60; done
ts "tavan bitti, mimari taramasi basliyor"

TABAN=(--n_obj $N --steps $ST --batch 2 --n_sup 1 --res 256
       --tp_res 64 --tp_ch 32 --nerf_layers 4 --nerf_hidden 64
       --n_samples 96 --ray_chunk 32768 --w_tv 0)

kos(){  # kos <ad> <ek bayraklar...>
  local AD=$1; shift
  local OUT="$O/$AD.pt"
  if [ -f "$OUT" ]; then ts "ATLANDI: $AD"; return; fi
  ts "$AD : $*"
  if ! python scripts/fit_teacher.py --train_list "$LIST" --renders_dir "$REND" \
       "${TABAN[@]}" "$@" --out "$OUT"; then
    ts "!! $AD BASARISIZ -- devam"
  fi
}

kos syA_pe6      --pos_enc 6
kos syB_pe10     --pos_enc 10
kos syC_onem     --n_fine 96
kos syD_mlp      --nerf_layers 8 --nerf_hidden 128
kos syE_lpips1   --w_lpips 1.0
kos syF_lpips0   --w_lpips 0.0
kos syG_hepsi    --pos_enc 6 --n_fine 96 --nerf_layers 8 --nerf_hidden 128
kos syH_hepsi128 --pos_enc 6 --n_fine 192 --nerf_layers 8 --nerf_hidden 128 \
                 --tp_res 128 --tp_ch 64 --n_samples 192

ts "olcum"
ARGS="$O/tvB_notv.pt"
for AD in syA_pe6 syB_pe10 syC_onem syD_mlp syE_lpips1 syF_lpips0 syG_hepsi syH_hepsi128; do
  [ -f "$O/$AD.pt" ] && ARGS="$ARGS $O/$AD.pt"
done
python scripts/bench_keskinlik.py --n_obj $N --res 256 --n_gorsel 4 \
  --out "$O/MIMARI" --teachers $ARGS
ts "MIMARI TARAMASI BITTI"
