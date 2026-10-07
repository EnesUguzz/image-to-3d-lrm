#!/usr/bin/env bash
# TAVAN PROBU: "bu temsil, hile yapabildigi EN IYI halde nereye kadar cikar?"
#
# Neden: bugune kadar olculen her sey belirli (dar) bir butceyle yapilmis
# fitlerdi. Tavanin kendisi HIC olculmedi. Kullanici hakli olarak sordu:
# 0.52'nin ust sinir oldugunu neye gore soyluyorsun? Soylemiyorum -- 0.52
# sadece "128px kadar keskin" demek. Gercek tavan ampirik olarak bulunur.
#
# Kurulum: 4 obje, her objeye 1000 guncelleme (uretim ogretmeninde 100),
# denetim 256px (olculen en iyi), w_tv 0 (duzlestirme yok).
# Kapasite merdiveni: doyuyor mu, doyuyorsa nerede?
#
# NYQUIST: hucre = 2*bound/tp_res. Isin adimi = 1.4/n_samples. Adim hucreden
# buyukse ince izgara ORNEKLEMEDE atlanir ve tavan YANLIS dusuk olculur.
#   tp  64 -> hucre 0.0188 -> n_samples >=  75   (96 kullaniliyor)
#   tp 128 -> hucre 0.0094 -> n_samples >= 149   (192)
#   tp 256 -> hucre 0.0047 -> n_samples >= 299   (288, ~sinirda)
set -e
cd "$(dirname "$0")/.."
O=dataset/lrm_bench; mkdir -p "$O"
LIST=dataset/train_list_v2.json; REND=dataset/renders_opp_score3
N=4; ST=2000
ts(){ echo "[$(date '+%F %T')] $*"; }

#     ad          tp   ch  ornek  chunk   w_tv
for C in "tvA_simdi   64  32   96   32768  0.05" \
         "tvB_notv    64  32   96   32768  0" \
         "tvC_orta   128  64  192   32768  0" \
         "tvD_maks   256  64  288   32768  0"; do
  set -- $C; AD=$1; TP=$2; CH=$3; NS=$4; CK=$5; TV=$6
  OUT="$O/$AD.pt"
  if [ -f "$OUT" ]; then ts "ATLANDI: $AD"; continue; fi
  ts "$AD  (tp ${TP}^2 x$CH, ornek $NS, w_tv $TV)"
  if ! python scripts/fit_teacher.py --train_list "$LIST" --renders_dir "$REND" \
      --n_obj $N --steps $ST --batch 2 --n_sup 1 --res 256 \
      --tp_res $TP --tp_ch $CH --nerf_layers 4 --nerf_hidden 64 \
      --n_samples $NS --ray_chunk $CK --w_tv $TV --out "$OUT"; then
    ts "!! $AD BASARISIZ -- digerleri devam ediyor"
  fi
done

ts "olcum"
ARGS=""
for AD in tvA_simdi tvB_notv tvC_orta tvD_maks; do
  [ -f "$O/$AD.pt" ] && ARGS="$ARGS $O/$AD.pt"
done
python scripts/bench_keskinlik.py --n_obj $N --res 256 --n_gorsel 4 \
  --out "$O/TAVAN" --teachers $ARGS
ts "TAVAN PROBU BITTI"
