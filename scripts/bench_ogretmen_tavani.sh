#!/usr/bin/env bash
# ASAMA 2 -- OGRETMEN TAVANI. Tek, kesin deney.
# Belge: docs/KESKINLIK-TESHISI.md §6. Kapilar ORADA, ONCEDEN yazili.
#
# NEDEN TEK DENEY: bugune kadarki keskinlik olcumlerinin hepsi ya n=4 ya
# Nyquist ihlalli ya da 2026-09-02'de duzeltilen kucultme defektiyle kirliydi
# (docs/KESKINLIK-TESHISI.md §3-4). Ikisi birbiriyle celisiyordu. Bu kosu
# hepsinin yerine gecer.
#
# TASARIM
#   sabit : 24 obje (train_list_v2 ILK 24 -- bench_keskinlik uid sirasini
#           assert ediyor), obje basina 500 guncelleme, batch 2 => 6000 adim
#   sabit : denetim BAGLAYICI DEGIL -- bolge 128 <- render U[256,512].
#           Isin maliyeti 128^2'de sabit, detay tavani 512 px, her kolda AYNI.
#           (Uretim tarifesi res 64 tam kare = obje 35 piksel; oradaki duvar
#           denetimdi, bu kosuda o duvar kaldiriliyor ki TEMSIL olculebilsin.)
#   degisken: SADECE tp_res. Kanal 32'de SABIT -- TAVAN probunda tp128 ile kanal
#           birlikte degismisti (confound). n_samples Nyquist'ten ZORUNLU:
#           n >= (FAR-NEAR)*tp_res/(2*bound) = 1.4*tp/1.2
#           tp64 -> 75 (96), tp128 -> 150 (192), tp256 -> 299 (320)
#   w_tv  : dunya olceginde sabit tutulur. TV = mean|delta|, komsu farki ~1/res
#           => ayni bastirma icin w_tv ~ res. 64:0.05  128:0.10  256:0.20
#   K64_notv: TAVAN probunda YALNIZ w_tv degisince kor+kaydir 0.230 <-> 0.138
#           oynadi (n=4). Gercek mi gurultu mu bilinmeden tp_res okunamaz.
#
# SIRA: ucuz ve en bilgilendirici kollar ONCE. K256 bitmese bile 64->128
# merdiveni elde kalir.
#
# ONBILINEN CONFOUND (pesinen yazildi): K256'nin obje basina parametresi
# K64'unkinin 16 kati, guncelleme sayisi ayni. Kaybederse "kapasite yetmedi"
# degil "butce yetmedi" olabilir => son 1000 adimdaki PSNR egimi de raporlanir.
set -u
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
O=dataset/lrm_bench
N=24
ADIM=${ADIM:-6000}          # SMOKE=1 iken kucultulur
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then ADIM=3; N=4; fi

ts(){ date '+%F %T'; }
ON=${ON:-}                  # ckpt adi oneki (duman testinde SMOKE_)

kol(){  # kol <ad> <tp_res> <n_samples> <w_tv>
  local AD="$ON$1" TP=$2 NS=$3 TV=$4
  local OUT="$O/TAVAN2_$AD.pt"
  if [ -f "$OUT" ]; then echo "[$(ts)] $AD ATLANDI (var)"; return 0; fi
  echo "[$(ts)] === $AD ===  tp ${TP}^2 x32 | n_samples $NS | w_tv $TV"
  # --line-buffered SART: yoksa kol bitene kadar tek satir gorunmez.
  python -u scripts/fit_teacher.py \
    --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $ADIM --batch 2 --n_sup 2 \
    --region 128 --render_low 256 --render_high 512 \
    --tp_res $TP --tp_ch 32 --nerf_layers 4 --nerf_hidden 64 \
    --n_samples $NS --w_tv $TV --ray_chunk 32768 \
    --eval_res 256 --ckpt_every 1000 \
    --out "$OUT" 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|detach()"
  echo "[$(ts)] $AD bitti"
}

olc(){  # olc <etiket> <kol adlari...>
  local ET="$1"; shift
  echo "[$(ts)] === KESKINLIK OLCUMU ($ET, $N obje, 256px) ==="
  local ARGS=""
  for AD in "$@"; do
    [ -f "$O/TAVAN2_$AD.pt" ] && ARGS="$ARGS $O/TAVAN2_$AD.pt"
  done
  [ -z "$ARGS" ] && { echo "[$(ts)] $ET: olculecek kol yok"; return 0; }
  python -u scripts/bench_keskinlik.py --train_list "$LIST" --renders_dir "$REND"     --n_obj $N --res 256 --n_gorsel 6 --teachers $ARGS --out "$O/TAVAN2_$ET" 2>&1     | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn"
  echo "[$(ts)] === YUZEY TESHISI ($ET) ==="
  python -u scripts/diag_yuzey.py --n_obj 8 --res 256 $ARGS 2>&1     | grep -v --line-buffered "UserWarning\|warnings.warn\|^  warn"
}

# --- DALGA 1: karar veren kollar. 64 -> 128 merdiveni + w_tv kontrolu. ~2.5 sa
kol K64      64   96  0.05
kol K64_notv 64   96  0
kol K128    128  192  0.10

if [ "$SMOKE" = "1" ]; then echo "[$(ts)] DUMAN TESTI (dalga 1) BITTI"; fi
olc dalga1 K64 K64_notv K128

# --- DALGA 2: "ne kadar ileri gider" noktasi. Tek basina ~5 sa; dalga 1'in
# sonucu buna bagli DEGIL, o yuzden sonra kosuyor.
kol K256    256  320  0.20
olc dalga2 K64 K64_notv K128 K256

echo "[$(ts)] TAVAN2 BITTI"
