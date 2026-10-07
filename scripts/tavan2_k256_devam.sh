#!/usr/bin/env bash
# TAVAN2 -- K256 kolunu KALDIGI YERDEN devam ettirir, sonra dalga 2 olcumunu yapar.
#
# NEDEN AYRI BETIK: `bench_ogretmen_tavani.sh`'in `kol()` fonksiyonu, .pt dosyasi
# VARSA kolu ATLIYOR. Yarim kalmis bir kosuda o betigi tekrar calistirmak K256'yi
# atlar ve YARIM checkpoint'i tam egitilmis K64/K128 ile yan yana olcer -- tam da
# 2026-09-01 23:39'da durdurulan sey buydu.
#
# DURUM (2026-09-01 23:40'ta kesildi):
#   TAVAN2_K256.pt  -> adim 2000/6000, done=False, opt+sched icinde  (devam eder)
#   TAVAN2_K256_adim2000_YEDEK.pt -> ayni dosyanin yedegi (kaza icin)
#
# BAYRAKLAR bench_ogretmen_tavani.sh'teki K256 satiriyla BIREBIR AYNI olmali:
# `--resume` optimizer/scheduler'i checkpoint'ten yukler ama MODELI CLI'dan kurar.
# --tp_res/--tp_ch farkli verilirse sekil hatasi (gurultulu, iyi); --n_samples ya da
# --w_tv farkli verilirse SESSIZCE baska bir deney olur (kotu). Degistirme.
#
# TEK BILINCLI SAPMA: --ray_chunk 32768 -> 8192.
# ray_chunk deneyin bir degiskeni DEGIL, yalnizca isinlarin kac parcada
# islendigi; sonuc bit-bazinda ayni (tests/test_render_chunk.py: ileri gecis,
# gradyan ve jitterli CUDA gradyani ozdes). 2026-09-02'deki OOM tam da
# degerlendirme parcasinda oldu: 32768 isin x 320 ornek = 10,5M nokta x 96
# boyut ~ 4 GB tek ayirma. 8192 ile ~1 GB.
set -u
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
O=dataset/lrm_bench
N=24
ts(){ date '+%F %T'; }

echo "[$(ts)] K256 devam ediyor (adim 2000/6000'den; ~4,5 sa)"
python -u scripts/fit_teacher.py \
  --train_list "$LIST" --renders_dir "$REND" \
  --n_obj $N --steps 6000 --batch 2 --n_sup 2 \
  --region 128 --render_low 256 --render_high 512 \
  --tp_res 256 --tp_ch 32 --nerf_layers 4 --nerf_hidden 64 \
  --n_samples 320 --w_tv 0.20 --ray_chunk 8192 \
  --eval_res 256 --ckpt_every 500 --resume \
  --out "$O/TAVAN2_K256.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|detach()"
RC=${PIPESTATUS[0]}
if [ "$RC" != "0" ]; then
  echo "[$(ts)] !! K256 HATA (cikis $RC) -- OLCUM YAPILMIYOR."
  echo "[$(ts)] !! Yarim kolu tam kollarla yan yana olcmek gecersiz tablo uretir."
  exit 1
fi

# TAMAMLANDI MI, KODLA DOGRULA. "bitti" echo'suna guvenme -- 23:39'da tam olarak
# o echo, oldurulen bir surecin ardindan basildi ve logu yaniltici yapti.
python - "$O/TAVAN2_K256.pt" << 'PY'
import sys, torch
tk = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not tk.get("done") or tk.get("step", 0) < tk.get("cfg", {}).get("steps", 10**9):
    sys.exit(f"K256 TAMAMLANMAMIS (adim {tk.get('step')}, done={tk.get('done')})")
print(f"K256 dogrulandi: adim {tk['step']}, done=True")
PY
[ $? -ne 0 ] && { echo "[$(ts)] !! dogrulama basarisiz, olcum yapilmiyor"; exit 1; }

echo "[$(ts)] === KESKINLIK OLCUMU (dalga2, 4 kol, $N obje, 256px) ==="
python -u scripts/bench_keskinlik.py --train_list "$LIST" --renders_dir "$REND" \
  --n_obj $N --res 256 --n_gorsel 6 --out "$O/TAVAN2_dalga2" \
  --teachers "$O/TAVAN2_K64.pt" "$O/TAVAN2_K64_notv.pt" \
             "$O/TAVAN2_K128.pt" "$O/TAVAN2_K256.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn"

echo "[$(ts)] === YUZEY TESHISI (dalga2) ==="
python -u scripts/diag_yuzey.py --n_obj 8 --res 256 \
  "$O/TAVAN2_K64.pt" "$O/TAVAN2_K64_notv.pt" \
  "$O/TAVAN2_K128.pt" "$O/TAVAN2_K256.pt" 2>&1 \
  | grep -v --line-buffered "UserWarning\|warnings.warn\|^  warn"

echo "[$(ts)] DALGA 2 BITTI"
