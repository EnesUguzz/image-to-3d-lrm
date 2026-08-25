#!/usr/bin/env bash
# Gece zinciri: egitim oncesi netlesmesi gereken sorulari sirayla olcer.
# Her blok kendi logunu yazar; bir blok patlarsa zincir DURMAZ (tek istisna:
# duman testi -- o patlarsa son buyuk blok bosa gider, o yuzden bayrakla korunur).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
L=dataset/overnight_logs
mkdir -p "$L"
S=$(date +%s)
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/00_zincir.log"; }

RD=dataset/renders_opp_score3
TL=dataset/train_list_opp_score3.json

say "=== GECE ZINCIRI BASLADI ==="

# ------------------------------------ 0. CERCEVELEME A/B (yeniden-render kapisi)
# A kolu (eski "sphere" cerceveleme, dataset/renders) zaten kosuldu:
#   trainPSNR 15.80dB, top1 %3, mse 0.0325 == ortalama-baseline 0.0325 -> TAM COKUS.
# B kolu ayni 32 uid + ayni bayraklar, ama YENI render dizini. Tek degisken: veri.
say "0) Cerceveleme A/B, B kolu (yeni fit+EEVEE render, ayni 32 uid)"
AB_UIDS=$(python -c "import json,io;u=json.load(io.open('dataset/_ab_uids.json',encoding='utf-8'));u=u if isinstance(u,list) else u.get('train',u);print(','.join(u))")
python scripts/bench_overfit.py --render_dir "$RD" --train_list "$TL"   --uids "$AB_UIDS" --steps 2000 --res 64 --w_lpips 0.25 --eval_every 500   --tag AB_YENI > "$L/00_ab_yeni.log" 2>&1
tail -1 "$L/00_ab_yeni.log" | tee -a "$L/00_zincir.log"
say "0) Karsilastir: A(eski)=15.80dB/top1 %3/oran 1.000  <-> B(yeni)=yukarida"

# ------------------------------------------------- 1. duman testi (fail-fast)
# Yeni val_metrics yolu calisiyor mu? Calismazsa 4 saatlik son blok bosa gider.
say "1) Duman testi: train_lrm + sayisal val (60 adim)"
SMOKE_OK=0
python scripts/train_lrm.py --train_list "$TL" --renders_dir "$RD" \
  --steps 60 --micro_batch 1 --grad_accum 1 --render_res 64 --n_sup 2 \
  --val_every 25 --ckpt_every 100000 --val_n 8 --teacher_subset 0 \
  > "$L/01_duman.log" 2>&1 && SMOKE_OK=1
if grep -q "\[VAL step" "$L/01_duman.log"; then
  say "1) OK -- val metrigi uretiliyor:"; grep "\[VAL step" "$L/01_duman.log" | tail -2 | tee -a "$L/00_zincir.log"
else
  SMOKE_OK=0; say "1) BASARISIZ -- son blok atlanacak. Son 20 satir:"
  tail -20 "$L/01_duman.log" | tee -a "$L/00_zincir.log"
fi
rm -f dataset/lrm_ckpts/last.pt dataset/lrm_logs/val_metrics.jsonl

# ------------------------------------------- 2. sphere20 render (48 obje)
say "2) sphere20 (24 gorunum, ust+alt) ile 48 obje render ediliyor"
RENDER_VIEW_SCHEME=sphere20 python scripts/run_batch.py \
  --subset dataset/subset_viewcov48.json --output_dir dataset/_viewcov \
  --workers 8 > "$L/02_render_sphere20.log" 2>&1
say "2) bitti -- $(ls -d dataset/_viewcov/*/ 2>/dev/null | wc -l) obje klasoru"

# ------------------- 3. KAPSAMA TESTI: alt gorunum bilgi katiyor mu?
# Ayni obje, ayni sayida fit gorunumu (12). Tek fark: B alt kutbu goruyor.
# Ikisi de ayni held-out ALT gorunumlerde olculur.
for IB in 0 1; do
  say "3) Kapsama testi: alt_dahil=$IB"
  python scripts/bench_view_coverage.py --render_dir dataset/_viewcov \
    --n_obj 24 --k_fit 12 --include_bottom $IB --steps 2500 --batch 4 \
    --tag "COV_alt$IB" > "$L/03_kapsama_alt$IB.log" 2>&1
  tail -2 "$L/03_kapsama_alt$IB.log" | tee -a "$L/00_zincir.log"
done

# ------------------------------- 4. TARIFE MATRISI (32 obje, 2000 adim)
# Hepsi ayni veri/seed/program; tek degisken basliktaki sey.
COMMON="--render_dir $RD --train_list $TL --n_obj 32 --steps 2000 --res 64 --w_lpips 0.25 --eval_every 500"
run_m () { # $1=etiket  $2..=ek bayraklar
  local tag=$1; shift
  say "4) matris: $tag"
  python scripts/bench_overfit.py $COMMON --tag "$tag" "$@" \
    > "$L/04_$tag.log" 2>&1
  tail -1 "$L/04_$tag.log" | tee -a "$L/00_zincir.log"
}
run_m M_base
run_m M_enc4    --unfreeze_last 4
run_m M_encall  --train_encoder
run_m M_crop08  --crop 0.8
run_m M_batch8  --batch 8

# ------------------------------------------- 5. matris kazananini sec
say "5) Matris siralamasi (dusuk oran = daha iyi conditioning)"
python - <<'PY' 2>&1 | tee -a "$L/00_zincir.log"
import glob, json, io
rows = []
for f in sorted(glob.glob("dataset/lrm_bench/M_*.json")):
    d = json.load(io.open(f, encoding="utf-8")); fi = d["final"]
    rows.append((fi["mse"]/max(fi["mean_mse"],1e-9), -fi["top1"], fi["psnr"],
                 d["cfg"]["tag"], d["cfg"]))
rows.sort()
for r, nt, ps, tag, cfg in rows:
    print(f"  {tag:10s} oran={r:.3f} top1={-nt:.0%} psnr={ps:.2f}dB")
if rows:
    cfg = rows[0][4]; extra = []
    if cfg.get("train_encoder"): extra.append("--train_encoder")
    elif cfg.get("unfreeze_last"): extra.append(f"--unfreeze_last {cfg['unfreeze_last']}")
    if cfg.get("batch", 4) > 4: extra.append("--grad_accum 8")
    io.open("dataset/overnight_logs/kazanan.txt", "w", encoding="utf-8").write(" ".join(extra))
    print("KAZANAN:", rows[0][3], "| tam egitime eklenecek:", " ".join(extra) or "(yok)")
PY
EXTRA=$(cat "$L/kazanan.txt" 2>/dev/null || echo "")

# --------------------------- 6. TAM VERI EGITIMI (13k) + sayisal val
if [ "$SMOKE_OK" = "1" ]; then
  say "6) Tam veri egitimi basliyor (13k obje) ek bayraklar: ${EXTRA:-yok}"
  python scripts/train_lrm.py --train_list "$TL" --renders_dir "$RD" \
    --steps 16000 --micro_batch 2 --grad_accum 4 --render_res 128 \
    --n_sup 4 --w_lpips 0.25 --val_every 500 --ckpt_every 1000 \
    --val_n 64 --teacher_subset 0 $EXTRA \
    > "$L/06_tam_egitim.log" 2>&1
  say "6) bitti. Son val:"; grep "\[VAL step" "$L/06_tam_egitim.log" | tail -3 | tee -a "$L/00_zincir.log"
else
  say "6) ATLANDI (duman testi basarisiz)"
fi

say "=== ZINCIR BITTI, toplam $(( ($(date +%s)-S)/60 )) dk ==="
