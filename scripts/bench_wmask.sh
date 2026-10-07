#!/usr/bin/env bash
# w_mask TARAMASI -- "sifirdan render kaybi gercekten olcekte mi calismiyor?"
#
# NEDEN (docs/KESKINLIK-TESHISI.md §8.5):
# Uc asamali zincirin TEK gerekcesi "sifirdan render kaybi olcekte calismiyor"
# (kol A: 1024 obje, 6000 adim, 8,8 dB, acc~0.50'de cakili). Ama o kosuda
# kaybin ~%66'si `mask` (alfa L1) terimiydi ve referans tarifede (LRM/OpenLRM)
# AYRI bir alfa-L1 terimi YOK -- yani "referanstan 5 noktada sapiyoruz"
# listesinde OLMAYAN 6. sapma. Terim hic tek degiskenli izole edilmedi.
#
# Bu terimin iki karari noktasi var ve ikisi de cokus: acc->0.50 (dolu kup,
# kol A) ve acc->0 (bos sahne, kol D). Arasinda bir egim oldugunu gosteren
# hicbir kosu yok.
#
# EGER w_mask=0 ile sifirdan render kaybi CALISIYORSA:
#   uc asamali zincir, ogretmen VRAM duvari, CPU sayfalama isi ve
#   tp_res-ogretmen bagi TUMUYLE gereksizlesir.
# Bu yuzden proje kuyrugundaki en yuksek beklenen-deger/maliyet oranli deney.
#
# ⛔ KURULUM DUZELTMESI (2026-09-02, kosarken yakalandi):
# Bu betik once 'kol A'nin BIREBIR rejimi' iddiasindaydi. O IDDIA TERK EDILDI --
# kol A birebir tekrar EDILEMEZ, cunku o kosu `normalize_cams` ile calisti ve
# `canonicalize` O TARIHTE BOZUKTU (meta.json extrinsic'leri ortonormal degil;
# 2026-08-27 gecesi duzeltildi). Kol A'yi tekrar etmek duzeltilmis bir hatayi
# geri getirmek olurdu. Sorulan soru bu yuzden guncellendi:
#   'BUGUNKU kodla, sifirdan render kaybi 1024 objede calisiyor mu ve
#    w_mask terimi bunda belirleyici mi?'
# Kol A ile ORTAK tutulanlar (bunlar tekrar edilebilir ve onemli):
#   1024 obje, 6000 adim, micro_batch 8 x grad_accum 2 (=16), lr 4e-4,
#   --amp (bf16). AMP ozellikle: varsayilani False ve ilk baslatmada fp32
#   kosmaya basladi -- oysa hem kol A bf16'ydi hem de benimsenen uretim
#   tarifesi bf16+density_bias. Ustelik fp32 ~1.8x yavas.
# Kol A'dan AYRILANLAR (bilincli): normalize_cams kapali (bozuk kodla acilmisti),
#   unfreeze_last 0.
#
# ESKI GEREKCE (tarihsel): kol A'nin rejimi -- cunku basarisizlik
# 2026-09-02 DUZELTME: --micro_batch 8 --grad_accum 2 EKLENDI.
# Varsayilanlar 2x4 = efektif batch 8; kol A ise 8x2 = 16 kosmustu
# (train_20260827_113101.log). Yani 'tek degisken w_mask' iddiasi YANLISTI:
# batch de yariya inmis, obje basina maruziyet 94 -> 47 dusmustu. Bir kol
# calissaydi w_mask'a mi batch'e mi baglayacagimizi ayirt edemezdik.
# (--lr zaten varsayilan 4e-4 = kol A ile ayni; ekleme gerekmiyor.)
# ORADA gozlendi; 256 objede test etmek "kucukte zaten calisiyordu" tuzagina duser.
# Tek degisken w_mask. --density_bias 1.0 UC KOLDA DA acik (olculmus panzehir:
# acc->0 cokusunde softplus gradyani oluyor, K2 kolu 15,52 -> 18,42 dB).
#
# --teacher_subset 0 SART: varsayilan 512 ve --dry_run bunu yakaladi --
# ortak ogretmen fazi Blok 2'de terk edildi, acik kalirsa deney baska bir sey olur.
#
# ON-KAYITLI KAPI (kosudan ONCE yazildi), 6000. adimda:
#   PSNR > 15 dB VE acc 0,05-0,35 arasi VE top-1 > sans   -> KOL CALISIYOR
#   acc ~ 0,50 sabit                                       -> dolu kup cokusu
#   acc < 0,01                                             -> bos sahne cokusu
#   PSNR virgulden sonra iki hane sabit (3 olcum)          -> donmus (t1_taban gibi)
# Herhangi bir kol CALISIYORSA zincir sorgulanir; hicbiri calismiyorsa
# "sifirdan calismiyor" gerekcesi NIHAYET tek degiskenli olarak dogrulanmis olur.
set -u
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
CK=dataset/lrm_ckpts_wmask
LG=dataset/lrm_logs
N=1024; STEPS=6000
ts(){ date '+%F %T'; }
mkdir -p "$CK"

kol(){  # kol <ad> <w_mask>
  local AD=$1 WM=$2
  local LOG="$LG/wmask_$AD.log"
  if [ -f "$CK/last_$AD.pt" ]; then echo "[$(ts)] $AD ATLANDI (var)"; return 0; fi
  echo "[$(ts)] === $AD === w_mask=$WM"
  python -u scripts/train_lrm.py \
    --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $STEPS --warmup 300 \
    --micro_batch 8 --grad_accum 2 --amp \
    --w_mask "$WM" --density_bias 1.0 \
    --teacher_subset 0 \
    --ckpt_dir "$CK" --tag "$AD" --val_every 1000 --ckpt_every 3000 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|xFormers\|Using cache" \
    | tee "$LOG"
  echo "[$(ts)] $AD bitti"
}

# SIRA BILINCLI (2026-09-02): BELIRLEYICI kol ONCE.
# WM_0p0 iddiayi curutebilecek TEK kol -- once o kossun ki 6 saatlik
# zincirin sonunda bir aksilik olursa kaybedilen kol o olmasin.
# WM_0p25 KOSULLU: yalnizca iki uc de cokerse (dolu kup / bos sahne)
# aralarinda egim var mi diye anlamli; WM_0p0 CALISIRSA gereksiz --
# o durumda bu kolu iptal et, 2 saat kazan.
kol WM_0p0  0.0
kol WM_1p0  1.0
kol WM_0p25 0.25

echo
echo "[$(ts)] ================= w_mask OZETI ================="
python - << 'PY'
import os, re
print(f"{'kol':<10} {'PSNR':>7} {'acc':>7} {'obj_std':>8} {'top1':>7} {'oran':>6}  tani")
print("-" * 74)
for ad in ("WM_1p0", "WM_0p25", "WM_0p0"):
    p = f"dataset/lrm_logs/wmask_{ad}.log"
    if not os.path.exists(p):
        print(f"{ad:<10} {'KOSMADI':>7}"); continue
    son, psnrler = None, []
    for L in open(p, encoding="utf-8", errors="replace"):
        if "[VAL step" in L and "psnr=" in L:
            son = L
            m = re.search(r"psnr=([0-9.]+)", L)
            if m: psnrler.append(float(m.group(1)))
    if son is None:
        print(f"{ad:<10} {'VAL YOK':>7}"); continue
    g = lambda k, d=float("nan"): (lambda m: float(m.group(1)) if m else d)(
        re.search(k + r"=([0-9.]+)", son))
    psnr, acc, std, oran = g("psnr"), g("acc"), g("obj_std"), g("oran")
    t1 = (lambda m: float(m.group(1)) if m else float("nan"))(
        re.search(r"top1=([0-9.]+)%", son))
    if acc == acc and acc < 0.01:        tani = "BOS SAHNE cokusu"
    elif acc == acc and acc > 0.40:      tani = "DOLU KUP cokusu"
    elif len(psnrler) >= 3 and max(psnrler[-3:]) - min(psnrler[-3:]) <= 0.01:
        tani = "DONMUS (t1_taban gibi)"
    elif psnr > 15 and 0.05 <= acc <= 0.35:
        tani = "*** CALISIYOR ***"
    else:                                 tani = "belirsiz -- onizlemeye BAK"
    print(f"{ad:<10} {psnr:>7.2f} {acc:>7.4f} {std:>8.4f} {t1:>6.1f}% {oran:>6.3f}  {tani}")
print("-" * 74)
print("Herhangi bir kol CALISIYOR ise: uc asamali zincirin gerekcesi cokuyor,")
print("ogretmen VRAM duvari ve CPU sayfalama isi gereksizlesir.")
print("Hicbiri calismiyorsa: 'sifirdan calismiyor' NIHAYET tek degiskenli dogrulandi.")
print("Her durumda onizlemelere BAK: dataset/lrm_val_previews/<tag>/")
PY
echo "[$(ts)] w_mask BITTI"
