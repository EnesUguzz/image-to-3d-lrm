#!/usr/bin/env bash
# DIS-HEAD -- ortusmeli head tabani GERCEK ZINCIRDE fark yaratiyor mu? (2026-09-03)
#
# ⚠️ BU BETIK v4'U BEKLER. GPU'ya v4 bitmeden dokunmaz.
#
# NEDEN:
# ZINCIR3 olcumu keskinlik kaybinin TAMAMININ ogretmenden SONRA oldugunu
# gosterdi (kor_orta 0.292 -> 0.199 -> 0.024). `diag_kafes` sebebi buldu:
#
#   sistem                          blok_sinir   Nyq
#   teacher_1024_v3 [head YOK]         1.03      1.08   <- KONTROL
#   TAVAN2_K64      [head YOK]         1.01      1.51   <- KONTROL
#   distilled_v3    [head'den]         1.97      6.80
#   last_V3         [head'den]         2.91     19.31
#
# `TriplaneHead` = ConvTranspose2d(k=2,s=2), k==s ⇒ ORTUSME SIFIR: her token
# bagimsiz bir 2x2 blok uretir. Hata blok sinirlarinda birikiyor.
#
# CPU tezgahi (bench_head_kafes.py; transformer/encoder/renderer YOK, sadece
# serbest token + head; hedef TAVAN2_K64) gercek calisma noktasinda (dim=4):
#   k2s2  rel 0.351  blok_sinir 8.67
#   k4s2  rel 0.315  blok_sinir 1.01     <- hem daha iyi uyum HEM kafes yok
#
# ⚠️ ON-KONTROL TEMIZ GECMEDI, KAYDA GECSIN: uzmanin on-kayitli esigi
# `blok_sinir(last_V3) >= 3.0` idi; olculen 2.91 +- 0.36. Esigi SONRADAN
# dusurmuyorum. Kontrollerden ayrim tartismasiz (1.03 <-> 2.91, GA'lar
# birbirine degmiyor) ⇒ mekanizma gercek zincirde VAR, ama CPU tezgahinin
# uc noktasindaki kadar siddetli DEGIL. Deneyi kosturmaya yeter,
# "kapi gecti" demeye yetmez.
#
# KURULUM: bench_oracle.sh'in BIREBIR konfigurasyonu (45 dk/kol olctuk).
# Ogretmen `teacher_1024_v3`: cfg'si region=0, res=64 ⇒ 512->64 = 8x TAM SAYI
# ⇒ `area`/`kutu` defektinden ETKILENMEMIS, kontrol olarak temiz.
#
# ⚠️ UC KOL, IKISI KONTROL. Bu projenin tekrarlayan tek hatasi n=1
# karsilastirma. Kontrolun iki tohumu arasindaki fark, A/B farkinin
# okunabilmesi icin SART.
#
# ON-KAYITLI KAPILAR (kosudan ONCE yazildi):
#   SAGLAMA  blok_sinir(H_k2s2_a) >= 3.0  -> tezgah gercek zincire transfer oldu
#            blok_sinir(H_k2s2_a) <  1.5  -> transfer YOK, head karari IPTAL
#   KARAR    d_kor_orta = kor_orta(k4s2) - kor_orta(k2s2_a) >= +0.03
#            VE kontrolun iki tohumu arasi |fark| < 0.015     -> HEAD DEGISTIR
#            d_kor_orta < +0.01                               -> kafes KOZMETIK,
#                                                                head'e dokunma
#   BEDEL    heldout_rel(k4s2) - heldout_rel(k2s2_a) > +0.02  -> ortusme uyumu
#            bozuyor; CPU tezgahiyla CELISIR (orada -0.036 bekleniyor), teshis et.
set -u
cd "$(dirname "$0")/.."

T=dataset/lrm_ckpts/teacher_1024_v3.pt
LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
O=dataset/lrm_bench
LG=dataset/lrm_logs
N=256; STEPS=3000; BATCH=8
ts(){ date '+%F %T'; }

# --- v4'u BEKLE: GPU'yu paylasmak v4'u CUBLAS hatasiyla oldururdu ---
#
# ⚠️ `pgrep -f fit_teacher.py` KULLANMA. 2026-09-03 00:54'te denendi:
# v4 Windows tarafinda `Start-Process` ile baslatilmisti, Git Bash'in pgrep'i
# onu GORMEDI ⇒ "GPU bosaldi" deyip hemen basladi ve v4'un yanina kuruldu.
# (Sansa DH birkac saniye icinde kesildi; v4 hayatta kaldi.)
#
# Bunun yerine v4'un KENDI LOGUNDAKI bitis isaretlerini bekle. Basari da
# basarisizlik da yakalanir, yoksa sonsuza kadar beklenir. Ustune VRAM
# kontrolu: log bitis yazsa bile surec kapanmamis olabilir.
V4LOG=dataset/lrm_logs/v4_zincir.out
BEKLE_MAX=$((9 * 3600))     # sert ust sinir: 9 saat
gecen=0
echo "[$(ts)] v4 bekleniyor (log: $V4LOG)..."
while true; do
  if grep -qE "OGRETMEN v4 BITTI|v4 HATA|dogrulama basarisiz" "$V4LOG" 2>/dev/null; then
    echo "[$(ts)] v4 bitis isareti gorundu"; break
  fi
  if [ "$gecen" -ge "$BEKLE_MAX" ]; then
    echo "[$(ts)] !! v4 $((BEKLE_MAX/3600)) saatte bitmedi -- DIS-HEAD IPTAL"; exit 1
  fi
  sleep 60; gecen=$((gecen + 60))
done
# VRAM gercekten bosalana kadar bekle (en fazla 10 dk)
for _ in $(seq 1 60); do
  kul=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$kul" ] && break
  [ "$kul" -lt 3000 ] && { echo "[$(ts)] VRAM bosaldi (${kul} MiB)"; break; }
  sleep 10
done
echo "[$(ts)] DIS-HEAD basliyor"
sleep 20

kol(){  # kol <ad> <head> <seed>
  local AD=$1 HD=$2 SD=$3
  local OUT="$O/DH_$AD.pt" LOG="$LG/dh_$AD.log"
  if [ -f "$OUT" ]; then echo "[$(ts)] $AD ATLANDI (var)"; return 0; fi
  echo "[$(ts)] === $AD === head=$HD seed=$SD"
  python -u scripts/distill_lrm.py \
    --teacher "$T" --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $STEPS --batch $BATCH --log_every 200 \
    --force_n_input 1 --head "$HD" --seed "$SD" --holdout 32 \
    --out "$OUT" 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|xFormers\|Using cache" \
    | tee "$LOG"
  echo "[$(ts)] $AD bitti"
}

kol H_k2s2_a  k2s2  0     # KONTROL (mevcut mimari)
kol H_k4s2    k4s2  0     # DENEY
kol H_k2s2_b  k2s2  1     # KONTROLUN 2. TOHUMU -- fark okunabilsin diye SART

echo
echo "[$(ts)] ================= DIS-HEAD OLCUMU ================="
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 6 \
  --out "$O/DISHEAD" \
  --students "$O/DH_H_k2s2_a.pt" "$O/DH_H_k4s2.pt" "$O/DH_H_k2s2_b.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
  | tee "$LG/dishead_keskinlik.log"

echo
python -u scripts/diag_kafes.py --n_obj 8 --device cuda \
  --teachers "$T" \
  --students "$O/DH_H_k2s2_a.pt" "$O/DH_H_k4s2.pt" "$O/DH_H_k2s2_b.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
  | tee "$LG/dishead_kafes.log"

echo
echo "[$(ts)] ================= KAPI ================="
python - << 'PY'
import os, re
def son(p, anahtar):
    if not os.path.exists(p): return None
    v = None
    for L in open(p, encoding="utf-8", errors="replace"):
        m = re.search(anahtar + r"=([0-9.]+)", L)
        if m: v = float(m.group(1))
    return v
kollar = ["H_k2s2_a", "H_k4s2", "H_k2s2_b"]
ho = {k: son(f"dataset/lrm_logs/dh_{k}.log", "heldout_rel") for k in kollar}
print(f"{'kol':<12} {'heldout_rel':>12}")
for k in kollar:
    print(f"{k:<12} {ho[k] if ho[k] is not None else float('nan'):>12.4f}")
print()
if all(ho[k] is not None for k in kollar):
    kontrol = abs(ho["H_k2s2_a"] - ho["H_k2s2_b"])
    bedel = ho["H_k4s2"] - ho["H_k2s2_a"]
    print(f"kontrolun iki tohumu arasi |fark| = {kontrol:.4f}  "
          f"(A/B okunabilmesi icin < 0.015 olmali)")
    print(f"BEDEL d_heldout_rel(k4s2 - k2s2_a) = {bedel:+.4f}  "
          f"(> +0.02 ise ortusme uyumu bozuyor; CPU'da -0.036 bekleniyordu)")
print()
print("KARAR SUTUNU `kor_orta` -- yukaridaki bench_keskinlik ESLESMIS FARK")
print("tablosundan oku. Referans kol H_k2s2_a.")
print("  d_kor_orta >= +0.03 VE kontrol yayilimi < 0.015  -> HEAD DEGISTIR")
print("  d_kor_orta <  +0.01                              -> kafes KOZMETIK")
print()
print("SAGLAMA: diag_kafes'te blok_sinir(H_k2s2_a) >= 3.0 ise CPU tezgahi")
print("transfer oldu; < 1.5 ise head karari IPTAL (tezgah yanlis rejimdeymis).")
print()
print("GORSELE BAK: dataset/lrm_bench/DISHEAD_kare.png")
PY
echo "[$(ts)] DIS-HEAD BITTI"
