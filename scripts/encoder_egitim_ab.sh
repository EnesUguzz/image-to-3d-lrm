#!/usr/bin/env bash
# ENCODER EGITIMI A/B -- distilasyonda DINOv2'nin son 4 blogu acilsin mi? (2026-09-04)
#
# NEDEN: `encoder.py:15` tum DINOv2 parametrelerini donduruyor, `:27` forward'i
# @torch.no_grad ile sariyor. `train_lrm.py:527` gerektiginde bu sarmalayiciyi
# ACIYOR -- ama `distill_lrm` HIC acmiyordu. Yani goruntu->triplane haritasinin
# FIILEN ogrenildigi asamada encoder her zaman tamamen donuktu.
#
# Bunu isaret eden ONCEKI KOSU (encoder duvari A/B, 2026-09-03, uc kol):
#   girdi 224 -> 448:  heldout_rel d = -0.0011 (kontrol gurultusu 0.0050) => YOK
#                      insample_rel d = -0.0163 (= 2.8x gurultu)          => VAR
# Yani 4x daha cok piksel giriyor, model onunla EZBERLIYOR ama GENELLEMIYOR.
# En dogal aciklama: ozellikler donuk ve semantik; dokuyu tasimiyorlar.
# Referans uygulamalarin UCU DE (LRM / OpenLRM / TripoSR) encoder'i egitir.
#
# KOSUDAN ONCE FIILEN DOGRULANDI (bayragin bir sey yaptigi VARSAYILMADI):
#   ENCODER: son 4 blok ACIK | 7.1M parametre @ lr x 0.1 = 4.00e-05
#   kunye unfreeze_last = 4 | enc_lr_scale = 0.1
#   son 4 blok  (ACIK olmali) : 16/16 tensor DEGISTI
#   ilk 4 blok  (DONUK olmali):  0/16 tensor DEGISTI
# (`M_enc4` gizemi tam bu kontrol yapilmadigi icin haftalarca acik kalmisti.)
#
# KOLLAR (tek degisken: --unfreeze_last). Ucuncu kol PAZARLIK KONUSU DEGIL.
#   D0a  unfreeze_last 0   seed 0   KONTROL (bugune kadarki davranis)
#   D4   unfreeze_last 4   seed 0   DENEY
#   D0b  unfreeze_last 0   seed 1   KONTROLUN 2. TOHUMU (gurultu bandi)
#
# Geri kalan her sey encoder duvari kosusuyla BIREBIR AYNI (1024 obje,
# 896 egitim + 128 held-out, 6000 adim, batch 8, girdi 224, seed 0/1).
# Boylece D0a, oncekinin E224a'siyla dogrudan kiyaslanabilir bir tekrardir --
# olcum altyapisinin kendi tekrar edilebilirligi de bedava olculmus olur.
#
# ON-KAYITLI KAPILAR (KOSUDAN ONCE YAZILDI):
#
#   SAGLAMA : heldout_rel(D0a) < 0.97  VE  |heldout_rel(D0a) - 0.8067| < 0.02
#     Ikincisi TEKRAR URETILEBILIRLIK kontrolu: D0a, E224a ile ayni tarife.
#     Tutmuyorsa olcum altyapisinda bir sey kaymistir; A/B okunmaz.
#
#   GURULTU BANDI  b = |heldout_rel(D0a) - heldout_rel(D0b)|   (OLCULUYOR)
#
#   GO      : heldout_rel(D4) < heldout_rel(D0a) - max(2b, 0.03)
#             -> Donuk encoder BAGLAYICI. Distilasyon varsayilani degisir,
#                zincir yeniden kosar. Keskinlik programi ACIK kalir.
#   NO-GO   : |heldout_rel(D4) - heldout_rel(D0a)| < b
#             -> Encoder'i egitmek bu butcede genellemeyi degistirmiyor.
#                KESKINLIK PROGRAMI KAPANIR: cozunurluk cercevesinin uc duvari
#                + decoder + head + denetim + encoder egitimi, hepsi elendi.
#                Faz C'ye gecilir.
#   BELIRSIZ: digeri.
#
#   IKINCIL (karar VERMEZ, rapor edilir): `insample_rel` ve `kor_orta`.
#     Encoder egitimi in-sample'i iyilestirip held-out'u iyilestirmiyorsa,
#     bu 448 kosusuyla AYNI desendir ve "ezber" teshisini guclendirir.
set -u
cd "$(dirname "$0")/.."

T=dataset/lrm_ckpts/teacher_1024_v4.pt
LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
O=dataset/lrm_bench
LG=dataset/lrm_logs
N=1024; ADIM=6000; BATCH=8; HO=128
ts(){ date '+%F %T'; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

kol(){   # kol <ad> <unfreeze_last> <seed>
  local AD=$1 UF=$2 SD=$3
  local OUT="$O/DENC_$AD.pt"
  if [ -f "$OUT" ]; then echo "[$(ts)] $AD ATLANDI (var)"; return 0; fi
  echo "[$(ts)] ===== KOL $AD ===== unfreeze_last=$UF seed=$SD"
  python -u scripts/distill_lrm.py \
    --teacher "$T" --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $ADIM --batch $BATCH --log_every 250 \
    --force_n_input 1 --holdout $HO --seed "$SD" \
    --unfreeze_last "$UF" --enc_lr_scale 0.1 \
    --out "$OUT" 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|xFormers\|Using cache" \
    | tee "$LG/denc_$AD.log"
  local RC=${PIPESTATUS[0]}
  [ "$RC" != "0" ] && { echo "[$(ts)] !! $AD kod $RC ile dustu"; return 1; }
  # KUNYE DOGRULAMASI: kolun gercekten istenen ayarla kostugunu ckpt SOYLESIN.
  python - "$OUT" "$UF" << 'PY'
import sys, torch
tk = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = tk.get("render_cfg") or {}
if int(c.get("unfreeze_last", -1)) != int(sys.argv[2]):
    sys.exit(f"KUNYE UYUSMAZLIGI: beklenen unfreeze_last={sys.argv[2]}, "
             f"ckpt'te {c.get('unfreeze_last')}")
print(f"  kunye dogrulandi: unfreeze_last={c.get('unfreeze_last')}, "
      f"input_res={c.get('input_res')}, adim={tk.get('step')}")
PY
  [ $? -ne 0 ] && { echo "[$(ts)] !! $AD kunye dogrulamasi BASARISIZ"; return 1; }
  echo "[$(ts)] $AD bitti"
}

kol D0a 0 0 || exit 1
kol D4  4 0 || exit 1
kol D0b 0 1 || exit 1

echo
echo "[$(ts)] ================= OLCUM ================="
# NEREDEYIZ gorseliyle AYNI referanslar + yeni kollar => tek tabloda
# "nereden nereye" okunabilsin.
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 6 --zoom_px 96 \
  --ray_chunk 8192 --out "$O/ENCEGITIM" \
  --teachers dataset/lrm_ckpts/teacher_1024_v3.pt "$T" \
  --students dataset/lrm_ckpts/distilled_v3.pt \
             "$O/DENC_D0a.pt" "$O/DENC_D4.pt" "$O/DENC_D0b.pt" \
             dataset/lrm_ckpts_v3/last_V3.pt 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|Using cache" \
  | tee "$LG/denc_keskinlik.log"

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
kollar = ["D0a", "D4", "D0b"]
ho = {k: son(f"dataset/lrm_logs/denc_{k}.log", "heldout_rel") for k in kollar}
ins = {k: son(f"dataset/lrm_logs/denc_{k}.log", "insample_rel") for k in kollar}
print(f"{'kol':<6} {'unfreeze':>9} {'heldout_rel':>12} {'insample_rel':>13}")
for k, u in zip(kollar, [0, 4, 0]):
    h = ho[k] if ho[k] is not None else float('nan')
    i = ins[k] if ins[k] is not None else float('nan')
    print(f"{k:<6} {u:>9} {h:>12.4f} {i:>13.4f}")
print()
if all(ho[k] is not None for k in kollar):
    b = abs(ho["D0a"] - ho["D0b"])
    d = ho["D4"] - ho["D0a"]
    esik = max(2 * b, 0.03)
    tekrar = abs(ho["D0a"] - 0.8067)
    print(f"SAGLAMA-1 : heldout_rel(D0a) = {ho['D0a']:.4f} < 0.97  "
          f"{'GECTI' if ho['D0a'] < 0.97 else 'KALDI'}")
    print(f"SAGLAMA-2 : |D0a - E224a(0.8067)| = {tekrar:.4f} < 0.02  "
          f"{'GECTI (olcum tekrar uretilebilir)' if tekrar < 0.02 else 'KALDI (altyapi kaymis!)'}")
    print(f"GURULTU   : b = |D0a - D0b| = {b:.4f}   (OLCULDU)")
    print(f"ETKI      : d = D4 - D0a = {d:+.4f}   GO esigi: d < -{esik:.4f}")
    if ho["D0a"] >= 0.97 or tekrar >= 0.02: karar = "BELIRSIZ (saglama kapisi kaldi)"
    elif d < -esik:  karar = "GO -- donuk encoder BAGLAYICI, distilasyon varsayilani degisir"
    elif abs(d) < b: karar = "NO-GO -- KESKINLIK PROGRAMI KAPANIR, Faz C'ye gecilir"
    else:            karar = "BELIRSIZ"
    print(f"\nKARAR     : {karar}")
    if ins["D4"] is not None and ins["D0a"] is not None:
        bi = abs(ins["D0a"] - ins["D0b"]) if ins["D0b"] is not None else float('nan')
        print(f"\nIKINCIL   : insample d = {ins['D4'] - ins['D0a']:+.4f} "
              f"(kontrol yayilimi {bi:.4f}). In-sample iyilesip held-out "
              f"iyilesmiyorsa 448 kosusuyla AYNI desen = EZBER.")
print()
print("GORSEL    : dataset/lrm_bench/ENCEGITIM_kare.png")
print("            (GT | ogretmen v3 | ogretmen v4 | distile | D0a | D4 | D0b | URUN)")
PY
echo "[$(ts)] ENCODER EGITIMI A/B BITTI"
