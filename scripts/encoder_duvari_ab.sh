#!/usr/bin/env bash
# ENCODER DUVARI A/B -- girdi cozunurlugu 224 vs 448 (2026-09-03)
#
# NEDEN: cozunurluk butcesi tablosundaki (CLAUDE.md 2026-09-02) UC duvardan
# ikisi olculdu ve gevsetildi; bu HIC denenmedi.
#
#   master render 512px          -> obje 276 px
#   encoder 224 / DINOv2 patch14 -> obje   8.6 patch   <-- BU DUVAR
#   token izgarasi 32^2          -> obje  24 token
#   triplane 64^2                -> obje  48 hucre
#   olcum 256px                  -> obje 138 px
#
# 448'de obje 17.2 patch. Iddia: model girdi fotografinda dokuyu GORMUYOR;
# gormeden uretmesi ancak EZBERLE mumkun -- ve olculdu ki oyle oluyor
# (distilled_v3 187 maruziyet/obje -> kor_orta 0.199; last_V3 40 -> 0.024;
# heldout_rel 0.935 = gorulmemis objede hedef varyansinin %6.5'i).
#
# MALIYET OLCULDU (kosudan once, 120 adim, 1024 obje):
#   GIRDI_RES=224  0.70 it/s        GIRDI_RES=448  0.70 it/s
# Bedava. Sebep: transformer goruntu + triplane token'larini BIRLESTIRIP
# self-attention yapiyor (transformer.py:108) => dizi 256+3072=3328 ->
# 1024+3072=4096, attention (4096/3328)^2 = 1.51x; encoder donuk ViT-S.
# Yani "4x pahali" beklentisi YANLISTI, olculdu.
#
# KOSUDAN ONCE FIILEN DOGRULANANLAR (hicbiri varsayim degil):
#  1. LRM 448'de kosuyor: 1024 patch, side 32, triplane (3,32,64,64).
#  2. Kunye `input_res` tasiyor (distill_lrm.py:199).
#  3. `compat.load_lrm` onu GERI KURUYOR -- eksikti, eklendi. Eksikken 448'de
#     egitilmis ckpt 224 Plucker haritasiyla render edilir ve HIC HATA VERMEZ
#     (model.py:52 `scale_intrinsics(K, INPUT_RES, side)`; `side` gercek
#     girdiden, `INPUT_RES` surecin varsayilanindan gelir).
#  4. `bench_keskinlik` karisik 224/448 ogrenci listesini dogru olcuyor
#     (girdi cozunurlugune gore ayri dataset; supervision kameralari ortak).
#     Sahte bir 448 ckpt ile fiilen kosturuldu, exit 0.
#  5. Testler 227 (2 yeni regresyon: kunyeden geri kurma + kunyesiz ckpt).
#
# KOLLAR (tek degisken: GIRDI_RES). Ucuncu kol PAZARLIK KONUSU DEGIL --
# tek kollu A/B bu projede uc kez okunamaz sonuc verdi (en son DIS-HEAD:
# kontrolun kendi tohum yayilimi 0.021, aranan etkiden buyuk).
#   E224a  GIRDI_RES=224  seed 0   KONTROL
#   E448   GIRDI_RES=448  seed 0   DENEY
#   E224b  GIRDI_RES=224  seed 1   KONTROLUN 2. TOHUMU (gurultu bandi)
#
# holdout 128 (varsayilan 32 degil): karar metrigi `heldout_rel` ve 32 obje
# uzerinde cok gurultulu. 1024 - 128 = 896 egitim objesi, uc kolda AYNI.
# 6000 adim x batch 8 / 896 = 53.6 maruziyet/obje.
#
# ON-KAYITLI KAPILAR (KOSUDAN ONCE YAZILDI):
#
#   SAGLAMA (once buna bak): heldout_rel(E224a) < 0.97 olmali.
#     Kollar hic ogrenmediyse (1.0 = hicbir sey) fark okunamaz ve etiket
#     BELIRSIZ'dir, "encoder sucsuz" DEGIL. 120 adimda 1.009 idi; 6000'de
#     0.97 altina inmezse rejim yetersiz demektir.
#
#   GURULTU BANDI  b = |heldout_rel(E224a) - heldout_rel(E224b)|
#                  Bu ON-KAYITLI DEGIL, OLCULUYOR. Karar ona gore veriliyor.
#
#   GO      : heldout_rel(E448) < heldout_rel(E224a) - max(2b, 0.03)
#             -> Encoder duvari GERCEK ve baglayici. 448 benimsenir; zincir
#                (distilasyon + 3. asama) 448'de yeniden kosar.
#   NO-GO   : |heldout_rel(E448) - heldout_rel(E224a)| < b
#             -> Girdi cozunurlugu bu butcede BAGLAYICI DEGIL. Encoder duvari
#                kapanir. Keskinlik programinda gevsetilecek duvar kalmaz.
#   BELIRSIZ: digeri.
#
#   IKINCIL (karar vermez, RAPOR EDILIR): `kor_orta` (bench_keskinlik, 24 obje).
#     `heldout_rel` triplane regresyon sadakati; `kor_orta` render edilmis
#     goruntudeki doku. Ikisi ayni yone gitmezse SEBEBI ARANIR, biri secilmez.
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

kol(){   # kol <ad> <girdi_res> <seed>
  local AD=$1 IR=$2 SD=$3
  local OUT="$O/ENC_$AD.pt"
  if [ -f "$OUT" ]; then echo "[$(ts)] $AD ATLANDI (var)"; return 0; fi
  echo "[$(ts)] ===== KOL $AD ===== GIRDI_RES=$IR seed=$SD"
  GIRDI_RES=$IR python -u scripts/distill_lrm.py \
    --teacher "$T" --train_list "$LIST" --renders_dir "$REND" \
    --n_obj $N --steps $ADIM --batch $BATCH --log_every 250 \
    --force_n_input 1 --holdout $HO --seed "$SD" \
    --out "$OUT" 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|xFormers\|Using cache" \
    | tee "$LG/enc_$AD.log"
  local RC=${PIPESTATUS[0]}
  [ "$RC" != "0" ] && { echo "[$(ts)] !! $AD kod $RC ile dustu"; return 1; }
  # KUNYE DOGRULAMASI: kol gercekten istenen cozunurlukte mi kosmus?
  # "o bayragi verdim" yeterli degil; ckpt'in kendisi soylesin.
  python - "$OUT" "$IR" << 'PY'
import sys, torch
tk = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
g = (tk.get("render_cfg") or {}).get("input_res")
if int(g or 0) != int(sys.argv[2]):
    sys.exit(f"KUNYE UYUSMAZLIGI: beklenen input_res={sys.argv[2]}, ckpt'te {g}")
print(f"  kunye dogrulandi: input_res={g}, adim={tk.get('step')}")
PY
  [ $? -ne 0 ] && { echo "[$(ts)] !! $AD kunye dogrulamasi BASARISIZ"; return 1; }
  echo "[$(ts)] $AD bitti"
}

kol E224a 224 0 || exit 1
kol E448  448 0 || exit 1
kol E224b 224 1 || exit 1

echo
echo "[$(ts)] ================= OLCUM ================="
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 6 --zoom_px 96 \
  --ray_chunk 8192 --out "$O/ENCDUVAR" \
  --teachers "$T" \
  --students dataset/lrm_ckpts/distilled_v3.pt \
             "$O/ENC_E224a.pt" "$O/ENC_E448.pt" "$O/ENC_E224b.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|Using cache" \
  | tee "$LG/enc_keskinlik.log"

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
kollar = ["E224a", "E448", "E224b"]
ho = {k: son(f"dataset/lrm_logs/enc_{k}.log", "heldout_rel") for k in kollar}
ins = {k: son(f"dataset/lrm_logs/enc_{k}.log", "insample_rel") for k in kollar}
print(f"{'kol':<8} {'heldout_rel':>12} {'insample_rel':>13}")
for k in kollar:
    h = ho[k] if ho[k] is not None else float('nan')
    i = ins[k] if ins[k] is not None else float('nan')
    print(f"{k:<8} {h:>12.4f} {i:>13.4f}")
print()
if all(ho[k] is not None for k in kollar):
    b = abs(ho["E224a"] - ho["E224b"])
    d = ho["E448"] - ho["E224a"]
    esik = max(2 * b, 0.03)
    print(f"SAGLAMA : heldout_rel(E224a) = {ho['E224a']:.4f}  "
          f"({'GECTI' if ho['E224a'] < 0.97 else 'KALDI -> rejim yetersiz, BELIRSIZ'})")
    print(f"GURULTU : kontrolun iki tohumu arasi b = {b:.4f}  (OLCULDU, on-kayitli degil)")
    print(f"ETKI    : d = heldout_rel(E448) - heldout_rel(E224a) = {d:+.4f}")
    print(f"          GO esigi: d < -{esik:.4f}   NO-GO: |d| < {b:.4f}")
    if ho["E224a"] >= 0.97:      karar = "BELIRSIZ (saglama kapisi kaldi)"
    elif d < -esik:              karar = "GO -- encoder duvari GERCEK, 448 benimsenir"
    elif abs(d) < b:             karar = "NO-GO -- girdi cozunurlugu baglayici DEGIL"
    else:                        karar = "BELIRSIZ"
    print(f"\nKARAR   : {karar}")
print()
print("IKINCIL : `kor_orta` yukaridaki bench_keskinlik tablosundan oku (karar VERMEZ).")
print("GORSEL  : dataset/lrm_bench/ENCDUVAR_kare.png")
PY
echo "[$(ts)] ENCODER DUVARI A/B BITTI"
