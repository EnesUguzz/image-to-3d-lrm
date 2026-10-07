#!/usr/bin/env bash
# 3. ASAMA DENETIM UYUSMAZLIGI A/B  (2026-09-03)
#
# SORU: 3. asama (train_lrm) neden distile ogrencinin dokusunun %88'ini yok
# ediyor? Olculmus zincir (kor_orta, 24 obje, 256px, bench_keskinlik):
#
#   ogretmen v3   0.292
#   distile       0.199   -> 2. asama %68'ini TASIYOR   (saglikli)
#   last_V3       0.024   -> 3. asama %12'sini tasiyor  (felaket)
#
# HIPOTEZ: 3. asamanin DENETIMI, distile edildigi ogretmeninkinden KABA.
#   ogretmen v4     : region 128 yama <- render U[256,512]  (detay tavani 512px)
#   3. asama (V3)   : region  64 yama <- render U[ 64,192]  (detay tavani 192px)
# `zincir_v3_asama23.sh` bu bayraklari HIC gecersiz kilmamis => varsayilan.
# Ince ayar, triplane'i kendi kaba denetiminin gerekcelendirdigi yere ceker;
# miras alinan ince yapi silinir.
#
# KOLLAR (tek degiskenli, --dry_run ile dogrulandi 2026-09-03 11:06):
#   A: region  64 <- U[ 64,192]   VARSAYILAN = KONTROL
#   B: region  64 <- U[192,512]   detay tavani yukseldi, ISIN BUTCESI AYNI
# A->B tek degisken: denetimin detay tavani. B ayni hizda kosar (64^2 isin),
# yani ise yararsa BEDAVA bir duzeltme.
#   C: region 128 <- U[192,512]   (ogretmenle tam eslesmis; ~4 sa, SADECE
#      B belirsiz kalirsa kosulur -- ayri betik)
#
# ON-KAYITLI KAPILAR (KOSUDAN ONCE YAZILDI):
#
#   SAGLAMA (once buna bak): kor_orta(A) < 0.17 olmali.
#     A varsayilan tarifedir; 30800 adimda 0.024'e dusuyor. 3000 adimda
#     0.199'dan asagi HIC inmediyse deney COK KISA demektir ve B'nin sonucu
#     -- iyi de cikse kotu de -- OKUNAMAZ. O durumda etiket BELIRSIZ'dir,
#     "denetim sucsuz" DEGIL. Bu projede uc kez guclu-olmayan deneyden
#     sonuc cikarildi; dordunculer olmayacak.
#
#   GO      : kor_orta(B) - kor_orta(A) >= +0.05  VE  kor_orta(B) >= 0.12
#             -> Kayip denetim uyusmazligindan. B benimsenir (tek bayrak, kod
#                yok, hiz bedeli yok). SONRA v4 ogretmen (diskte hazir).
#                SONRA tp128 kod bedelini hak eder.
#   NO-GO   : ikisi de < 0.08 VE |B-A| < 0.027
#             -> Denetim SEBEP DEGIL. Keskinlik programi KALICI kapanir;
#                tp128 de v4 de yazilmaz. Faz C'ye gecilir.
#   BELIRSIZ: digeri. Tohum gurultu bandi +-0.027 (DIS-HEAD'de kontrolun kendi
#             iki tohumu bu kadar ayrildi). Bu bandin icindeki fark OKUNMAZ.
#
# REFERANS NOKTALARI (ayni araca, ayni 24 objeye ait; olcum kolu da onlari
# yeniden olcuyor ki tablo kendi icinde tutarli olsun):
#   distilled_v3 0.199 (BASLANGIC)  |  last_V3 0.024 (A tarifesinin 30800 adimi)
#   ogretmen v4  0.361              |  GT 64px tavani 0.767
set -u
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
CK=dataset/lrm_ckpts_s3
LG=dataset/lrm_logs
DIS=dataset/lrm_ckpts/distilled_v3.pt
ADIM=3000
ts(){ date '+%F %T'; }
mkdir -p "$CK"

# distilled_v3 KUNYESIZ (eski format) => train_lrm zincirin fiili varsayilanini
# kabul ediyor: density_bias 0.0, bound 0.6. Ikisini de ACIKCA veriyoruz ki
# defaults.py ileride degisirse bu kosu sessizce kaymasin.
ORTAK=(--train_list "$LIST" --renders_dir "$REND"
       --steps $ADIM --warmup 200 --workers 6 --amp
       --teacher_subset 0 --unfreeze_last 4 --enc_lr_scale 0.1
       --density_bias 0.0 --bound 0.6 --w_lpips 0.25
       --input_crop 0.0 --input_pool mixed --mixed_p 0.5
       --val_every 500 --ckpt_every 1000 --snapshot_every 1000 --val_n 64
       --seed 0 --ckpt_dir "$CK" --init_from "$DIS")

# ray_chunk yerine: parcalama zaten renderer'da; asil risk B'nin 512'lik
# render'lari yuklerken RAM/VRAM. expandable_segments parcalanmayi azaltir.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

kol(){   # kol <ad> <region> <low> <high>
  local AD=$1 RG=$2 LO=$3 HI=$4
  if [ -f "$CK/last_S3_$AD.pt" ]; then echo "[$(ts)] $AD ATLANDI (var)"; return 0; fi
  echo "[$(ts)] ===== KOL $AD ===== region $RG <- U[$LO,$HI]"
  python -u scripts/train_lrm.py "${ORTAK[@]}" \
    --region "$RG" --render_low "$LO" --render_high "$HI" \
    --tag "S3_$AD" --micro_batch 8 --grad_accum 2 2>&1 \
    | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
    | tee "$LG/s3ab_$AD.log"
  local RC=${PIPESTATUS[0]}
  if [ "$RC" != "0" ]; then
    # V3'te de yasandi: ayni EFEKTIF batch (16), daha kucuk mikro-batch +
    # gradyan checkpoint. Sonuc matematiksel olarak ayni (BatchNorm yok).
    echo "[$(ts)] !! $AD kod $RC ile dustu -- micro_batch 4 x accum 4 + grad_ckpt"
    python -u scripts/train_lrm.py "${ORTAK[@]}" \
      --region "$RG" --render_low "$LO" --render_high "$HI" \
      --tag "S3_$AD" --micro_batch 4 --grad_accum 4 --grad_ckpt --resume 2>&1 \
      | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
      | tee -a "$LG/s3ab_$AD.log"
    [ "${PIPESTATUS[0]}" != "0" ] && { echo "[$(ts)] !! $AD IKINCI DENEMEDE DE DUSTU"; return 1; }
  fi
  echo "[$(ts)] $AD bitti"
}

kol A  64  64 192 || exit 1
kol B  64 192 512 || exit 1

echo
echo "[$(ts)] ================= OLCUM ================="
# 1) SAYISAL TABLO: yorunge dahil (snapshot'lar) -- "A gercekten bozuluyor mu"
#    sorusu ancak egriden cevaplanir, tek noktadan degil.
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 0 \
  --out "$LG/S3AB_sayisal" \
  --teachers dataset/lrm_ckpts/teacher_1024_v4.pt \
  --students "$DIS" \
             "$CK/snap_S3_A_step001000.pt" "$CK/snap_S3_A_step002000.pt" "$CK/last_S3_A.pt" \
             "$CK/snap_S3_B_step001000.pt" "$CK/snap_S3_B_step002000.pt" "$CK/last_S3_B.pt" \
             dataset/lrm_ckpts_v3/last_V3.pt 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
  | tee "$LG/s3ab_sayisal.log"

echo
# 2) GORSEL: sade tutuldu -- GT | baslangic | A | B. Kullanicinin karari
#    gozle dogrulayabilmesi icin sutun sayisi az olmali.
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 6 --zoom_px 96 \
  --out dataset/lrm_bench/S3DENETIM \
  --students "$DIS" "$CK/last_S3_A.pt" "$CK/last_S3_B.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
  | tee "$LG/s3ab_gorsel.log"

echo
python -u scripts/diag_kafes.py --n_obj 8 --device cuda \
  --teachers dataset/lrm_ckpts/teacher_1024_v4.pt \
  --students "$DIS" "$CK/last_S3_A.pt" "$CK/last_S3_B.pt" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
  | tee "$LG/s3ab_kafes.log"

echo
echo "[$(ts)] ================= KAPI ================="
echo "SAGLAMA once: kor_orta(A) < 0.17 mi? Degilse deney COK KISA => BELIRSIZ."
echo "  (A varsayilan tarife; 30800 adimda 0.024. 3000'de kipirdamadiysa okunmaz.)"
echo "GO      : kor_orta(B) - kor_orta(A) >= +0.05 VE kor_orta(B) >= 0.12"
echo "NO-GO   : ikisi de < 0.08 VE |B-A| < 0.027"
echo "BELIRSIZ: digeri (tohum gurultu bandi +-0.027)"
echo
echo "GORSEL: dataset/lrm_bench/S3DENETIM_tam.png ve _zoom.png"
echo "[$(ts)] S3 DENETIM A/B BITTI"
