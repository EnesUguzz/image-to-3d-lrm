#!/usr/bin/env bash
# GECE ZINCIRI 2026-09-02 -- uc bagimsiz deney, sirayla.
#
#   1) K256 devam (adim 2000/6000'den) + dalga 2 keskinlik olcumu   ~4,5 sa
#   2) ORACLE KOSULLANDIRMA (O0/O1/O2)                              ~3,0 sa
#   3) w_mask TARAMASI (1.0 / 0.25 / 0.0)                           ~3,8 sa
#                                                            TOPLAM ~11,3 sa
#
# UCU DE BAGIMSIZ: biri patlarsa digerleri kosar. Her asama kendi kapisini
# ve ozetini kendi basar; bu betik yalnizca siralar.
#
# ⚠️ BU BETIGI KOSARKEN DUZENLEME. bash betigi bayt konumundan artimli okur;
# calisirken degistirmek yurutmeyi bozar (2026-08-29'da yasandi). Bir asamayi
# degistirmen gerekirse once bu betigi durdur.
set -u
cd "$(dirname "$0")/.."
ts(){ date '+%F %T'; }

echo "############################################################"
echo "[$(ts)] GECE ZINCIRI BASLIYOR"
echo "  1) K256 devam + dalga2      2) oracle      3) w_mask"
echo "############################################################"

echo
echo "[$(ts)] >>>>>>>>>> ASAMA 1/3: K256 devam <<<<<<<<<<"
bash scripts/tavan2_k256_devam.sh || echo "[$(ts)] !! ASAMA 1 BASARISIZ -- devam ediliyor"

echo
echo "[$(ts)] >>>>>>>>>> ASAMA 2/3: ORACLE KOSULLANDIRMA <<<<<<<<<<"
bash scripts/bench_oracle.sh || echo "[$(ts)] !! ASAMA 2 BASARISIZ -- devam ediliyor"

echo
echo "[$(ts)] >>>>>>>>>> ASAMA 3/3: w_mask TARAMASI <<<<<<<<<<"
bash scripts/bench_wmask.sh || echo "[$(ts)] !! ASAMA 3 BASARISIZ"

echo
echo "############################################################"
echo "[$(ts)] GECE ZINCIRI BITTI"
echo "  Ozetler: yukarida her asamanin sonunda."
echo "  GORSELLERE BAK -- sayilar yeter demiyor:"
echo "    dataset/lrm_bench/TAVAN2_dalga2_kare.png"
echo "    dataset/lrm_val_previews/WM_*/"
echo "############################################################"
