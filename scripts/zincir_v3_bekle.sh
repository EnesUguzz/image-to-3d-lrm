#!/usr/bin/env bash
# 1. asama (ayri surec olarak koskuyor) bitene kadar bekle, sonra 2+3'u calistir.
# Boylece zincir oturumdan da, benden de bagimsiz olarak sabaha kadar akar.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=dataset/lrm_logs/v3_asama1.out
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] 1. asamanin bitmesi bekleniyor ($LOG icinde 'kaydedildi:')"
while true; do
  if grep -q "^kaydedildi:" "$LOG" 2>/dev/null; then
    echo "[$(ts)] 1. asama bitti"; break
  fi
  # surec olduyse ve 'kaydedildi' yoksa -> cokme; bekleme, haber ver ve cik
  if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q python.exe; then
    echo "[$(ts)] !! HATA: python sureci yok ama 1. asama tamamlanmamis. Zincir durdu."
    tail -20 dataset/lrm_logs/v3_asama1.err
    exit 1
  fi
  sleep 60
done

exec bash scripts/zincir_v3_asama23.sh
