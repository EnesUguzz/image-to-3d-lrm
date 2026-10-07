#!/usr/bin/env bash
# OGRETMEN v4 -- uretim ogretmenini BOLGE KIRPMA denetimiyle yeniden oturt.
#
# NEDEN (2026-09-03, ZINCIR3 olcumu):
# Uretim ogretmeni `teacher_1024_v3` kendi izgarasini KULLANMIYOR. Ayni 24
# objede, ayni tp64 izgarasi, tek fark DENETIM:
#
#   sistem                          kor     kor_orta
#   GT 64->256 tavani              0.190     0.767
#   TAVAN2_K64 (bolge128<-U[256,512])
#                                  0.158     0.418
#   teacher_1024_v3 (res 64 tam kare)
#                                  0.087     0.292     <- URETIM
#
# Yani tp64 izgarasi 0.418 verebiliyorken uretim ogretmeni 0.292'de.
# Izgara (tp_res) bagLAYICI DEGIL -- bu, "keskinlik = cozunurluk butcesi"
# cercevesinin ORTA BANTTA yanlis oldugunu gosteren ilk dogrudan olcum.
# Maliyet ayni: isin butcesi region^2'de SABIT (32768 isin, K64 ile birebir).
#
# ⚠️ CONFOUND (bilincli olarak KABUL EDILDI, cozumu bu kosu):
# K64 obje basina 500 guncelleme aldi, uretim ogretmeni 100. 0.292 -> 0.418
# farkinin ne kadari denetim, ne kadari guncelleme sayisi -- BILINMIYOR.
# Bu kosu uretimle AYNI 100 guncellemeyi kullanir => farki denetime izole eder.
#
# ⚠️ IKINCI FARK (kacinilamaz): bu kosu `OLCEK_YONTEMI=kutu` varsayilaniyla
# koser, `teacher_1024_v3` ise `area` ile oturtulmustu. E-R'de olculdu:
# tp64'te etki ANLAMLI DEGIL (d_kor +0.010 +- 0.033, GA sifiri iceriyor).
# Yine de tek degiskenli degil; raporlarken yaz.
#
# ON-KAYITLI KAPI (kosudan ONCE yazildi), ayni 24 objede, `kor_orta`:
#   >= 0.38  -> Fark DENETIMDENDI. Uretim ogretmeni bu tarifeyle degistirilir;
#              distilasyon ve 3. asama YENIDEN kosar. tp128 gundemden duser.
#   0.32-0.38-> Kismi. Denetim faydali ama guncelleme sayisi da rol oynuyor.
#   <= 0.32  -> Fark GUNCELLEME SAYISINDANDI (K64'un 500'u). Denetim tek
#              basina yetmiyor; obje basina butce artirilmali.
set -u
cd "$(dirname "$0")/.."

LIST=dataset/train_list_v2.json
REND=dataset/renders_opp_score3
OUT=dataset/lrm_ckpts/teacher_1024_v4.pt
LG=dataset/lrm_logs
N=1024
# uretimle AYNI: obje basina 100 guncelleme. batch 2 x 51200 / 1024 = 100.
# batch/n_sup/region/ray_chunk K64 ile BIREBIR (o kolun 0.418'ini hedefliyoruz).
ADIM=51200
ts(){ date '+%F %T'; }

echo "[$(ts)] OGRETMEN v4: 1024 obje, bolge128 <- U[256,512], tp64, ~3.2 sa"
echo "[$(ts)] banka bellegi: 1024 x 1.57 MB = 1.61 GB (+Adam 4.8 GB)"

# 2026-09-03 00:36: ILK DENEME COKTU -- `--ray_chunk 0` ile VRAM 15.9/16.3 GB'ye
# cikti ve backward()'da CUBLAS_STATUS_INTERNAL_ERROR verdi (cuBLAS calisma
# alani ayiramiyor = OOM komsusu). K64 ayni ayarla sigiyordu cunku bankasi
# 24 obje = 0.04 GB; burada banka+Adam 6.4 GB fazladan.
# ray_chunk sonucu BIT-BAZINDA degistirmez (tests/test_render_chunk.py),
# yalnizca isinlari parcalara boler => tepe bellek O(chunk).
# expandable_segments: parcalanmayi azaltir (bu boyutta kritik).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u scripts/fit_teacher.py \
  --train_list "$LIST" --renders_dir "$REND" \
  --n_obj $N --steps $ADIM --batch 2 --n_sup 2 \
  --region 128 --render_low 256 --render_high 512 \
  --tp_res 64 --tp_ch 32 --nerf_layers 4 --nerf_hidden 64 \
  --n_samples 96 --w_tv 0.05 --ray_chunk 8192 \
  --eval_res 256 --ckpt_every 2000 --resume \
  --out "$OUT" 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn\|detach()" \
  | tee "$LG/ogretmen_v4.log"
RC=${PIPESTATUS[0]}
if [ "$RC" != "0" ]; then
  echo "[$(ts)] !! v4 HATA (cikis $RC) -- OLCUM YAPILMIYOR."; exit 1
fi

# TAMAMLANDI MI, KODLA DOGRULA (echo'ya guvenme -- 2026-09-01'de oldurulen bir
# surecin ardindan "bitti" basildi ve logu yaniltici yapti).
python - "$OUT" << 'PY'
import sys, torch
tk = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not tk.get("done"):
    sys.exit(f"v4 TAMAMLANMAMIS (adim {tk.get('step')}, done={tk.get('done')})")
print(f"v4 dogrulandi: adim {tk['step']}, done=True, "
      f"olcek={tk.get('render_cfg', {}).get('olcek_yontemi')}")
PY
[ $? -ne 0 ] && { echo "[$(ts)] !! dogrulama basarisiz"; exit 1; }

# OLCUM: ZINCIR3 ile AYNI 24 obje, AYNI cagri sekli => eslesmis karsilastirma.
echo "[$(ts)] === KESKINLIK: v4 <-> v3 <-> zincirin geri kalani ==="
python -u scripts/bench_keskinlik.py --n_obj 24 --res 256 --n_gorsel 6 \
  --out dataset/lrm_bench/OGRETMEN_V4 \
  --teachers dataset/lrm_ckpts/teacher_1024_v3.pt "$OUT" \
             dataset/lrm_bench/TAVAN2_K64.pt 2>&1 \
  | grep -v --line-buffered "Setting up\|Loading model\|UserWarning\|warnings.warn\|^  warn" \
  | tee "$LG/ogretmen_v4_keskinlik.log"

echo
echo "[$(ts)] KAPI: v4'un kor_orta'si >=0.38 ise denetim, <=0.32 ise guncelleme"
echo "[$(ts)] sayisi belirleyiciydi. Referans: v3 0.292, K64 0.418, tavan 0.767."
echo "[$(ts)] GORSELE BAK: dataset/lrm_bench/OGRETMEN_V4_kare.png"
echo "[$(ts)] OGRETMEN v4 BITTI"
