"""ASAMA 1: ogretmen triplane'lerini AYRI olarak oturt.

Neden ayri: egitim dongusu icinde ogretmen, obje basina adim basina batch/N_dataset
(=8/2635=0.003) guncelleme aliyor; 15000 adimda sadece ~34 guncelleme. Olculdu
(diag_teacher_quality.py, step 4000): ogretmen 15.02 dB, ogrenci 16.38 dB --
yani ogretmen ogrenciyi ASAGI cekiyordu.

Burada transformer/encoder yok, adim basina 4 obje dogrudan guncelleniyor:
1024 obje icin 25600 adim = obje basina 100 guncelleme (oracle testinde 94
guncelleme 23.8 dB veriyordu). Sonuc: ogrenciden cok daha iyi bir hedef.

Cikti: dataset/lrm_ckpts/teacher_init.pt  {triplanes, nerf, uids, cfg}
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras, imutil
from lrm.dataset import LRMDataset
from lrm import defaults
from lrm.nerf import TriplaneNeRF
from lrm.triplane import sample_triplane, tv_loss
from lrm.renderer import volume_render_chunked, volume_render_importance
from lrm.losses import LRMLoss

DEV = "cuda"
EVAL_EPOCH = 10_000_003   # egitimde asla kullanilmayan sabit epoch (olcum tutarli kalsin)
OUT = "dataset/lrm_ckpts/teacher_init.pt"
PREVIEW = "dataset/lrm_val_previews/fit_teacher.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--n_obj", type=int, default=1024, help="ilk N train objesi")
    ap.add_argument("--steps", type=int, default=25600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pos_enc", type=int, default=0,
                    help="NeRF girdisine frekans-kodlu xyz ekle (bant sayisi; "
                         "0=kapali). Bilineer triplane + kodlamasiz MLP hucre "
                         "altinda yapi tasiyamaz.")
    ap.add_argument("--n_fine", type=int, default=0,
                    help="hiyerarsik onem ornekleme: kaba --n_samples gecisin "
                         "agirliklarina gore N ek ornek (0=kapali, tekduze)")
    ap.add_argument("--ray_chunk", type=int, default=0,
                    help="isinlari N'lik parcalara bol (0=kapali). Bellek "
                         "O(chunk)'a iner, sure ~1.3-2x. Yuksek --res / "
                         "--n_samples icin SART: 256px x 192 ornek x 2 obje "
                         "tek seferde ~25M nokta => OOM.")
    ap.add_argument("--n_sup", type=int, default=4,
                    help="adim basina obje basina supervision goruumu. Yuksek "
                         "--res'te VRAM bunun x res^2 ile buyur: res 256 + "
                         "n_sup 4 tek adimda ~50M nokta => OOM. Dusur, adim artir.")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--region", type=int, default=0,
                    help="BOLGE KIRPMA (OpenLRM/TripoSR; 3. asamada zaten var, "
                         "ogretmende yoktu). >0 ise denetim hedefi, U[low,high] "
                         "cozunurlukte render'dan kirpilan region x region bir "
                         "yamadir. ISIN MALIYETI region^2'de SABIT kalir ama "
                         "detay olcegi render_high kadar olur => 512 piksellik "
                         "detayi 512^2 isin odemeden alirsin. --res'i gecersiz "
                         "kilar (o zaman low/high gecerli).")
    ap.add_argument("--render_low", type=int, default=64,
                    help="--region>0 iken kirpmanin alindigi render'in alt siniri")
    ap.add_argument("--render_high", type=int, default=192,
                    help="--region>0 iken ust sinir. Detay tavani budur.")
    ap.add_argument("--eval_res", type=int, default=128,
                    help="PSNR egrisi + onizleme cozunurlugu. Egitim kirpmali "
                         "olsa bile olcum DAIMA tam karede ve sabit res'te "
                         "yapilir; yoksa egri konfigler arasi kiyaslanamaz.")
    ap.add_argument("--tp_ch", type=int, default=32,
                    help="triplane kanal sayisi (hucre BASINA kapasite)")
    ap.add_argument("--nerf_hidden", type=int, default=64)
    ap.add_argument("--nerf_layers", type=int, default=4,
                    help="TriplaneNeRF gizli katman sayisi (referans: 4)")
    ap.add_argument("--tp_res", type=int, default=64,
                    help="triplane duzlem cozunurlugu. 2026-08-31: 64'te bir "
                         "hucre = 0.01875 dunya = 256px render'da 3.5 piksel; "
                         "bunun altinda yapi tasinamaz (bkz. keskinlik olcumu)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--nerf_lr", type=float, default=1e-3)
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--w_tv", type=float, default=5e-4,
                    help="triplane TV regularizasyonu. 0 = ESKI DAVRANIS: "
                         "olculdu, TV'siz ogretmen varyansinin %40.5'i gurultu "
                         "ve distilasyon kaybina ~0.40 ogrenilemez taban koyuyor.")
    ap.add_argument("--bound", type=float, default=defaults.BOUND)
    ap.add_argument("--n_samples", type=int, default=defaults.N_SAMPLES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--ckpt_every", type=int, default=2000,
                    help="ara kayit sikligi (0=kapat). ONCEDEN HIC YOKTU: kosu "
                         "yarida kesilirse SAATLERCE is kayboluyordu; 24.6k objelik "
                         "ogretmen bankasi ~23 saat surecek, bunsuz kosulamaz.")
    ap.add_argument("--resume", action="store_true",
                    help="--out dosyasindaki ara kayittan devam et")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # deterministic=False => her erisimde farkli supervision gorunumleri
    # (ogretmen coklu-gorunum denetimi alsin, tek aciya ezberlemesin)
    ds = LRMDataset(a.train_list, a.renders_dir, split="train", render_res=a.res,
                    n_sup=a.n_sup, augment=False, deterministic=False,
                    region=a.region, render_low=a.render_low,
                    render_high=a.render_high)
    # OLCUM dataset'i: kirpmasiz, sabit cozunurlukte. Egitim kirpmali olsa bile
    # PSNR egrisi ve onizleme tam karede olcusun (konfigler arasi kiyaslanabilirlik).
    ds_eval = LRMDataset(a.train_list, a.renders_dir, split="train",
                         render_res=a.eval_res, n_sup=a.n_sup, augment=False,
                         deterministic=False)
    ds_eval.set_epoch(EVAL_EPOCH)
    n = min(a.n_obj, len(ds))
    triplanes = nn.Parameter(
        torch.randn(n, 3, a.tp_ch, a.tp_res, a.tp_res, device=DEV) * 0.1)
    nerf = TriplaneNeRF(in_dim=3 * a.tp_ch, hidden=a.nerf_hidden,
                        layers=a.nerf_layers, pos_enc=a.pos_enc,
                        pos_bound=a.bound).to(DEV)
    opt = torch.optim.Adam([{"params": [triplanes], "lr": a.lr},
                            {"params": nerf.parameters(), "lr": a.nerf_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    loss_fn = LRMLoss(use_lpips=True, w_lpips=a.w_lpips).to(DEV)
    per_obj = a.steps * a.batch / n
    print(f"ogretmen fit: {n} obje | {a.steps} adim x batch {a.batch} "
          f"=> obje basina ~{per_obj:.0f} guncelleme "
          f"| {('bolge %d <- render U[%d,%d]' % (a.region, a.render_low, a.render_high)) if a.region else ('res %d tam kare' % a.res)} "
          f"| olcum res {a.eval_res} "
          f"| triplane {a.tp_res}^2 x{a.tp_ch} | nerf {a.nerf_layers}x{a.nerf_hidden} "
          f"| bound {a.bound} | n_sup {a.n_sup} | n_samples {a.n_samples} "
          f"| ray_chunk {a.ray_chunk} | pos_enc {a.pos_enc} | n_fine {a.n_fine} "
          f"| bellek {triplanes.numel()*4/1e9:.2f} GB", flush=True)

    def render(tp, c2w, K, bg, res):
        os_, ds_ = [], []
        for v in range(c2w.shape[0]):
            o, d = cameras.rays_from_camera(c2w[v], K[v], res, res)
            os_.append(o); ds_.append(d)
        o, d = torch.cat(os_).to(DEV), torch.cat(ds_).to(DEV)

        def q(pts):
            den, rgb = nerf(sample_triplane(tp, pts, bound=a.bound), pts)
            ins = (pts.abs().amax(-1, keepdim=True) <= a.bound).to(den.dtype)
            return den * ins, rgb
        if a.n_fine > 0:
            rgb, acc = volume_render_importance(
                o, d, defaults.NEAR, defaults.FAR, a.n_samples, a.n_fine,
                q, bg_color=bg, chunk=a.ray_chunk)
        else:
            rgb, acc = volume_render_chunked(o, d, defaults.NEAR, defaults.FAR,
                                             a.n_samples, q, bg_color=bg,
                                             chunk=a.ray_chunk)
        V = c2w.shape[0]
        return (rgb.reshape(V, res, res, 3).permute(0, 3, 1, 2),
                acc.reshape(V, res, res, 1).permute(0, 3, 1, 2))

    def save(step, final=False):
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        torch.save({"triplanes": triplanes.detach().cpu(), "nerf": nerf.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "done": final,
                    "uids": ds.uids[:n], "cfg": vars(a),
                    "render_cfg": {"olcek_yontemi": imutil._yontem(),
                                   "bound": a.bound,
                                   "n_samples": a.n_samples,
                                   "tp_res": a.tp_res}}, a.out)

    start = 0
    if a.resume and os.path.isfile(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        if ck.get("done"):
            print(f"{a.out} zaten tamamlanmis (adim {ck['step']}); yeniden fit gerekmiyor.",
                  flush=True)
            return
        with torch.no_grad():
            # `.to(DEV)` YAZMA: tum bankanin GECICI bir GPU kopyasini ayirir
            # (tp256/24 obje = 604 MB). CPU->GPU `copy_` dogrudan H2D yapar,
            # gecici yoktur. 2026-09-02'de resume tam bu 604 MB yuzunden
            # ilk degerlendirmede OOM verdi (orijinal kosu 15,8/16,3 GB'de gidiyordu).
            triplanes.copy_(ck["triplanes"])
        nerf.load_state_dict(ck["nerf"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["step"]
        # ck ~1,8 GB CPU + yukleme sirasinda buyuyen ayirici parcalari tutuyor;
        # Adam durumlari (2 x 604 MB) burada, ilk backward'dan ONCE ayrildi ->
        # taze surecte bellek duzeni orijinal kosudan farkli. Temizle.
        del ck
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"resume: adim {start}/{a.steps}", flush=True)

    rng = np.random.default_rng(0); t0 = time.time()
    for step in range(start, a.steps):
        # 2026-08-29 DUZELTME: set_epoch() cagrilmiyordu -> LRMDataset RNG'si
        # donuktu (_epoch daima 0) ve her obje 16 goruumun SABIT 4'unden
        # ogreniliyordu; 12 render oluydu. Ayni hata train_lrm.py'de
        # duzeltilmis ama bu iki asamaya tasinmamisti.
        ds.set_epoch(step)
        opt.zero_grad(); agg = 0.0
        for _ in range(a.batch):
            i = int(rng.integers(n))
            it = ds[i]
            alpha = it["sup_alpha"].to(DEV); prem = it["sup_premult"].to(DEV)
            c = torch.rand(3, device=DEV)
            target = prem + (1 - alpha) * c[None, :, None, None]
            rgb, acc = render(triplanes[i], it["sup_c2w"].to(DEV),
                              it["sup_K"].to(DEV), c, int(it["sup_res"]))
            total, _ = loss_fn(rgb, acc, target, alpha)
            if a.w_tv > 0:
                total = total + a.w_tv * tv_loss(triplanes[i])
            (total / a.batch).backward(); agg += float(total) / a.batch
        opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            with torch.no_grad():
                ps = []
                for i in range(0, min(n, 24), 4):
                    it = ds_eval[i]
                    gt = it["sup_rgb"][:1].to(DEV)
                    r, ac = render(triplanes[i], it["sup_c2w"][:1].to(DEV),
                                   it["sup_K"][:1].to(DEV), None, a.eval_res)
                    ps.append(float(((r + (1 - ac) - gt) ** 2).mean()))
                psnr = -10 * math.log10(max(float(np.mean(ps)), 1e-9))
            el = time.time() - t0
            eta = (a.steps - step - 1) / max((step - start + 1) / el, 1e-6) / 3600
            print(f"  step {step:6d}/{a.steps} loss={agg:.4f} PSNR={psnr:.2f}dB "
                  f"[{(step-start+1)/el:.2f} it/s, kalan ~{eta:.1f} sa]", flush=True)
        if a.ckpt_every and step > start and step % a.ckpt_every == 0:
            save(step)
    save(a.steps, final=True)
    print(f"kaydedildi: {a.out}", flush=True)
    with torch.no_grad():
        rows = []
        for i in range(min(6, n)):     # n_obj<6 iken IndexError veriyordu
            it = ds_eval[i]
            gt = it["sup_rgb"][0].to(DEV)
            r, ac = render(triplanes[i], it["sup_c2w"][:1].to(DEV),
                           it["sup_K"][:1].to(DEV), None, a.eval_res)
            rows.append(torch.cat([gt, (r + (1 - ac))[0].clamp(0, 1)], -1).cpu())
    Image.fromarray((torch.cat(rows, 1).permute(1, 2, 0).numpy() * 255
                     ).astype(np.uint8)).save(PREVIEW)
    print(f"gorsel: {PREVIEW}", flush=True)


if __name__ == "__main__":
    main()
