"""YUZEY TESHISI: yeniden kurulan yuzey opak mi, derinlikte ne kadar kalin?

NEDEN VAR (2026-09-02, keskinlik teshisi):
Bulaniklik uc ayri yerden gelebilir ve caresi UCU DE FARKLI:
  (a) denetim detayi hic verilmemis   -> kirpma / denetim cozunurlugu
  (b) triplane izgarasi cok kaba      -> tp_res
  (c) yuzey derinlikte yayilmis       -> yogunluk keskinligi
`bench_keskinlik` uc durumu da "bulanik" diye raporlar. Bu betik (c)'yi
dogrudan olcer, boylece (c) elenince kalan iki adaya odaklanilabilir.

Olculen iki sayi:
  acc_ic   : on-plan isinlarinda toplam agirlik. <1 ise beyaz zemin renge
             karisir -> "yikanmis" gorunum. (Bu proje ucusuz uc kez `acc->0`
             cokusu yasadi; bkz. lrm/guards.py.)
  kalinlik : agirlik dagiliminin t boyunca std'si. Egik bir yuzeyde derinlik
             kalinligi DOGRUDAN yanal doku bulanikligina cevrilir.
             Hem hucre hem piksel cinsinden basilir; piksel cinsinden 1'in
             altiysa yuzey render cozunurlugunde keskin demektir.

Ilk kullanimda bulunan sonuc (docs/KESKINLIK-TESHISI.md §2):
  denetim 256px + 1000 guncelleme -> acc 0.997, kalinlik 1.35 px  => KESKIN
  uretim tarifesi (64px, 100 gun.) -> acc 0.988, kalinlik 3.64 px
yani (c) bir kok neden degil, denetim yoksunlugunun BELIRTISI.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm import cameras, defaults          # noqa: E402
from lrm.dataset import LRMDataset         # noqa: E402
from lrm.nerf import TriplaneNeRF          # noqa: E402
from lrm.renderer import render_with_t     # noqa: E402
from lrm.triplane import sample_triplane   # noqa: E402

DEV = "cuda"
# Obje dunya capi ~0.906 (denetim raporu: yaricap medyan 0.453) ve 256px
# render'da ~138 piksel genis (100 obje olcumu) => dunya birimi -> piksel.
PX_PER_DUNYA = 138.0 / 0.906


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="fit_teacher ciktilari")
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--n_obj", type=int, default=8)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n_isin", type=int, default=3000,
                    help="obje basina orneklenen on-plan isini (tam kare "
                         "256^2 x 320 ornek bellege sigmaz)")
    a = ap.parse_args()

    ds = LRMDataset(a.train_list, a.renders_dir, split="train", render_res=a.res,
                    n_sup=1, augment=False, deterministic=True)
    print(f"\nyuzey teshisi: {a.n_obj} obje | {a.res}px | "
          f"{a.n_isin} on-plan isini/obje")
    print(f"{'sistem':<30} {'acc_ic':>8} {'kalinlik':>10} {'=hucre':>8} "
          f"{'=px@%d' % a.res:>9}")
    print("-" * 70)
    g = torch.Generator().manual_seed(0)
    for p in a.ckpts:
        tk = torch.load(p, map_location="cpu", weights_only=False)
        cfg = tk.get("cfg", {})
        ch = cfg.get("tp_ch", 32)
        tpr = int(cfg.get("tp_res", 64))
        bnd = float(cfg.get("bound", defaults.BOUND))
        nerf = TriplaneNeRF(in_dim=3 * ch, hidden=cfg.get("nerf_hidden", 64),
                            layers=cfg.get("nerf_layers", 4),
                            pos_enc=cfg.get("pos_enc", 0),
                            pos_bound=bnd).to(DEV).eval()
        nerf.load_state_dict(tk["nerf"])
        # NYQUIST: isin adimi hucreden buyukse ince izgarali kol HAKSIZ olculur.
        ns_min = int(math.ceil((defaults.FAR - defaults.NEAR) * tpr / (2 * bnd)))
        ns = max(int(cfg.get("n_samples", defaults.N_SAMPLES)), ns_min)
        hucre = 2 * bnd / tpr
        accs, kal = [], []
        with torch.no_grad():
            for i in range(min(a.n_obj, len(tk["uids"]))):
                it = ds[i]
                tp = tk["triplanes"][i].to(DEV)
                o, d = cameras.rays_from_camera(it["sup_c2w"][0], it["sup_K"][0],
                                                a.res, a.res)
                al = it["sup_alpha"][0, 0].reshape(-1)
                fg = torch.nonzero(al > 0.99).flatten()
                if fg.numel() < 200:
                    continue
                sel = fg[torch.randperm(fg.numel(), generator=g)[:a.n_isin]]
                o, d = o[sel].to(DEV), d[sel].to(DEV)

                def q(pt):
                    den, rgb = nerf(sample_triplane(tp, pt, bound=bnd), pt)
                    ins = (pt.abs().amax(-1, keepdim=True) <= bnd).to(den.dtype)
                    return den * ins, rgb

                # 2x Nyquist: agirlik dagiliminin std'sini olcuyoruz, ornekleme
                # araligi kalinliktan buyuk olursa olcum tabanı yapay cikar.
                t = torch.linspace(defaults.NEAR, defaults.FAR, ns * 2,
                                   device=DEV).expand(o.shape[0], ns * 2).clone()
                _, acc, w = render_with_t(o, d, t, q, agirlik_dondur=True)
                accs.append(float(acc.mean()))
                pw = w / w.sum(1, keepdim=True).clamp(min=1e-6)
                mu = (pw * t).sum(1)
                var = (pw * (t - mu[:, None]) ** 2).sum(1)
                kal.append(float(var.clamp(min=0).sqrt().median()))
        if not accs:
            print(f"{os.path.basename(p)[:-3]:<30} {'--- on plan isini yok':>36}")
            continue
        A, K = float(np.mean(accs)), float(np.mean(kal))
        ad = f"{os.path.basename(p)[:-3]} [tp{tpr}]"
        print(f"{ad:<30} {A:>8.3f} {K:>10.4f} {K / hucre:>8.2f} "
              f"{K * PX_PER_DUNYA:>9.2f}")
    print()


if __name__ == "__main__":
    main()
