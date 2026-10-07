"""GIRDI ELEVATION DUYARLILIK EGRISI  (Asama 0.1)

SORU: model girdi elevation'ina ne kadar duyarli? Egitimde girdi DAIMA +20
oldugu icin (dataset.py, input_pool="canon"), gercek bir telefon fotosu
(tipik 30-60 derece) dagitim disi kaliyor olabilir.

TASARIM -- tek degisken:
  HEDEFLER daima ayni: kanonik[1], [2], [3]  (azimuth 90/180/270, elev 20)
  GIRDI degisir      : kanonik[0] (elev 20, EGITILEN kosul)  ya da
                       secilen elevation bandindan rastgele bir ek gorunum
Boylece olculen tek sey girdi acisinin etkisi.

Referans: SV3D App. G.2 Tablo 8 (statik <-> sine-30 <-> sine-50 ablasyonu),
One-2-3-45'in elevation kestirim modulunun varlik gerekcesi.
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm import cameras, defaults
from lrm.compat import load_lrm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BANTLAR = [("EGITILEN +20", None), ("[10,25)", (10, 25)), ("[25,40)", (25, 40)),
           ("[40,55)", (40, 55)), ("[55,75)", (55, 75))]


def elev_of(v):
    e = v.get("elevation_deg")
    if e is not None:
        return float(e)
    E = np.array(v["extrinsic"], dtype=np.float64)
    p = np.linalg.inv(E)[:3, 3]
    return float(np.degrees(np.arcsin(p[2] / np.linalg.norm(p))))


def load_view(R, uid, v, res, master):
    im = Image.open(os.path.join(R, uid, v["file"])).convert("RGBA")
    im = im.resize((res, res), Image.LANCZOS)
    t = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.
    rgb = t[:3] * t[3:4] + (1 - t[3:4])
    K = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                 master, res)
    c2w = torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float32))
    return rgb, (t[3] > 0.5), K, c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_obj", type=int, default=40)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="elev")
    ap.add_argument("--n_samples", type=int, default=None, help="adil karsilastirma icin egitim degeri")
    a = ap.parse_args()

    _kw = {} if a.n_samples is None else {'n_samples': a.n_samples}
    model, arch, tk = load_lrm(a.ckpt, device=DEV, **_kw)
    print(f"ckpt {a.ckpt} adim={tk.get('step','?')} mimari={arch}")
    print(f"girdi cozunurlugu {defaults.INPUT_RES} | n_samples {model.n_samples}")
    uids = json.load(open(a.train_list, encoding="utf-8"))[a.split][:a.n_obj]
    rng = np.random.RandomState(a.seed)
    R, ir = a.renders_dir, defaults.INPUT_RES

    sonuc = {ad: {"psnr": [], "iou": []} for ad, _ in BANTLAR}
    for uid in uids:
        meta = json.load(open(f"{R}/{uid}/meta.json", encoding="utf-8"))
        views, ci = meta["views"], meta["canonical_indices"]
        master = meta.get("resolution", 512)
        # HEDEFLER sabit: kanonik 1,2,3
        tg = [load_view(R, uid, views[ci[k]], a.res, master) for k in (1, 2, 3)]
        G = torch.stack([t[0] for t in tg]).to(DEV)
        Ga = torch.stack([torch.from_numpy(t[1].numpy().astype(np.float32))[None] for t in tg]).to(DEV)
        tK = torch.stack([t[2] for t in tg]).to(DEV)
        tc = torch.stack([t[3] for t in tg]).to(DEV)

        for ad, band in BANTLAR:
            if band is None:
                iv = views[ci[0]]
            else:
                aday = [v for i, v in enumerate(views)
                        if i not in ci and band[0] <= elev_of(v) < band[1]]
                if not aday:
                    continue
                iv = aday[rng.randint(len(aday))]
            irgb, _, iK, ic = load_view(R, uid, iv, ir, master)
            with torch.no_grad():
                rgb, acc = model(irgb[None].to(DEV), ic[None].to(DEV), iK[None].to(DEV),
                                 tc, tK, (a.res, a.res),
                                 bg_color=torch.ones(3, device=DEV))
            mse = ((rgb.float() - G) ** 2).mean(dim=(1, 2, 3))
            sonuc[ad]["psnr"] += [float(x) for x in (10 * torch.log10(1.0 / mse.clamp(min=1e-10)))]
            pm = (acc.float() > 0.5); gm = (Ga > 0.5)
            inter = (pm & gm).flatten(1).sum(1).float()
            union = (pm | gm).flatten(1).sum(1).float().clamp(min=1)
            sonuc[ad]["iou"] += [float(x) for x in (inter / union)]

    print(f"\n{len(uids)} test objesi | hedefler SABIT (kanonik 1,2,3) | tek degisken: GIRDI acisi")
    print(f"\n{'girdi elevation':<18}{'n':>5}{'PSNR':>9}{'IoU':>9}{'PSNR farki':>12}")
    print("-" * 55)
    taban = None
    for ad, _ in BANTLAR:
        d = sonuc[ad]
        if not d["psnr"]:
            print(f"{ad:<18}{'-':>5}  (bu bantta gorunum yok)"); continue
        p, i = float(np.mean(d["psnr"])), float(np.mean(d["iou"]))
        if taban is None:
            taban = p
        print(f"{ad:<18}{len(d['psnr']):>5}{p:>9.2f}{i:>9.3f}{p - taban:>+12.2f}")
    os.makedirs("dataset/lrm_eval", exist_ok=True)
    out = f"dataset/lrm_eval/elev_{a.tag}.json"
    json.dump({"ckpt": a.ckpt, "arch": arch, "n_obj": len(uids),
               "sonuc": {k: {m: float(np.mean(v)) if v else None for m, v in d.items()}
                         for k, d in sonuc.items()}},
              open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
