"""Gercek foto -> LRM -> girdi goruumu yeniden izdusumu + yeni acilar.

ON-KAYITLI OLCUT (kosudan once yazildi):
  Gercek fotoda 3D GT yok. Olculebilir tek durust sayi, tahmin edilen
  hacmi GIRDI ACISINDAN render edip fotonun kendi siluetiyle karsilastirmak.
  Ayni metrigin in-domain degeri BILINIYOR: test seti girdi goruumu IoU 0.780.

  IoU >= 0.60  -> domain boslugu yonetilebilir, Faz C'ye gec
  0.45 - 0.60  -> girdi augmentation'i ile bir finetune turu gerek
  IoU <  0.45  -> domain boslugu baskin, mimari turu ERTELENIR

KAMERA VARSAYIMI: gercek fotonun pozu bilinmiyor. Egitimin kanonik[0]
pozu varsayilir (azimuth 0, elevation 20). Faz C de bunu yapacak.
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lrm import cameras, defaults
from lrm.model import LRM

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def canonical_pose(renders_dir, idx=0):
    """Kanonik[idx] kamerasi -- TUM egitim objelerinde ayni."""
    uid = sorted(os.listdir(renders_dir))[0]
    with open(os.path.join(renders_dir, uid, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    v = meta["views"][meta["canonical_indices"][idx]]
    return v["intrinsic"], v["extrinsic"], meta.get("resolution", 512)


def load_prepped(path, res):
    im = Image.open(path).convert("RGBA").resize((res, res), Image.LANCZOS)
    a = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0
    return a                                    # [4,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--photo", required=True, help="prep_photo.py ciktisi RGBA png")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--res", type=int, default=256, help="render cozunurlugu")
    ap.add_argument("--in_res", type=int, default=defaults.INPUT_RES,
                    help="encoder girdi cozunurlugu. TEK KAYNAK "
                         "lrm/defaults.py; sabit 224 yazmak, INPUT_RES "
                         "degistiginde Plucker haritasiyla sessiz "
                         "ayrismaya yol acar.")
    ap.add_argument("--out", default="dataset/gercek_foto/sonuc")
    ap.add_argument("--tag", default="foto")
    a = ap.parse_args()

    model = LRM().to(DEV).eval()
    tk = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(tk["model"])
    step = tk.get("step", "?")

    Kraw, Eraw, master = canonical_pose(a.renders_dir, 0)
    K = cameras.scale_intrinsics(torch.tensor(Kraw, dtype=torch.float32),
                                 master, a.in_res)[None].to(DEV)
    c2w = torch.linalg.inv(torch.tensor(Eraw, dtype=torch.float32))[None].to(DEV)

    rgba = load_prepped(a.photo, a.in_res)
    img = (rgba[:3] * rgba[3:4] + (1 - rgba[3:4]))[None].to(DEV)   # beyaz kompozit
    gt_alpha = (rgba[3] > 0.5).numpy()

    with torch.no_grad():
        triplane = model.make_triplane(img, c2w, K)

        # --- GIRDI ACISINDAN yeniden izdusum (on-kayitli olcut) ---
        Kr = cameras.scale_intrinsics(torch.tensor(Kraw, dtype=torch.float32),
                                      master, a.res)[None].to(DEV)
        rgb_in, acc_in = model.render_view(triplane, c2w[0], Kr[0], a.res, a.res)

        # --- yeni acilar: diger 3 kanonik ---
        novels = []
        for i in (1, 2, 3):
            Kn_raw, En_raw, _ = canonical_pose(a.renders_dir, i)
            cn = torch.linalg.inv(torch.tensor(En_raw, dtype=torch.float32))[None].to(DEV)
            Kn = cameras.scale_intrinsics(torch.tensor(Kn_raw, dtype=torch.float32),
                                          master, a.res)[None].to(DEV)
            rgb_n, _ = model.render_view(triplane, cn[0], Kn[0], a.res, a.res)
            novels.append(rgb_n)

    # siluet IoU: tahmin (acc) vs fotonun alfasi
    pred_m = (acc_in[0].float().cpu().numpy() > 0.5)
    gt_m = np.array(Image.fromarray(gt_alpha).resize((a.res, a.res), Image.NEAREST))
    inter = (pred_m & gt_m).sum(); union = (pred_m | gt_m).sum()
    iou = float(inter) / float(union) if union else 0.0
    kapl_gt, kapl_pred = gt_m.mean(), pred_m.mean()

    os.makedirs(a.out, exist_ok=True)
    def to_img(t):
        x = t.clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray((x * 255).astype(np.uint8)).convert("RGB")

    tiles = [Image.open(a.photo).convert("RGBA")]
    bgw = Image.new("RGBA", tiles[0].size, (255, 255, 255, 255))
    tiles[0] = Image.alpha_composite(bgw, tiles[0]).convert("RGB").resize((a.res, a.res))
    tiles += [to_img(rgb_in)] + [to_img(n) for n in novels]
    W = Image.new("RGB", (a.res * len(tiles), a.res), (255, 255, 255))
    for k, t in enumerate(tiles): W.paste(t, (a.res * k, 0))
    grid = os.path.join(a.out, f"{a.tag}.png")
    W.save(grid)

    print(f"[{a.tag}] ckpt adim {step}")
    print(f"  GIRDI GORUUMU SILUET IoU = {iou:.4f}   (in-domain referans 0.780)")
    print(f"  kaplama: foto {kapl_gt:.4f}  tahmin {kapl_pred:.4f}")
    kapi = ("GECTI (>=0.60) -> Faz C" if iou >= 0.60 else
            "ARA BOLGE (0.45-0.60) -> finetune gerek" if iou >= 0.45 else
            "KALDI (<0.45) -> domain boslugu baskin")
    print(f"  ON-KAYITLI KAPI: {kapi}")
    print(f"  -> {grid}   (sira: foto | girdi acisi | yan | arka | diger yan)")
    with open(os.path.join(a.out, f"{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": a.tag, "ckpt": a.ckpt, "step": str(step), "iou": iou,
                   "kaplama_foto": float(kapl_gt), "kaplama_tahmin": float(kapl_pred),
                   "kapi": kapi}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
