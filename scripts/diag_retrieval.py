"""Kesin conditioning testi: N objeyi AYNI view index'inden render et,
pred_i'yi tum gt_j'lerle karsilastir. Kosegen en kucuk mu?"""
import os, sys, json
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm import cameras
from PIL import Image

CKPT = "dataset/lrm_ckpts/last.pt"; DEV = "cuda"; RES = 64
SUP_VIEW = 6   # kanonik olmayan sabit bir gorunum
IN_VIEW = 0    # tek girdi: kanonik on


def load(uid, ds, view, res):
    meta = ds._meta(uid); v = meta["views"][view]
    im = Image.open(os.path.join(ds.renders_dir, uid, v["file"])).convert("RGBA")
    a = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.
    a = F.interpolate(a[None], size=(res, res), mode="bilinear", align_corners=False)[0]
    rgb = a[:3] * a[3:4] + (1 - a[3:4])
    K = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                 meta.get("resolution", 512), res)
    c2w = torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float32))
    return rgb, c2w, K, a[3:4]


@torch.no_grad()
def main():
    ds = LRMDataset("dataset/train_list.json", "dataset/renders", split="train",
                    render_res=RES, n_sup=4, augment=False)
    uids = ds.uids[:12]
    m = LRM(n_samples=48).to(DEV).eval()
    st = torch.load(CKPT, map_location="cpu", weights_only=False); m.load_state_dict(st["model"])
    print(f"ckpt step {st['step']} | {len(uids)} train objesi | girdi view {IN_VIEW}, hedef view {SUP_VIEW}")

    preds, gts = [], []
    for u in uids:
        inp, ic2w, iK, _ = load(u, ds, IN_VIEW, 224)
        gt, tc2w, tK, _ = load(u, ds, SUP_VIEW, RES)
        rgb, _ = m(inp[None].to(DEV), ic2w[None].to(DEV), iK[None].to(DEV),
                   tc2w[None].to(DEV), tK[None].to(DEV), (RES, RES))
        preds.append(rgb[0].cpu()); gts.append(gt)
    P, G = torch.stack(preds), torch.stack(gts)
    D = ((P[:, None] - G[None]) ** 2).mean((2, 3, 4))    # (i pred, j gt)
    diag = D.diag()
    off = (D.sum(1) - diag) / (len(uids) - 1)
    rank = (D < diag[:, None]).sum(1)                    # kac gt daha yakin
    print(f"\nkosegen (dogru obje) MSE ort = {diag.mean():.4f}")
    print(f"kosegen-disi (yanlis obje) MSE ort = {off.mean():.4f}")
    print(f"top-1 retrieval dogrulugu = {(rank == 0).float().mean():.2%}  (sans = {1/len(uids):.1%})")
    print(f"ortalama sira = {rank.float().mean():.2f} / {len(uids)-1}")
    # baseline: tum pred'lerin ORTALAMASI her gt'ye ne kadar yakin?
    Pm = P.mean(0, keepdim=True)
    dmean = ((Pm - G) ** 2).mean((1, 2, 3))
    print(f"\n[ortalama-pred] her gt'ye MSE = {dmean.mean():.4f}  (kosegen {diag.mean():.4f})")
    print(f"pred'ler birbirine benzerligi: pred-arasi MSE ort = "
          f"{((P[:,None]-P[None])**2).mean((2,3,4)).sum()/(len(uids)*(len(uids)-1)):.4f}")
    print(f"gt-arasi MSE ort = {((G[:,None]-G[None])**2).mean((2,3,4)).sum()/(len(uids)*(len(uids)-1)):.4f}")
    grid = torch.cat([torch.cat([G[i], P[i]], -1) for i in range(len(uids))], 1)
    Image.fromarray((grid.permute(1,2,0).numpy()*255).astype(np.uint8)).save(
        "dataset/lrm_val_previews/diag_retrieval.png")


main()
