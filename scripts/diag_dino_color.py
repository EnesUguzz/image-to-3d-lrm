"""Donuk DINOv2 patch token'larindan patch ortalama RGB'si geri kurtarilabiliyor mu?
Dusuk R^2 => encoder rengi atmis => model rengi ogrenemez (kahverengi ortalama)."""
import os, sys, json, random
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm.encoder import DinoEncoder

DEV = "cuda"; RES = 224; PATCH = 14


@torch.no_grad()
def main():
    uids = json.load(open("dataset/train_list.json", encoding="utf-8"))["train"]
    random.seed(0); uids = random.sample(uids, 150)
    enc = DinoEncoder().to(DEV).eval()
    X, Y = [], []
    for u in uids:
        meta = json.load(open(f"dataset/renders/{u}/meta.json", encoding="utf-8"))
        v = meta["views"][0]
        im = Image.open(f"dataset/renders/{u}/{v['file']}").convert("RGBA")
        a = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.
        a = F.interpolate(a[None], size=(RES, RES), mode="bilinear", align_corners=False)[0]
        rgb = a[:3] * a[3:4] + (1 - a[3:4])
        tok = enc(rgb[None].to(DEV))[0]                       # (256, 384)
        # patch ortalama rgb + alpha kaplama
        pr = F.avg_pool2d(rgb[None], PATCH)[0].reshape(3, -1).T  # (256,3)
        al = F.avg_pool2d(a[3:4][None], PATCH)[0].reshape(1, -1).T
        X.append(tok.cpu()); Y.append(torch.cat([pr, al], 1))
    X = torch.cat(X).double(); Y = torch.cat(Y).double()
    n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = idx[:int(.8*n)], idx[int(.8*n):]
    Xtr = torch.cat([X[tr], torch.ones(len(tr), 1).double()], 1)
    Xte = torch.cat([X[te], torch.ones(len(te), 1).double()], 1)
    W = torch.linalg.lstsq(Xtr.T @ Xtr + 1e-2 * torch.eye(Xtr.shape[1]).double(),
                           Xtr.T @ Y[tr]).solution
    P = Xte @ W
    for i, name in enumerate(["R", "G", "B", "alpha"]):
        yt = Y[te][:, i]
        r2 = 1 - ((P[:, i] - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        print(f"  {name}: lineer prob R^2 = {r2:.3f}   (rmse={((P[:,i]-yt)**2).mean().sqrt():.4f})")
    # sadece obje patchleri (alpha>0.5) icin renk R^2
    m = Y[te][:, 3] > 0.5
    for i, name in enumerate(["R", "G", "B"]):
        yt = Y[te][m][:, i]
        r2 = 1 - ((P[m][:, i] - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        print(f"  [sadece obje pikselleri] {name}: R^2 = {r2:.3f}")
    print(f"\n  ornek: {len(X)} patch, {len(uids)} obje, token dim {X.shape[1]}")


main()
