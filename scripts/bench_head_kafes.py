"""HEAD TABANI A/B -- ortusmeli cekirdek kafesi olduruyor mu?

NEDEN (2026-09-03, diag_kafes.py olcumu):
`TriplaneHead` = tek bir `ConvTranspose2d(k=2, s=2)`. k == s ⇒ ORTUSME SIFIR:
her token kendi 2x2 blogunu BAGIMSIZ uretir, komsu bloklar arasi hicbir
sureklilik yok. Olculdu (periyot-2 gucu / komsu bant medyani):

    ogretmen (serbest parametre, head YOK)   1.11   <- KONTROL
    distilled_v3 (head'den geciyor)          5.90
    last_V3      (head'den geciyor)         15.82

ve bu, orta bant korelasyonunun cokusuyle BIREBIR ayni sirada:
    kor_orta   0.292 -> 0.199 -> 0.024

HIPOTEZ: hata, head'in TABANI blok tabani oldugu icin blok sinirlarinda
birikiyor. Ortusmeli bir taban ayni serbestlik derecesiyle bunu dagitir.

⚠️ RUTBE DEGISMIYOR, TABAN DEGISIYOR. Bagimsiz denetim hakli olarak
"ortusme toplam serbestlik sinirini degistirmez" dedi (3 x 32^2 x 512 =
1.572.864, kernel ne olursa olsun). Ama artefakti ureten sey sinir degil
TABAN. Bu test tam o ayrimi olcer.

KURULUM (transformer/encoder YOK -- sadece head):
Obje basina SERBEST token izgarasi (3, 32, 32, dim) ogrenilir ve head'den
gecirilerek ogretmenin triplane'ine regresyon edilir. Bu, head'in
ULASABILECEGI EN IYI sonuctur: kosullandirma, kapasite, veri -- hepsi
devre disi. Kalan tek degisken head tabani.

KOLLAR:
  k2s2     : mevcut (ConvTranspose2d k=2, s=2)            -- ORTUSME YOK
  k4s2     : ConvTranspose2d k=4, s=2, padding=1          -- her cikis 4 token
  up_conv  : nearest upsample x2 + Conv2d 3x3             -- Odena vd. onerisi

ON-KAYITLI KAPI (kosudan ONCE yazildi):
  Nyquist orani: k2s2 >> 1 VE (k4s2 ya da up_conv) < 2  -> HIPOTEZ DOGRULANDI,
      head degistirilir (ve zincir yeniden kosar).
  Ucu de > 2                                            -> kafes head'den DEGIL,
      baska yerden (ornekleme/renderer) geliyor; teshis yeniden.
  Ucu de ~1                                             -> kafes egitim dinamigi,
      taban degil; `rel` farkina bak.
  `rel` (uyum hatasi) ayrica raporlanir: taban degisiminin bedeli var mi?
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_kafes import nyquist_orani  # noqa: E402


class Head(nn.Module):
    """Uc taban da AYNI cikti seklini ve AYNI baslangic olcegini kullanir."""

    HEDEF_STD = 0.24     # TriplaneHead.HEDEF_STD ile ayni

    def __init__(self, tip, dim=512, ch=32, up=2):
        super().__init__()
        self.tip = tip
        if tip == "k2s2":
            self.op = nn.ConvTranspose2d(dim, ch, kernel_size=up, stride=up)
            fan = dim
        elif tip == "k4s2":
            self.op = nn.ConvTranspose2d(dim, ch, kernel_size=2 * up,
                                         stride=up, padding=up // 2)
            fan = dim * 4          # her cikis pikseline 4 token katkisi
        elif tip == "up_conv":
            self.op = nn.Conv2d(dim, ch, kernel_size=3, padding=1)
            self.up = up
            fan = dim * 9
        else:
            raise ValueError(tip)
        nn.init.normal_(self.op.weight, std=self.HEDEF_STD / math.sqrt(fan))
        nn.init.zeros_(self.op.bias)

    def forward(self, g):                    # g: (3, r, r, dim)
        x = g.permute(0, 3, 1, 2)
        if self.tip == "up_conv":
            x = F.interpolate(x, scale_factor=self.up, mode="nearest")
        return self.op(x)                    # (3, ch, r*up, r*up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="dataset/lrm_bench/TAVAN2_K64.pt")
    ap.add_argument("--n_obj", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kollar", nargs="*", default=["k2s2", "k4s2", "up_conv"])
    ap.add_argument("--zaman_testi", type=int, default=0,
                    help=">0 ise sadece bu kadar adim kosup sn/adim basar")
    a = ap.parse_args()
    DEV = a.device
    torch.manual_seed(0)

    tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
    hedef = tk["triplanes"][:a.n_obj].to(DEV).float()      # (N,3,ch,H,W)
    N, _, ch, H, W = hedef.shape
    up = H // a.grid
    hvar = hedef.var().item()
    print(f"head tabani A/B: {N} obje | hedef {tuple(hedef.shape[1:])} "
          f"| token izgarasi 3x{a.grid}^2x{a.dim} | up={up} | cihaz {DEV}")
    print(f"  serbest token/obje = 3*{a.grid}^2*{a.dim} = {3*a.grid*a.grid*a.dim:,}"
          f"  | hedef/obje = 3*{H}^2*{ch} = {3*H*W*ch:,}")

    sonuc = {}
    for tip in a.kollar:
        torch.manual_seed(0)
        head = Head(tip, dim=a.dim, ch=ch, up=up).to(DEV)
        tok = nn.Parameter(torch.randn(N, 3, a.grid, a.grid, a.dim,
                                       device=DEV) * 0.02)
        opt = torch.optim.Adam([{"params": head.parameters(), "lr": a.lr},
                                {"params": [tok], "lr": a.lr * 10}])
        adim = a.zaman_testi or a.steps
        t0 = time.time()
        for s in range(adim):
            opt.zero_grad()
            kayip = 0.0
            for i in range(N):
                out = head(tok[i])
                kayip = kayip + ((out - hedef[i]) ** 2).mean() / hvar
            (kayip / N).backward()
            opt.step()
        dt = time.time() - t0
        if a.zaman_testi:
            print(f"  {tip:<9} {dt/adim:.3f} sn/adim  => {a.steps} adim = "
                  f"{dt/adim*a.steps/60:.1f} dk/kol")
            continue
        with torch.no_grad():
            rel, nyq = [], []
            for i in range(N):
                out = head(tok[i])
                rel.append(float(((out - hedef[i]) ** 2).mean() / hvar))
                for d in range(3):
                    nyq.append(nyquist_orani(out[d])[0])
        sonuc[tip] = (float(np.mean(rel)), float(np.mean(nyq)),
                      float(np.std(nyq)), dt)
        print(f"  {tip:<9} rel={np.mean(rel):.4f}  Nyq={np.mean(nyq):.2f}"
              f" (+-{np.std(nyq):.2f})  [{dt/60:.1f} dk]", flush=True)

    if a.zaman_testi:
        return
    # hedefin kendi Nyquist orani = KONTROL (serbest parametre, head yok)
    nyq_h = [nyquist_orani(hedef[i, d])[0]
             for i in range(N) for d in range(3)]
    print(f"\n{'kol':<12} {'rel':>8} {'Nyquist':>9}   yorum")
    print("-" * 58)
    print(f"{'HEDEF':<12} {0.0:>8.4f} {np.mean(nyq_h):>9.2f}   kontrol (head yok)")
    for tip, (r, n_, s_, _) in sonuc.items():
        yorum = "kafes YOK" if n_ < 2 else ("KAFES" if n_ > 2 else "sinirda")
        print(f"{tip:<12} {r:>8.4f} {n_:>9.2f}   {yorum}")
    print("-" * 58)
    print("KAPI: k2s2 >> 1 VE (k4s2 ya da up_conv) < 2 ise hipotez dogrulandi.")
    print("      `rel` sutunu tabanin bedelini gosterir (dusuk = daha iyi uyum).")


if __name__ == "__main__":
    main()
