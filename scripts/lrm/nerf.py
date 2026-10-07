"""Triplane ozelliginden yogunluk + renk ureten kucuk MLP."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneNeRF(nn.Module):
    def __init__(self, in_dim, hidden=64, density_bias=0.0, noise_std=0.0,
                 layers=4, pos_enc=0, pos_bound=0.6):
        """layers: gizli katman sayisi (referans OpenLRM model kartı: 4).

        2026-08-29: onceki deger 2'ydi. Triplane ozelligini yogunluk/renge
        ceviren TEK dogrusal-olmayan blok bu; referansin yarisindaydik.
        """
        super().__init__()
        # KONUM KODLAMASI (2026-08-31). Girdi simdiye kadar SADECE
        # interpolasyonlu triplane ozelligiydi. grid_sample bilineer =>
        # ozellik alani parcali-DOGRUSAL; iki hucre arasinda MLP'nin
        # gorebilecegi tek sey dogrusal bir gecis. Yani hucre altinda yapi
        # uretecek hicbir mekanizma yoktu. Literaturde frekans kodlamasi tam
        # olarak "dusuk uzamsal cozunurlugu telafi etmek" icin ekleniyor.
        # pos_enc = frekans bandi sayisi (0 = kapali, davranis degismez).
        self.pos_enc = int(pos_enc)
        self.pos_bound = float(pos_bound)
        if self.pos_enc > 0:
            self.register_buffer(
                "_freq", 2.0 ** torch.arange(self.pos_enc).float() * math.pi,
                persistent=False)
            in_dim = in_dim + 3 * 2 * self.pos_enc
        seq = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(max(0, layers - 1)):
            seq += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        self.backbone = nn.Sequential(*seq)
        self.density_head = nn.Linear(hidden, 1)
        self.rgb_head = nn.Linear(hidden, 3)
        # foggy pozitif baslangic
        self.density_bias = density_bias
        # egitimde raw gurultu (NeRF raw_noise_std): density'nin sert 0/inf
        # doygunluguna oturmasini engeller => softplus gradyani olmez, model
        # ne 'bos sahne'ye ne 'sisli dolgu'ya kalici cokemez
        self.noise_std = noise_std

    def forward(self, feats, pts=None):
        if self.pos_enc > 0:
            if pts is None:
                raise ValueError(
                    "pos_enc>0 ama pts verilmedi. Cagiran taraf nerf(feats, pts) "
                    "kullanmali; sessizce kodlamasiz calismak, bayragin ACIK "
                    "sanildigi bir kosu uretirdi.")
            p = (pts / self.pos_bound).clamp(-1, 1)          # (N,3)
            a = p[..., None] * self._freq                    # (N,3,L)
            enc = torch.cat([a.sin(), a.cos()], -1).flatten(-2)   # (N,3*2L)
            feats = torch.cat([feats, enc.to(feats.dtype)], -1)
        h = self.backbone(feats)
        raw = self.density_head(h) + self.density_bias
        if self.training and self.noise_std > 0:
            raw = raw + torch.randn_like(raw) * self.noise_std
        density = F.softplus(raw)
        rgb = torch.sigmoid(self.rgb_head(h))
        return density, rgb
