"""Triplane ozelliginden yogunluk + renk ureten kucuk MLP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneNeRF(nn.Module):
    def __init__(self, in_dim, hidden=64, density_bias=0.0, noise_std=0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.density_head = nn.Linear(hidden, 1)
        self.rgb_head = nn.Linear(hidden, 3)
        # foggy pozitif baslangic
        self.density_bias = density_bias
        # egitimde raw gurultu (NeRF raw_noise_std): density'nin sert 0/inf
        # doygunluguna oturmasini engeller => softplus gradyani olmez, model
        # ne 'bos sahne'ye ne 'sisli dolgu'ya kalici cokemez
        self.noise_std = noise_std

    def forward(self, feats):
        h = self.backbone(feats)
        raw = self.density_head(h) + self.density_bias
        if self.training and self.noise_std > 0:
            raw = raw + torch.randn_like(raw) * self.noise_std
        density = F.softplus(raw)
        rgb = torch.sigmoid(self.rgb_head(h))
        return density, rgb
