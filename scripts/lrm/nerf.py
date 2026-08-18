"""Triplane ozelliginden yogunluk + renk ureten kucuk MLP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneNeRF(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.density_head = nn.Linear(hidden, 1)
        self.rgb_head = nn.Linear(hidden, 3)

    def forward(self, feats):
        h = self.backbone(feats)
        density = F.softplus(self.density_head(h))
        rgb = torch.sigmoid(self.rgb_head(h))
        return density, rgb
