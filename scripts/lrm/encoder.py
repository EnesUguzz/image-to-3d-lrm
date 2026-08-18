"""Donuk DINOv2 ViT-S/14 encoder. Sadece ozellik cikarir, egitilmez."""
import torch
import torch.nn as nn

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoEncoder(nn.Module):
    def __init__(self, name="dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.embed_dim  # 384 (vits14)
        self.patch = 14
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        # donuk kalsin: yine de eval sabitle
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, imgs):
        x = (imgs - self.mean) / self.std
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]  # (V, N_patch, 384)
