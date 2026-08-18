"""Triplane: token izgarasi -> 3 duzlem (upsample) + nokta ornekleme."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneHead(nn.Module):
    def __init__(self, dim, out_channels=32, upsample=2):
        super().__init__()
        self.proj = nn.Linear(dim, out_channels)
        self.up = nn.ConvTranspose2d(out_channels, out_channels,
                                     kernel_size=upsample, stride=upsample)

    def forward(self, tp_grid):
        x = self.proj(tp_grid).permute(0, 3, 1, 2)  # (3, C, r, r)
        return self.up(x)                            # (3, C, r*up, r*up)


def sample_triplane(triplane, points, bound=0.6):
    p = (points / bound).clamp(-1, 1)   # (N,3)
    planes_coords = [p[:, [0, 1]], p[:, [0, 2]], p[:, [1, 2]]]  # XY, XZ, YZ
    feats = []
    for plane, coords in zip(triplane, planes_coords):
        grid = coords.view(1, -1, 1, 2)  # (1, N, 1, 2)
        f = F.grid_sample(plane[None], grid, mode="bilinear",
                          align_corners=True, padding_mode="border")  # (1,C,N,1)
        feats.append(f.squeeze(0).squeeze(-1).T)  # (N, C)
    return torch.cat(feats, dim=-1)  # (N, 3C)
