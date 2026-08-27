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


def tv_loss(triplane):
    """Toplam-varyasyon: komsu triplane hucreleri arasi ani sicramalari cezalar.

    NEDEN ORTAK YERDE (2026-08-26): `fit_teacher.py` bunu HIC uygulamiyordu,
    `train_lrm.py`/`bench_overfit.py` ise `w_tv=5e-4` ile uyguluyordu. Sonuc:
    ogretmen triplane'lerinin varyansinin **%40.5'i** yuksek frekansti ve o
    yuksek frekans render kalitesine sadece **0.34 dB** katkı veriyordu -- yani
    bilgi degil gurultu. Bu gurultu iki yerden vuruyordu:
      1) mesh: Euler -92 (47 tunel), komsu ucgen aci p90 38 derece
      2) DISTILASYON HEDEFI: rel = ((tp-hedef)^2)/hedef.var() icin ~0.40'lik
         OGRENILEMEZ taban olusturuyor => "kapasite siniri" kapisi gecersiz oluyor
    """
    dh = (triplane[..., 1:, :] - triplane[..., :-1, :]).abs().mean()
    dw = (triplane[..., :, 1:] - triplane[..., :, :-1]).abs().mean()
    return dh + dw
