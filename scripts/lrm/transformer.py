"""Triplane token'lari -> triplane izgarasi.

Iki govde secenegi:
  - "joint" (eski): [image ; triplane] token'lari TEK self-attention yiginda birlesir.
    Sorun (olculdu): triplane token'lari goruntuye bakmak ZORUNDA degil; egitim
    ilerledikce image'a giden attention kutlesi 0.25 -> ~0.00'a dusuyor ve model
    girdiden bagimsiz "ortalama obje" basiyor.
  - "cross" (LRM/OpenLRM): her blokta triplane token'lari image token'lara
    CROSS-ATTENTION yapar (kacis yok) + kamera ozelligiyle modulasyonlu LayerNorm.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    """Eski birlesik self-attention blogu."""

    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class ModLN(nn.Module):
    """Modulasyonlu LayerNorm (adaLN): kamera ozelligi -> per-kanal scale/shift.
    Kamera bilgisi her katmanda yeniden enjekte edilir (LRM tarifi)."""

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)   # baslangicta saf LayerNorm

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class CondBlock(nn.Module):
    """LRM blogu: cross-attn(image) -> self-attn(triplane) -> MLP, hepsi modLN'li."""

    def __init__(self, dim, heads, cond_dim):
        super().__init__()
        self.n1 = ModLN(dim, cond_dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = ModLN(dim, cond_dim)
        self.selfattn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = ModLN(dim, cond_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x, img, cond):
        q = self.n1(x, cond)
        x = x + self.cross(q, img, img, need_weights=False)[0]
        h = self.n2(x, cond)
        x = x + self.selfattn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n3(x, cond))
        return x


class LRMTransformer(nn.Module):
    def __init__(self, dim=512, depth=12, heads=8, triplane_res=32, img_dim=384,
                 cross_attn=False, cam_dim=16):
        super().__init__()
        self.triplane_res = triplane_res
        self.n_tp = 3 * triplane_res * triplane_res
        self.tp_tokens = nn.Parameter(torch.randn(self.n_tp, dim) * 0.02)
        self.img_proj = nn.Linear(img_dim + 6, dim)
        self.cross_attn = cross_attn
        if cross_attn:
            self.cam_mlp = nn.Sequential(nn.Linear(cam_dim, dim), nn.SiLU(),
                                         nn.Linear(dim, dim))
            self.blocks = nn.ModuleList([CondBlock(dim, heads, dim) for _ in range(depth)])
        else:
            self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self._ckpt = False

    def enable_checkpointing(self):
        self._ckpt = True

    def forward(self, img_tokens, img_plucker, cam_feat=None):
        x_img = self.img_proj(torch.cat([img_tokens, img_plucker], dim=-1))  # (M,dim)
        if self.cross_attn:
            cond = self.cam_mlp(cam_feat).mean(0, keepdim=True)  # (1,dim)
            img = x_img[None]                                    # (1,M,dim)
            x = self.tp_tokens[None]                             # (1,n_tp,dim)
            for blk in self.blocks:
                if self._ckpt and self.training:
                    x = checkpoint(blk, x, img, cond, use_reentrant=False)
                else:
                    x = blk(x, img, cond)
            tp = self.norm(x[0])
        else:
            x = torch.cat([x_img, self.tp_tokens], dim=0)[None]
            for blk in self.blocks:
                if self._ckpt and self.training:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            tp = self.norm(x[0, -self.n_tp:])
        r = self.triplane_res
        return tp.reshape(3, r, r, self.dim)
