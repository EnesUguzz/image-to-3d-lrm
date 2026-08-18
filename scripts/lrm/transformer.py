"""Image token'lari + ogrenilebilir triplane token'lari tek self-attention
govdesinde birlesir. Cikis: triplane token izgarasi."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
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


class LRMTransformer(nn.Module):
    def __init__(self, dim=512, depth=12, heads=8, triplane_res=32, img_dim=384):
        super().__init__()
        self.triplane_res = triplane_res
        self.n_tp = 3 * triplane_res * triplane_res
        self.tp_tokens = nn.Parameter(torch.randn(self.n_tp, dim) * 0.02)
        self.img_proj = nn.Linear(img_dim + 6, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self._ckpt = False

    def enable_checkpointing(self):
        self._ckpt = True

    def forward(self, img_tokens, img_plucker):
        x_img = self.img_proj(torch.cat([img_tokens, img_plucker], dim=-1))  # (M,dim)
        x = torch.cat([x_img, self.tp_tokens], dim=0)[None]  # (1, M+n_tp, dim)
        for blk in self.blocks:
            if self._ckpt and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        tp = self.norm(x[0, -self.n_tp:])  # (n_tp, dim)
        r = self.triplane_res
        return tp.reshape(3, r, r, self.dim)
