"""Uctan uca LRM: girdi foto(lar) + poz -> triplane -> supervision render."""
import torch
import torch.nn as nn

from lrm import cameras
from lrm.transformer import LRMTransformer
from lrm.triplane import TriplaneHead, sample_triplane
from lrm.nerf import TriplaneNeRF
from lrm.renderer import volume_render
from lrm.encoder import DinoEncoder


class LRM(nn.Module):
    def __init__(self, dim=512, depth=12, heads=8, triplane_res=32,
                 triplane_ch=32, nerf_hidden=64, encoder=None,
                 bound=0.6, near=0.8, far=2.2, n_samples=64, white_bg=True):
        super().__init__()
        self.encoder = encoder if encoder is not None else DinoEncoder()
        self.transformer = LRMTransformer(dim=dim, depth=depth, heads=heads,
                                          triplane_res=triplane_res,
                                          img_dim=self.encoder.embed_dim)
        self.triplane_head = TriplaneHead(dim=dim, out_channels=triplane_ch)
        self.nerf = TriplaneNeRF(in_dim=3 * triplane_ch, hidden=nerf_hidden)
        self.bound, self.near, self.far, self.n_samples = bound, near, far, n_samples
        self.white_bg = white_bg  # beyaz arka plan: rgb'nin siyaha cokme tuzagini onler
        self.patch = self.encoder.patch

    def make_triplane(self, input_imgs, input_c2w, input_K):
        tok = self.encoder(input_imgs)            # (Vi, P, 384)
        Vi, P, _ = tok.shape
        side = int(P ** 0.5)                       # 16 (224/14)
        pl_list = []
        for i in range(Vi):
            Kp = cameras.scale_intrinsics(input_K[i], 224, side)
            pl = cameras.plucker_map(input_c2w[i], Kp, side, side)  # (side,side,6)
            pl_list.append(pl.reshape(-1, 6))
        plucker = torch.stack(pl_list).reshape(Vi * P, 6).to(tok.device)
        img_tokens = tok.reshape(Vi * P, -1)
        tp_grid = self.transformer(img_tokens, plucker)   # (3,r,r,dim)
        return self.triplane_head(tp_grid)                # (3,C,H,W)

    def render_view(self, triplane, c2w, K, H, W):
        o, d = cameras.rays_from_camera(c2w, K, H, W)
        o, d = o.to(triplane.device), d.to(triplane.device)

        def query(pts):
            feats = sample_triplane(triplane, pts, bound=self.bound)
            density, rgb = self.nerf(feats)
            # sinirli obje: bound kubu disindaki noktalar tanim geregi BOS.
            # (aksi halde objeyi iskalayan arka plan isinlari triplane kenarindan
            #  density toplayip 'sisli dolgu' yapiyordu -> siluet olusmuyordu.)
            inside = (pts.abs().amax(dim=-1, keepdim=True) <= self.bound).to(density.dtype)
            return density * inside, rgb

        rgb, acc = volume_render(o, d, self.near, self.far, self.n_samples, query,
                                 white_bg=self.white_bg)
        rgb = rgb.reshape(H, W, 3).permute(2, 0, 1)
        acc = acc.reshape(H, W, 1).permute(2, 0, 1)
        return rgb, acc

    def forward(self, input_imgs, input_c2w, input_K,
                render_c2w, render_K, render_hw):
        H, W = render_hw
        triplane = self.make_triplane(input_imgs, input_c2w, input_K)
        rgbs, accs = [], []
        for i in range(render_c2w.shape[0]):
            rgb, acc = self.render_view(triplane, render_c2w[i], render_K[i], H, W)
            rgbs.append(rgb)
            accs.append(acc)
        return torch.stack(rgbs), torch.stack(accs)
