"""Uctan uca LRM: girdi foto(lar) + poz -> triplane -> supervision render."""
import torch
import torch.nn as nn

from lrm import cameras
from lrm import defaults
from lrm.transformer import LRMTransformer
from lrm.triplane import TriplaneHead, sample_triplane, tv_loss as _tv_loss
from lrm.nerf import TriplaneNeRF
from lrm.renderer import volume_render
from lrm.encoder import DinoEncoder


class LRM(nn.Module):
    INPUT_RES = 224  # encoder giris cozunurlugu (intrinsic bu olcekte gelir)

    def __init__(self, dim=512, depth=12, heads=8, triplane_res=32,
                 triplane_ch=32, nerf_hidden=64, encoder=None,
                 bound=defaults.BOUND, near=defaults.NEAR, far=defaults.FAR,
                 n_samples=defaults.N_SAMPLES, white_bg=True,
                 cross_attn=False, density_bias=0.0, noise_std=0.0):
        super().__init__()
        self.encoder = encoder if encoder is not None else DinoEncoder()
        self.cross_attn = cross_attn
        self.transformer = LRMTransformer(dim=dim, depth=depth, heads=heads,
                                          triplane_res=triplane_res,
                                          img_dim=self.encoder.embed_dim,
                                          cross_attn=cross_attn)
        self.triplane_head = TriplaneHead(dim=dim, out_channels=triplane_ch)
        # ANTI-COKUS: yogunluk sifira oturunca softplus gradyani olur ve model
        # 'bos sahne' sogurucu durumundan bir daha cikamaz -- M_base, M_enc4 ve
        # K2_bf16 tam olarak boyle coktu (acc=0.0000). density_bias sisli bir
        # baslangic verir, noise_std sert doygunlugu engeller (NeRF raw_noise_std).
        # Cengeller nerf.py'de vardi ama buraya HIC baglanmamisti (K4).
        self.nerf = TriplaneNeRF(in_dim=3 * triplane_ch, hidden=nerf_hidden,
                                 density_bias=density_bias, noise_std=noise_std)
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
        cam_feat = self.camera_feature(input_c2w, input_K) if self.cross_attn else None
        tp_grid = self.transformer(img_tokens, plucker, cam_feat)   # (3,r,r,dim)
        triplane = self.triplane_head(tp_grid)            # (3,C,H,W)
        self._last_triplane = triplane                    # TV reg icin (egitim dongusu okur)
        return triplane

    @classmethod
    def camera_feature(cls, c2w, K):
        """Giris kamerasi -> modLN kosul vektoru (LRM): c2w'nin ilk 3 satiri (12)
        + normalize intrinsic (fx,fy,cx,cy / res) = 16 boyut, gorunum basina."""
        ext = c2w[:, :3, :4].reshape(c2w.shape[0], 12)
        r = float(cls.INPUT_RES)
        intr = torch.stack([K[:, 0, 0] / r, K[:, 1, 1] / r,
                            K[:, 0, 2] / r, K[:, 1, 2] / r], dim=-1)
        return torch.cat([ext, intr], dim=-1)  # (Vi,16)

    @staticmethod
    def tv_loss(triplane):
        """OpenLRM recetesi, TV 5e-4. Gercek uygulama lrm/triplane.py'de (ortak)."""
        return _tv_loss(triplane)

    def render_view(self, triplane, c2w, K, H, W, bg_color=None):
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
                                 white_bg=self.white_bg, bg_color=bg_color)
        rgb = rgb.reshape(H, W, 3).permute(2, 0, 1)
        acc = acc.reshape(H, W, 1).permute(2, 0, 1)
        return rgb, acc

    def forward(self, input_imgs, input_c2w, input_K,
                render_c2w, render_K, render_hw, bg_color=None):
        H, W = render_hw
        triplane = self.make_triplane(input_imgs, input_c2w, input_K)
        V = render_c2w.shape[0]
        # TUM supervision gorunumlerinin isinlarini TEK volume_render cagrisinda batch'le
        # (gorunumler bagimsiz => matematik birebir ayni; daha az kernel launch, GPU daha dolu)
        os_, ds_ = [], []
        for i in range(V):
            o, d = cameras.rays_from_camera(render_c2w[i], render_K[i], H, W)
            os_.append(o)
            ds_.append(d)
        o = torch.cat(os_, 0).to(triplane.device)
        d = torch.cat(ds_, 0).to(triplane.device)

        def query(pts):
            feats = sample_triplane(triplane, pts, bound=self.bound)
            density, rgb = self.nerf(feats)
            inside = (pts.abs().amax(dim=-1, keepdim=True) <= self.bound).to(density.dtype)
            return density * inside, rgb

        rgb, acc = volume_render(o, d, self.near, self.far, self.n_samples, query,
                                 white_bg=self.white_bg, bg_color=bg_color)
        rgb = rgb.reshape(V, H, W, 3).permute(0, 3, 1, 2)
        acc = acc.reshape(V, H, W, 1).permute(0, 3, 1, 2)
        return rgb, acc
