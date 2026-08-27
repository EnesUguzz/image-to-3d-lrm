import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm.nerf import TriplaneNeRF
from lrm.model import LRM


class FakeEncoder(torch.nn.Module):
    embed_dim = 384
    patch = 14

    def forward(self, imgs):
        return torch.randn(imgs.shape[0], 256, 384)


def test_density_bias_baslangicta_SIFIR_OLMAYAN_yogunluk_verir():
    """acc->0 sogurucu durumu: yogunluk sifira oturunca softplus gradyani oluyor
    ve model bir daha cikamiyor (M_base, M_enc4, K2_bf16 ayni sekilde coktu).
    density_bias 'sisli' bir baslangic verip gradyani canli tutar."""
    torch.manual_seed(0)
    feats = torch.randn(512, 96)
    d0, _ = TriplaneNeRF(96, density_bias=0.0)(feats)
    d1, _ = TriplaneNeRF(96, density_bias=1.0)(feats)
    assert d1.mean() > d0.mean()
    assert d1.min() > 0.5, "bias'li baslangic her noktada sifirdan uzak olmali"


def test_noise_std_sadece_egitimde_etkili():
    torch.manual_seed(0)
    feats = torch.randn(256, 96)
    net = TriplaneNeRF(96, noise_std=0.5)
    net.train()
    a, _ = net(feats)
    b, _ = net(feats)
    assert not torch.allclose(a, b), "egitimde gurultu uygulanmali"
    net.eval()
    c, _ = net(feats)
    d, _ = net(feats)
    assert torch.allclose(c, d), "eval'de gurultu OLMAMALI"


def test_LRM_bu_cengelleri_NeRF_e_gecirir():
    """K4: cengeller yazilmisti ama LRM.__init__ hic gecirmiyordu (daima 0.0)."""
    m = LRM(dim=32, depth=1, heads=4, triplane_res=8, triplane_ch=8, nerf_hidden=16,
            encoder=FakeEncoder(), density_bias=1.25, noise_std=0.3)
    assert m.nerf.density_bias == 1.25
    assert m.nerf.noise_std == 0.3


def test_varsayilan_davranis_DEGISMEZ():
    m = LRM(dim=32, depth=1, heads=4, triplane_res=8, triplane_ch=8, nerf_hidden=16,
            encoder=FakeEncoder())
    assert m.nerf.density_bias == 0.0 and m.nerf.noise_std == 0.0
