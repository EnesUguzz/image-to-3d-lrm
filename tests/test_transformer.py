import torch

from lrm.transformer import LRMTransformer


def test_transformer_output_shape():
    net = LRMTransformer(dim=64, depth=2, heads=4, triplane_res=8, img_dim=384)
    img_tokens = torch.randn(256, 384)   # ornegin 1 gorunum 16x16
    plucker = torch.randn(256, 6)
    out = net(img_tokens, plucker)
    assert out.shape == (3, 8, 8, 64)


def test_transformer_handles_multiview_token_count():
    net = LRMTransformer(dim=64, depth=2, heads=4, triplane_res=8, img_dim=384)
    # 3 gorunum => 3*256 token
    out = net(torch.randn(768, 384), torch.randn(768, 6))
    assert out.shape == (3, 8, 8, 64)


def test_cross_attn_govdesi_sekil_ve_kamera_kosulu():
    """cross_attn govdesi: triplane token'lari image'a cross-attention yapar,
    kamera ozelligi modLN ile enjekte edilir."""
    import torch
    from lrm.transformer import LRMTransformer
    t = LRMTransformer(dim=32, depth=2, heads=4, triplane_res=4, img_dim=16,
                       cross_attn=True)
    img = torch.randn(12, 16)
    pl = torch.randn(12, 6)
    cam = torch.randn(2, 16)
    out = t(img, pl, cam)
    assert out.shape == (3, 4, 4, 32)


def test_cross_attn_blogu_goruntuyu_okumak_ZORUNDA():
    """Kok-neden regresyonu: birlesik self-attention'da triplane token'lari
    goruntuyu yok sayabiliyordu (olculdu: attention kutlesi ~0 -> ortalama obje).
    CondBlock'ta cross-attention tek giris yolu; goruntu degisince cikti DEGISMELI."""
    import torch
    from lrm.transformer import CondBlock
    torch.manual_seed(0)
    blk = CondBlock(dim=32, heads=4, cond_dim=32).eval()
    x = torch.randn(1, 5, 32)
    cond = torch.randn(1, 32)
    o1 = blk(x, torch.randn(1, 7, 32), cond)
    o2 = blk(x, torch.randn(1, 7, 32), cond)
    assert (o1 - o2).abs().mean() > 1e-4


def test_kamera_ozelligi_16_boyut():
    import torch
    from lrm.model import LRM
    c2w = torch.eye(4)[None].repeat(3, 1, 1)
    K = torch.tensor([[[100., 0, 112.], [0, 100., 112.], [0, 0, 1.]]]).repeat(3, 1, 1)
    f = LRM.camera_feature(c2w, K)
    assert f.shape == (3, 16)
    assert torch.allclose(f[0, 12:], torch.tensor([100 / 224, 100 / 224, 0.5, 0.5]))
