import torch
import torch.nn as nn

from lrm.model import LRM
from train_lrm import save_checkpoint, load_checkpoint


class FakeEncoder(nn.Module):
    embed_dim = 384
    patch = 14

    def forward(self, imgs):
        return torch.randn(imgs.shape[0], 256, 384)


def _tiny_model():
    return LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
              nerf_hidden=16, encoder=FakeEncoder())


def test_checkpoint_roundtrip(tmp_path):
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    p = str(tmp_path / "ck.pt")
    save_checkpoint(p, m, opt, sched, step=123)

    m2 = _tiny_model()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.ConstantLR(opt2)
    step, teacher, opt_t = load_checkpoint(p, m2, opt2, sched2)
    assert step == 123
    assert teacher is None and opt_t is None   # ogretmensiz checkpoint
    # 2026-08-29: TriplaneHead.proj (Linear darbogazi) kaldirildi, agirlik
    # artik dogrudan deconv uzerinde. bkz. tests/test_triplane_rank.py
    assert torch.allclose(m.triplane_head.up.weight, m2.triplane_head.up.weight)


def test_checkpoint_ogretmeni_de_saklar(tmp_path):
    """Ogretmen triplane resume'da sifirdan baslamamali: yeniden isinma yerel
    minimuma geri dusmeye yol acabilir (bkz. docs/lrm-blob-diagnosis.md)."""
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    teacher = nn.Parameter(torch.randn(4, 3, 8, 16, 16))
    opt_t = torch.optim.Adam([teacher], lr=1e-2)
    teacher.sum().backward()
    opt_t.step()
    p = str(tmp_path / "ck_ogr.pt")
    save_checkpoint(p, m, opt, sched, step=7, teacher=teacher, opt_t=opt_t)

    m2 = _tiny_model()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.ConstantLR(opt2)
    step, t2, ot2 = load_checkpoint(p, m2, opt2, sched2)
    assert step == 7
    assert t2 is not None and t2.shape == teacher.shape
    assert torch.allclose(t2, teacher.detach().cpu())
    assert ot2 is not None


def test_kisa_kosu_uzun_kosunun_checkpointini_EZEMEZ(tmp_path):
    """GERCEKTEN OLDU (2026-08-26): 40 adimlik bir duman testi, kosu sonunda
    kosulsuz save_checkpoint cagirdigi icin 16.000 adimlik gece kosusunun
    last.pt'sini ezdi. Artik daha KUCUK adimli bir kayit, daha buyuk adimli
    mevcut bir kaydin uzerine force olmadan yazamaz."""
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    p = str(tmp_path / "last.pt")

    save_checkpoint(p, m, opt, sched, step=16000)
    written = save_checkpoint(p, m, opt, sched, step=40)      # duman testi
    assert written is False
    assert torch.load(p, map_location="cpu", weights_only=False)["step"] == 16000


def test_ilerleyen_kosu_kendi_checkpointini_yazabilir(tmp_path):
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    p = str(tmp_path / "last.pt")
    save_checkpoint(p, m, opt, sched, step=1000)
    assert save_checkpoint(p, m, opt, sched, step=2000) is True
    assert torch.load(p, map_location="cpu", weights_only=False)["step"] == 2000


def test_force_ile_bilerek_ezilebilir(tmp_path):
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    p = str(tmp_path / "last.pt")
    save_checkpoint(p, m, opt, sched, step=16000)
    assert save_checkpoint(p, m, opt, sched, step=40, force=True) is True
    assert torch.load(p, map_location="cpu", weights_only=False)["step"] == 40
