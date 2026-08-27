import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm import cameras, crop


def _K(res, f=None):
    f = f if f is not None else res * 1.09375        # lens 35 / sensor 32
    return torch.tensor([[f, 0.0, res / 2],
                         [0.0, f, res / 2],
                         [0.0, 0.0, 1.0]])


def _c2w():
    m = torch.eye(4)
    m[2, 3] = 1.4866
    return m


def test_kirpma_intrinsigi_sadece_merkezi_kaydirir():
    K = _K(192)
    Kc = crop.crop_K(K, ax=40, ay=17)
    assert Kc[0, 0] == K[0, 0] and Kc[1, 1] == K[1, 1]     # odak degismez
    assert Kc[0, 2] == pytest.approx(float(K[0, 2]) - 40)
    assert Kc[1, 2] == pytest.approx(float(K[1, 2]) - 17)


def test_ISIN_DENKLIGI_kirpilan_piksel_ayni_isini_verir():
    """KRITIK: kirpilmis goruntunun (i,j) pikselinden cikan isin, tam karedeki
    (i+ax, j+ay) pikselinden cikan isinla BIREBIR ayni olmali. Intrinsic kaydirmasi
    yanlissa her sey sessizce bozulur -- bu projede tam olarak bu sinifta hatalar
    yasandi (bkz. PROJE-DEVIR-BELGESI 7.2)."""
    R, region, ax, ay = 192, 64, 40, 17
    c2w = _c2w()
    K_full = _K(R)
    o_f, d_f = cameras.rays_from_camera(c2w, K_full, R, R)
    o_c, d_c = cameras.rays_from_camera(c2w, crop.crop_K(K_full, ax, ay), region, region)

    d_f = d_f.reshape(R, R, 3)[ay:ay + region, ax:ax + region].reshape(-1, 3)
    o_f = o_f.reshape(R, R, 3)[ay:ay + region, ax:ax + region].reshape(-1, 3)
    assert torch.allclose(d_c, d_f, atol=1e-6)
    assert torch.allclose(o_c, o_f, atol=1e-6)


def test_master_olcekleme_ve_kirpma_zinciri_dogru():
    """512 master -> r cozunurluk -> region kirpma; tek adimda ayni sonuc."""
    master, r, region, ax, ay = 512, 192, 64, 40, 17
    K_master = _K(master)
    K_r = cameras.scale_intrinsics(K_master, master, r)
    step_by_step = crop.crop_K(K_r, ax, ay)
    one_shot = crop.scale_and_crop_K(K_master, master, r, ax, ay)
    assert torch.allclose(step_by_step, one_shot)


def test_capa_cerceve_disina_tasmaz():
    rng = random.Random(0)
    for _ in range(200):
        r = rng.choice([64, 96, 192])
        ax, ay = crop.sample_anchor(None, r, region=64, rng=rng, fg_bias=0.0)
        assert 0 <= ax <= r - 64 and 0 <= ay <= r - 64


def test_region_r_ye_esitse_capa_sifir():
    rng = random.Random(0)
    assert crop.sample_anchor(None, 64, region=64, rng=rng, fg_bias=1.0) == (0, 0)


def test_fg_bias_1_ise_kirpma_DAIMA_on_plan_icerir():
    """Objelerimiz kareyi ort %9 kapliyor; rastgele kirpma cogu zaman bos
    arka plan getirir. fg_bias=1.0 kirpmanin alpha bbox'i ile kesismesini garanti eder."""
    r, region = 192, 64
    alpha = torch.zeros(1, r, r)
    alpha[0, 20:40, 150:175] = 1.0                  # kose-ust tarafta kucuk obje
    rng = random.Random(0)
    for _ in range(100):
        ax, ay = crop.sample_anchor(alpha, r, region, rng, fg_bias=1.0)
        patch = alpha[0, ay:ay + region, ax:ax + region]
        assert patch.sum() > 0, f"bos kirpma: ax={ax} ay={ay}"


def test_fg_bias_bos_alphada_patlamaz():
    """Tamamen bos bir hedef (dejenere obje) fg_bias'i kilitlememeli."""
    r, region = 128, 64
    rng = random.Random(0)
    ax, ay = crop.sample_anchor(torch.zeros(1, r, r), r, region, rng, fg_bias=1.0)
    assert 0 <= ax <= r - region and 0 <= ay <= r - region


def test_kirpma_piksel_icerigi_kaynakla_ayni():
    r, region, ax, ay = 128, 64, 33, 7
    img = torch.rand(3, r, r)
    out = crop.crop_image(img, ax, ay, region)
    assert out.shape == (3, region, region)
    assert torch.equal(out, img[:, ay:ay + region, ax:ax + region])


def test_cok_gorunumlu_tensor_de_kirpilir():
    V, r, region, ax, ay = 3, 128, 64, 10, 20
    img = torch.rand(V, 1, r, r)
    out = crop.crop_image(img, ax, ay, region)
    assert out.shape == (V, 1, region, region)
    assert torch.equal(out, img[..., ay:ay + region, ax:ax + region])


# ---------------------------------------------------------------- uid kumesi secimi
def _uids_for(argv_n_obj, uids_file_len=250):
    """bench_overfit'in uid secim mantigini birebir taklit eder."""
    import importlib
    bo = importlib.import_module("bench_overfit")
    return bo.select_uids(uids_file_uids=[f"u{i}" for i in range(uids_file_len)],
                          n_obj=argv_n_obj, uids_csv="", all_uids=[])


def test_uids_file_verildiginde_n_obj_VARSAYILANI_listeyi_KIRPMAZ():
    """GERCEKTEN OLDU (2026-08-26): --uids_file bench_uids_250.json verildi ama
    --n_obj'nin 32 olan VARSAYILANI listeyi sessizce 32'ye kirpti; 250-obje
    kapisi 32 objeyle kosmus gibi oldu. n_obj ancak ACIKCA verilirse kirpmali."""
    assert len(_uids_for(None)) == 250


def test_n_obj_acikca_verilirse_kirpar():
    assert len(_uids_for(40)) == 40


def test_n_obj_listeden_buyukse_hepsi_kalir():
    assert len(_uids_for(1000)) == 250
