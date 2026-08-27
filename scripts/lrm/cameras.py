"""Kamera geometrisi: intrinsic olcekleme, dunya-uzayi isinlar, Plucker haritasi.
Konvansiyon: OpenGL-stili kamera (x sag, y yukari, -z ileri). c2w = camera->world."""
import torch


def scale_intrinsics(K, src_res, dst_res):
    s = dst_res / src_res
    Ks = K.clone()
    Ks[0, 0] *= s  # fx
    Ks[1, 1] *= s  # fy
    Ks[0, 2] *= s  # cx
    Ks[1, 2] *= s  # cy
    return Ks


def rays_from_camera(c2w, K, H, W):
    device = c2w.device
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing="xy",
    )
    i = i + 0.5
    j = j + 0.5
    # kamera uzayinda yon: x sag, y yukari, -z ileri
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1)
    dirs_w = dirs @ c2w[:3, :3].T
    dirs_w = dirs_w / dirs_w.norm(dim=-1, keepdim=True)
    origins = c2w[:3, 3].expand_as(dirs_w)
    return origins.reshape(-1, 3), dirs_w.reshape(-1, 3)


def plucker_map(c2w, K, H, W):
    o, d = rays_from_camera(c2w, K, H, W)
    m = torch.cross(o, d, dim=-1)
    pl = torch.cat([d, m], dim=-1)
    return pl.reshape(H, W, 6)


def canonicalize(ref_c2w, c2w_stack):
    """Kamera normalizasyonu (LRM): dunyayi, referans (giris) kamerasi kanonik
    poza (+Z ekseninde, orijine bakan, identity rotasyon) gelecek sekilde dondurur;
    AYNI donusumu tum kameralara uygular => obje her ornekte ayni yonelimden gorunur,
    optimizasyon uzayi daralir (LRM'de PSNR 15.3->19.0). Saf rotasyon oldugu icin obje
    orijinde kalir. c2w_stack: (V,4,4). Donus: (V,4,4)."""
    # ORTONORMALLESTIRME SART (2026-08-27, bagimsiz denetim):
    # meta.json'daki `extrinsic` Blender'in cam.matrix_world.inverted()'i ve
    # sahne normalizasyonundan gelen OBJE-BASINA UNIFORM OLCEK tasiyor.
    # Olculdu (12/12 obje): olcek 0.31 - 893 arasi, det(R_c2w) 0.0000 - 34.7.
    # Isin yonleri normalize edildigi icin render/ogretmen ETKILENMIYOR, ama
    # burada R^T dogrudan OTELEMEYE de uygulaniyordu => kameralar orijine dogru
    # cekiliyor, `near` duzleminin icine giriyor, sahne bosaliyordu.
    _R = ref_c2w[:3, :3]
    _s = _R.norm(dim=0, keepdim=True).clamp_min(1e-12)   # uniform olcek (sutun boyu)
    R = (_R / _s).transpose(-1, -2)                # kanonik = identity => R = R_ref^T
    # Yigindaki rotasyonlar da ayni olcegi tasiyor; cikti ORTONORMAL olmali
    # cunku tuketicisi (LRM.camera_feature) ham matrisi modLN kosulu yapiyor.
    rot = c2w_stack[:, :3, :3]
    rot = rot / rot.norm(dim=1, keepdim=True).clamp_min(1e-12)
    trn = c2w_stack[:, :3, 3]
    out = c2w_stack.clone()
    out[:, :3, :3] = torch.einsum("ij,vjk->vik", R, rot)
    out[:, :3, 3] = torch.einsum("ij,vj->vi", R, trn)
    return out
