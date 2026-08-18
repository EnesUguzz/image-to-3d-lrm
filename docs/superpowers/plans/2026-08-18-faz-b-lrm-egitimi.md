# Faz B — LRM Eğitimi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 1-4 kanonik fotodan triplane-NeRF üreten, 16GB VRAM'e sığan multi-view LRM'i eğitilebilir hale getirmek.

**Architecture:** Donuk DINOv2 ViT-S/14 encoder → (image token'ları + Plücker) ve öğrenilebilir triplane token'ları tek self-attention transformer'da birleşir → triplane (3×64×64×32) → NeRF MLP → volume rendering ile supervision açıları render edilir → L2+LPIPS+mask loss. Model tek obje işler; eğitim döngüsü mikro-batch objelerini gradient accumulation ile toplar.

**Tech Stack:** PyTorch (CUDA 12.8, sm_120), torchvision, torch.hub DINOv2, lpips, numpy, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-faz-b-lrm-egitimi-design.md`

## Global Constraints

- **PyTorch cu128 şart:** RTX 5080 Blackwell (sm_120); eski wheel'ler çalışmaz. Kurulum doğrulanmadan koda geçme.
- **Dosya IO utf-8:** proje yolu `Ğ` içerir → tüm `open()` çağrılarında `encoding="utf-8"`.
- **Donuk encoder:** DINOv2 hiçbir zaman eğitilmez (`requires_grad=False`, `eval()`).
- **Koordinat konvansiyonu:** obje merkezli, yarıçap 0.5 (Faz A `TARGET_RADIUS`); triplane sınırı `bound=0.6`; kamera OpenGL-stili (x sağ, y yukarı, **-z ileri**); `c2w = inverse(meta.json extrinsic)` (meta world→camera saklar).
- **Çözünürlük:** encoder girdi 224, render hedef 128 (config'te değiştirilebilir); intrinsic 512'den ölçeklenir.
- **Precision:** eğitimde bf16 autocast + transformer'da gradient checkpointing.
- **Modül konumu:** `scripts/lrm/` paketi; testler `tests/` (mevcut conftest scripts/'i path'e ekliyor). Her yeni modül `scripts/lrm/__init__.py` ile paket içinde.
- **Tensör konvansiyonu:** görüntüler `(V,3,H,W)` float [0,1]; alpha `(V,1,H,W)`; pozlar `(V,4,4)` c2w; intrinsic `(V,3,3)`.

---

## Dosya Yapısı

```
scripts/lrm/
  __init__.py
  cameras.py       # intrinsic ölçekleme, ışın üretimi, Plücker haritası
  augment.py       # girdi-only augmentation
  dataset.py       # LRMDataset (girdi/supervision seçimi, resize, maske, poz)
  encoder.py       # donuk DINOv2 sarmalayıcı (+imagenet normalize)
  transformer.py   # image+triplane token self-attention gövdesi
  triplane.py      # TriplaneHead (token→düzlem, upsample) + sample_triplane
  nerf.py          # TriplaneNeRF MLP (feature → density+rgb)
  renderer.py      # volume_render (ray sample + integrate)
  model.py         # LRM: uçtan uca forward
  losses.py        # mse + lpips + mask
scripts/
  train_lrm.py     # eğitim döngüsü + checkpoint/resume + val preview + CLI
  overfit_lrm.py   # 1-2 obje overfit sağlık testi
tests/
  test_cameras.py test_augment.py test_dataset.py test_triplane.py
  test_renderer.py test_nerf.py test_losses.py test_transformer.py test_model_shapes.py
```

---

### Task 1: Ortam kurulumu (PyTorch cu128 + GPU doğrulama)

TDD değil; deliverable = GPU'da çalışan torch. Kod yazmadan önce şart.

**Files:**
- Create: `scripts/lrm/__init__.py` (boş)
- Create: `scripts/check_env.py`

**Interfaces:**
- Produces: doğrulanmış `torch` (cu128), `torch.cuda.is_available()==True`, sm_120 matmul çalışır.

- [ ] **Step 1: PyTorch cu128 kur**

PowerShell:
```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install lpips numpy Pillow tqdm pytest
```
(cu128 wheel yoksa PyTorch'un güncel "get started" sayfasındaki Blackwell/sm_120 destekli index'i kullan; nightly gerekebilir.)

- [ ] **Step 2: `scripts/check_env.py` yaz**

```python
"""GPU/torch ortam doğrulama. RTX 5080 sm_120 gercekten calisiyor mu."""
import torch

def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA yok - cu128 kurulumunu kontrol et"
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("gpu:", name, "capability:", cap)
    # sm_120 (Blackwell) gercek matmul: eski wheel burada patlar
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    c = (a @ b).sum().item()
    print("gpu matmul ok, sum=", c)
    # bf16 destegi
    x = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
    (x @ x).sum().item()
    print("bf16 ok")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Çalıştır ve doğrula**

Run: `python scripts/check_env.py`
Expected: `cuda available: True`, gpu adı RTX 5080, capability `(12, 0)`, "gpu matmul ok", "bf16 ok". Hata alırsan (özellikle `CUDA error: no kernel image is available for execution` → yanlış wheel) kurulumu düzelt, koda GEÇME.

- [ ] **Step 4: Commit**

```bash
git add scripts/lrm/__init__.py scripts/check_env.py
git commit -m "Faz B: ortam kurulumu + GPU dogrulama (check_env)"
```

---

### Task 2: cameras.py — ışın üretimi + Plücker

**Files:**
- Create: `scripts/lrm/cameras.py`
- Test: `tests/test_cameras.py`

**Interfaces:**
- Produces:
  - `scale_intrinsics(K: Tensor(3,3), src_res: int, dst_res: int) -> Tensor(3,3)`
  - `rays_from_camera(c2w: Tensor(4,4), K: Tensor(3,3), H: int, W: int) -> (origins: Tensor(H*W,3), dirs: Tensor(H*W,3))` (dirs birim vektör, dünya uzayı)
  - `plucker_map(c2w: Tensor(4,4), K: Tensor(3,3), H: int, W: int) -> Tensor(H,W,6)` (kanal: [dir(3), moment(3)])

- [ ] **Step 1: Testi yaz**

```python
import torch
from lrm import cameras

def _identity_c2w(dist=1.5):
    c2w = torch.eye(4)
    c2w[2, 3] = dist  # kamera +z'de, -z'ye bakar
    return c2w

def _K(res=128, f=200.0):
    K = torch.tensor([[f, 0, res/2], [0, f, res/2], [0, 0, 1]], dtype=torch.float32)
    return K

def test_scale_intrinsics_halves():
    K = _K(res=512, f=800.0)
    Ks = cameras.scale_intrinsics(K, 512, 256)
    assert torch.allclose(Ks[0, 0], torch.tensor(400.0))
    assert torch.allclose(Ks[0, 2], torch.tensor(128.0))

def test_center_ray_points_forward():
    c2w = _identity_c2w(dist=1.5)
    K = _K(res=128)
    o, d = cameras.rays_from_camera(c2w, K, 128, 128)
    center = (64 * 128 + 64)  # yaklasik merkez piksel
    # merkez isin dunya -z yonunde olmali (kamera +z'den -z'ye bakar)
    assert torch.allclose(d[center], torch.tensor([0.0, 0.0, -1.0]), atol=1e-2)
    assert torch.allclose(o[center], torch.tensor([0.0, 0.0, 1.5]), atol=1e-5)

def test_plucker_shape_and_center_moment_zero():
    c2w = _identity_c2w(dist=1.5)
    K = _K(res=16)
    pl = cameras.plucker_map(c2w, K, 16, 16)
    assert pl.shape == (16, 16, 6)
    # merkeze yakin isin origin ve dir kolinear degil ama moment = o x d
    # tam merkez pikselde d=-z, o=+1.5z => o x d = 0
    m = pl[8, 8, 3:]
    assert m.norm() < 0.2
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_cameras.py -v`
Expected: FAIL (`ModuleNotFoundError: lrm.cameras`).

- [ ] **Step 3: Implementasyonu yaz**

```python
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
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_cameras.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/cameras.py tests/test_cameras.py
git commit -m "Faz B: cameras - isin uretimi + Plucker haritasi"
```

---

### Task 3: augment.py — girdi-only augmentation

**Files:**
- Create: `scripts/lrm/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: girdi RGBA tensör `(4,H,W)` [0,1] (alpha dahil).
- Produces:
  - `composite_background(rgba: Tensor(4,H,W), rng) -> Tensor(3,H,W)` (şeffafı rastgele arka plana bindirir)
  - `augment_input(rgba: Tensor(4,H,W), rng: random.Random) -> Tensor(3,H,W)` (bg + blur + jitter + jpeg benzeri). Deterministik (rng verilince tekrarlanabilir). Sadece girdiye uygulanır.

- [ ] **Step 1: Testi yaz**

```python
import random
import torch
from lrm import augment

def _rgba(H=32, W=32, alpha_val=1.0):
    x = torch.zeros(4, H, W)
    x[:3, 8:24, 8:24] = 0.7      # ortada obje
    x[3, 8:24, 8:24] = alpha_val  # alpha
    return x

def test_composite_fills_transparent_background():
    rgba = _rgba()
    rng = random.Random(0)
    out = augment.composite_background(rgba, rng)
    assert out.shape == (3, 32, 32)
    # seffaf kose artik bos (0) degil - bir arka plan rengiyle dolu
    corner = out[:, 0, 0]
    assert corner.abs().sum() > 0.0

def test_augment_output_shape_and_range():
    rng = random.Random(1)
    out = augment.augment_input(_rgba(), rng)
    assert out.shape == (3, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_augment_is_deterministic_with_seed():
    a = augment.augment_input(_rgba(), random.Random(5))
    b = augment.augment_input(_rgba(), random.Random(5))
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_augment.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementasyonu yaz**

```python
"""Girdi-only augmentation: kullanici fotosu robustlugu.
SADECE girdiye uygulanir; supervision hedefleri temiz kalir."""
import torch
import torch.nn.functional as F


def composite_background(rgba, rng):
    rgb, alpha = rgba[:3], rgba[3:4]
    kind = rng.random()
    _, H, W = rgb.shape
    if kind < 0.6:  # duz renk
        color = torch.tensor([rng.random() for _ in range(3)]).view(3, 1, 1)
        bg = color.expand(3, H, W)
    else:  # dikey gradyan
        top = torch.tensor([rng.random() for _ in range(3)]).view(3, 1, 1)
        bot = torch.tensor([rng.random() for _ in range(3)]).view(3, 1, 1)
        t = torch.linspace(0, 1, H).view(1, H, 1)
        bg = top * (1 - t) + bot * t
        bg = bg.expand(3, H, W)
    return rgb * alpha + bg * (1 - alpha)


def _gaussian_blur(img, rng):
    if rng.random() < 0.5:
        return img
    k = rng.choice([3, 5])
    sigma = rng.uniform(0.4, 1.2)
    ax = torch.arange(k) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :]).view(1, 1, k, k).expand(3, 1, k, k)
    img = F.pad(img[None], (k // 2,) * 4, mode="reflect")
    return F.conv2d(img, kernel, groups=3)[0]


def _color_jitter(img, rng):
    b = rng.uniform(0.8, 1.2)   # parlaklik
    c = rng.uniform(0.8, 1.2)   # kontrast
    img = img * b
    mean = img.mean()
    img = (img - mean) * c + mean
    return img.clamp(0, 1)


def _jpeg_like(img, rng):
    # gercek jpeg yerine hafif kuantalama artefakti taklidi
    if rng.random() < 0.5:
        return img
    levels = rng.choice([16, 24, 32])
    return (img * levels).round() / levels


def augment_input(rgba, rng):
    img = composite_background(rgba, rng)
    img = _gaussian_blur(img, rng)
    img = _color_jitter(img, rng)
    img = _jpeg_like(img, rng)
    return img.clamp(0, 1)
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_augment.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/augment.py tests/test_augment.py
git commit -m "Faz B: augment - girdi-only bg/blur/jitter/jpeg"
```

---

### Task 4: dataset.py — LRMDataset

**Files:**
- Create: `scripts/lrm/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `lrm.cameras.scale_intrinsics`, `lrm.augment.augment_input`.
- Produces: `LRMDataset(train_list_path, renders_dir, split="train", input_res=224, render_res=128, n_sup=4, augment=True, seed=0)`. `__getitem__` bir dict döner (tek obje):
  - `input_imgs: (k,3,224,224)` [0,1], k∈{1,2,3,4}
  - `input_c2w: (k,4,4)`, `input_K: (k,3,3)`
  - `sup_rgb: (n_sup,3,128,128)`, `sup_alpha: (n_sup,1,128,128)`
  - `sup_c2w: (n_sup,4,4)`, `sup_K: (n_sup,3,3)`
  - `uid: str`
- `lrm_collate(batch) -> list[dict]` (değişken k → liste; model tek obje işler).

Not: `meta.json` extrinsic world→camera saklar; `c2w = inverse(extrinsic)`. Intrinsic 512 içindir → hedef çözünürlüğe ölçeklenir. Kanonik görünüm index'leri meta'dan (`canonical_indices`) ya da 000-003 varsayılır.

- [ ] **Step 1: Testi yaz (sentetik mini dataset üret)**

```python
import json, os
import numpy as np
import torch
from PIL import Image
from lrm.dataset import LRMDataset, lrm_collate

def _make_obj(root, uid, n_views=16, res=64):
    d = os.path.join(root, uid); os.makedirs(d, exist_ok=True)
    metas = {"intrinsic": [[400, 0, 256], [0, 400, 256], [0, 0, 1]],
             "canonical_indices": [0, 1, 2, 3], "views": []}
    for i in range(n_views):
        arr = np.zeros((res, res, 4), dtype=np.uint8)
        arr[16:48, 16:48, :3] = 180
        arr[16:48, 16:48, 3] = 255
        Image.fromarray(arr, "RGBA").save(os.path.join(d, f"{i:03d}.png"))
        c2w = np.eye(4); c2w[2, 3] = 1.5  # extrinsic = world->camera
        metas["views"].append({"extrinsic": c2w.tolist()})
    json.dump(metas, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"))

def _setup(tmp_path):
    rroot = tmp_path / "renders"; rroot.mkdir()
    uids = [f"u{i}" for i in range(3)]
    for u in uids:
        _make_obj(str(rroot), u)
    tl = tmp_path / "train_list.json"
    json.dump({"train": uids, "val": []}, open(tl, "w", encoding="utf-8"))
    return str(tl), str(rroot)

def test_item_shapes_and_input_count_in_range(tmp_path):
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", input_res=224, render_res=128, n_sup=4)
    item = ds[0]
    k = item["input_imgs"].shape[0]
    assert 1 <= k <= 4
    assert item["input_imgs"].shape[1:] == (3, 224, 224)
    assert item["input_c2w"].shape == (k, 4, 4)
    assert item["sup_rgb"].shape == (4, 3, 128, 128)
    assert item["sup_alpha"].shape == (4, 1, 128, 128)

def test_supervision_disjoint_from_input(tmp_path):
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", n_sup=4, seed=0)
    it = ds[0]
    # girdi ve supervision indexleri cakismamali
    assert set(it["input_view_idx"]).isdisjoint(set(it["sup_view_idx"]))

def test_collate_returns_list(tmp_path):
    tl, rroot = _setup(tmp_path)
    ds = LRMDataset(tl, rroot, split="train")
    batch = lrm_collate([ds[0], ds[1]])
    assert isinstance(batch, list) and len(batch) == 2
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementasyonu yaz**

```python
"""LRMDataset: render + poz okur; her ornekte 1-4 girdi + n_sup supervision secer.
Girdiye augmentation uygulanir, supervision temiz kalir."""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lrm import cameras
from lrm.augment import augment_input

RENDER_MASTER_RES = 512  # meta intrinsic bu cozunurluge gore


def _load_rgba(path, res):
    im = Image.open(path).convert("RGBA")
    arr = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0  # (4,H,W)
    if arr.shape[-1] != res:
        arr = F.interpolate(arr[None], size=(res, res), mode="bilinear",
                            align_corners=False)[0]
    return arr


class LRMDataset(torch.utils.data.Dataset):
    def __init__(self, train_list_path, renders_dir, split="train",
                 input_res=224, render_res=128, n_sup=4, augment=True, seed=0):
        with open(train_list_path, encoding="utf-8") as f:
            self.uids = json.load(f)[split]
        self.renders_dir = renders_dir
        self.input_res = input_res
        self.render_res = render_res
        self.n_sup = n_sup
        self.augment = augment
        self.base_seed = seed

    def __len__(self):
        return len(self.uids)

    def _meta(self, uid):
        with open(os.path.join(self.renders_dir, uid, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        meta = self._meta(uid)
        rng = random.Random(self.base_seed * 1_000_003 + idx)
        canon = meta.get("canonical_indices", [0, 1, 2, 3])
        n_views = len(meta["views"])

        k = rng.randint(1, min(4, len(canon)))
        input_idx = rng.sample(canon, k)
        remaining = [i for i in range(n_views) if i not in input_idx]
        sup_idx = rng.sample(remaining, min(self.n_sup, len(remaining)))

        K512 = torch.tensor(meta["intrinsic"], dtype=torch.float32)
        Kin = cameras.scale_intrinsics(K512, RENDER_MASTER_RES, self.input_res)
        Ksup = cameras.scale_intrinsics(K512, RENDER_MASTER_RES, self.render_res)

        def c2w(i):
            ext = torch.tensor(meta["views"][i]["extrinsic"], dtype=torch.float32)
            return torch.linalg.inv(ext)

        input_imgs, input_c2w = [], []
        for n, i in enumerate(input_idx):
            rgba = _load_rgba(os.path.join(self.renders_dir, uid, f"{i:03d}.png"),
                              self.input_res)
            if self.augment:
                img = augment_input(rgba, random.Random(rng.random() * 1e9))
            else:
                img = rgba[:3] * rgba[3:4]  # temiz: siyah bg uzerine
            input_imgs.append(img)
            input_c2w.append(c2w(i))

        sup_rgb, sup_alpha, sup_c2w = [], [], []
        for i in sup_idx:
            rgba = _load_rgba(os.path.join(self.renders_dir, uid, f"{i:03d}.png"),
                              self.render_res)
            sup_rgb.append(rgba[:3] * rgba[3:4])
            sup_alpha.append(rgba[3:4])
            sup_c2w.append(c2w(i))

        return {
            "uid": uid,
            "input_imgs": torch.stack(input_imgs),
            "input_c2w": torch.stack(input_c2w),
            "input_K": Kin[None].expand(k, 3, 3).clone(),
            "input_view_idx": input_idx,
            "sup_rgb": torch.stack(sup_rgb),
            "sup_alpha": torch.stack(sup_alpha),
            "sup_c2w": torch.stack(sup_c2w),
            "sup_K": Ksup[None].expand(len(sup_idx), 3, 3).clone(),
            "sup_view_idx": sup_idx,
        }


def lrm_collate(batch):
    """Degisken girdi sayisi (1-4) => padding yerine liste dondur.
    Model tek obje isler; egitim dongusu listeyi gezer."""
    return list(batch)
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/dataset.py tests/test_dataset.py
git commit -m "Faz B: LRMDataset - girdi/supervision secimi + poz + augment"
```

---

### Task 5: encoder.py — donuk DINOv2

**Files:**
- Create: `scripts/lrm/encoder.py`
- Test: `tests/test_encoder.py`

**Interfaces:**
- Produces: `DinoEncoder(name="dinov2_vits14")` — `embed_dim=384`, `patch=14`. `forward(imgs: (V,3,224,224)) -> (V, 256, 384)` patch token'ları. İçinde imagenet normalize. Parametreler donuk. İlk çağrıda torch.hub ağırlık indirir (internet gerekir).

- [ ] **Step 1: Testi yaz** (internet/indirme gerektirir; ağır → `@pytest.mark.slow`)

```python
import pytest
import torch
from lrm.encoder import DinoEncoder

@pytest.mark.slow
def test_dino_patch_tokens_shape():
    enc = DinoEncoder()
    imgs = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        tok = enc(imgs)
    assert tok.shape == (2, 256, 384)

@pytest.mark.slow
def test_dino_is_frozen():
    enc = DinoEncoder()
    assert all(not p.requires_grad for p in enc.parameters())
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_encoder.py -v -m slow`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementasyonu yaz**

```python
"""Donuk DINOv2 ViT-S/14 encoder. Sadece ozellik cikarir, egitilmez."""
import torch
import torch.nn as nn

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoEncoder(nn.Module):
    def __init__(self, name="dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.embed_dim  # 384 (vits14)
        self.patch = 14
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        # donuk kalsin: BN/dropout yok ama yine de eval sabitle
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, imgs):
        x = (imgs - self.mean) / self.std
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]  # (V, N_patch, 384)
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_encoder.py -v -m slow`
Expected: PASS (indirme sonrası, 2 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/encoder.py tests/test_encoder.py
git commit -m "Faz B: DinoEncoder - donuk DINOv2 ViT-S/14 sarmalayici"
```

---

### Task 6: transformer.py — image+triplane self-attention gövdesi

**Files:**
- Create: `scripts/lrm/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Produces: `LRMTransformer(dim=512, depth=12, heads=8, triplane_res=32, img_dim=384)`.
  `forward(img_tokens: (M,384), img_plucker: (M,6)) -> (3, 32, 32, dim)` triplane token ızgarası. (M = tüm girdi görünümlerinin patch token'ları düzleştirilmiş.)
  `enable_checkpointing()` — bloklarda gradient checkpointing açar.

- [ ] **Step 1: Testi yaz** (CPU, küçük boyut)

```python
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
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_transformer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementasyonu yaz**

```python
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
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_transformer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/transformer.py tests/test_transformer.py
git commit -m "Faz B: LRMTransformer - image+triplane self-attention govdesi"
```

---

### Task 7: triplane.py — TriplaneHead + örnekleme

**Files:**
- Create: `scripts/lrm/triplane.py`
- Test: `tests/test_triplane.py`

**Interfaces:**
- Produces:
  - `TriplaneHead(dim, out_channels=32, upsample=2)` — `forward(tp_grid: (3,r,r,dim)) -> (3, out_channels, r*upsample, r*upsample)`.
  - `sample_triplane(triplane: (3,C,H,W), points: (N,3), bound=0.6) -> (N, 3*C)` (bilinear grid_sample, 3 düzlemden okuyup birleştirir).

- [ ] **Step 1: Testi yaz**

```python
import torch
from lrm.triplane import TriplaneHead, sample_triplane

def test_head_upsamples():
    head = TriplaneHead(dim=64, out_channels=16, upsample=2)
    out = head(torch.randn(3, 8, 8, 64))
    assert out.shape == (3, 16, 16, 16)

def test_sample_shape():
    tri = torch.randn(3, 16, 32, 32)
    pts = torch.randn(100, 3) * 0.3
    feats = sample_triplane(tri, pts, bound=0.6)
    assert feats.shape == (100, 48)  # 3*16

def test_sample_center_matches_grid_center():
    # sabit dolu triplane => her nokta ayni ozellik
    tri = torch.ones(3, 4, 8, 8)
    feats = sample_triplane(tri, torch.zeros(5, 3), bound=0.6)
    assert torch.allclose(feats, torch.ones(5, 12), atol=1e-5)
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_triplane.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementasyonu yaz**

```python
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
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_triplane.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/triplane.py tests/test_triplane.py
git commit -m "Faz B: TriplaneHead + sample_triplane"
```

---

### Task 8: nerf.py — TriplaneNeRF MLP

**Files:**
- Create: `scripts/lrm/nerf.py`
- Test: `tests/test_nerf.py`

**Interfaces:**
- Produces: `TriplaneNeRF(in_dim, hidden=64)` — `forward(feats: (N, in_dim)) -> (density: (N,1) >=0, rgb: (N,3) in [0,1])`. density softplus, rgb sigmoid.

- [ ] **Step 1: Testi yaz**

```python
import torch
from lrm.nerf import TriplaneNeRF

def test_nerf_outputs_ranges():
    net = TriplaneNeRF(in_dim=48, hidden=32)
    d, c = net(torch.randn(50, 48))
    assert d.shape == (50, 1) and c.shape == (50, 3)
    assert (d >= 0).all()
    assert (c >= 0).all() and (c <= 1).all()
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_nerf.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementasyonu yaz**

```python
"""Triplane ozelliginden yogunluk + renk ureten kucuk MLP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneNeRF(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.density_head = nn.Linear(hidden, 1)
        self.rgb_head = nn.Linear(hidden, 3)

    def forward(self, feats):
        h = self.backbone(feats)
        density = F.softplus(self.density_head(h))
        rgb = torch.sigmoid(self.rgb_head(h))
        return density, rgb
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_nerf.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/nerf.py tests/test_nerf.py
git commit -m "Faz B: TriplaneNeRF MLP - feature -> density+rgb"
```

---

### Task 9: renderer.py — volume rendering

**Files:**
- Create: `scripts/lrm/renderer.py`
- Test: `tests/test_renderer.py`

**Interfaces:**
- Produces: `volume_render(origins: (R,3), dirs: (R,3), near: float, far: float, n_samples: int, query_fn, white_bg=False) -> (rgb: (R,3), acc: (R,1))`. `query_fn(pts: (P,3)) -> (density: (P,1), rgb: (P,3))`. Stratified sampling (eğitimde jitter).

- [ ] **Step 1: Testi yaz**

```python
import torch
from lrm.renderer import volume_render

def test_empty_scene_zero_alpha():
    def q(pts):
        n = pts.shape[0]
        return torch.zeros(n, 1), torch.zeros(n, 3)
    o = torch.zeros(4, 3); d = torch.tensor([[0., 0., -1.]]).expand(4, 3).contiguous()
    rgb, acc = volume_render(o, d, 0.5, 2.0, 16, q)
    assert rgb.shape == (4, 3) and acc.shape == (4, 1)
    assert acc.max() < 1e-3

def test_dense_wall_high_alpha():
    # her yerde cok yuksek yogunluk, kirmizi => acc ~1, rgb ~ kirmizi
    def q(pts):
        n = pts.shape[0]
        dens = torch.full((n, 1), 1e3)
        col = torch.tensor([1., 0., 0.]).expand(n, 3)
        return dens, col
    o = torch.zeros(3, 3); d = torch.tensor([[0., 0., -1.]]).expand(3, 3).contiguous()
    rgb, acc = volume_render(o, d, 0.5, 2.0, 32, q)
    assert acc.min() > 0.9
    assert rgb[:, 0].min() > 0.8 and rgb[:, 1].max() < 0.2
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_renderer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementasyonu yaz**

```python
"""NeRF volume rendering: isin boyunca ornekle, yogunluk+renk entegre et."""
import torch


def volume_render(origins, dirs, near, far, n_samples, query_fn,
                  white_bg=False, jitter=None):
    device = origins.device
    R = origins.shape[0]
    t = torch.linspace(near, far, n_samples, device=device)  # (S,)
    t = t.expand(R, n_samples).clone()
    if jitter is None:
        jitter = torch.is_grad_enabled()
    if jitter:
        mids = 0.5 * (t[:, 1:] + t[:, :-1])
        lower = torch.cat([t[:, :1], mids], dim=1)
        upper = torch.cat([mids, t[:, -1:]], dim=1)
        t = lower + (upper - lower) * torch.rand_like(t)

    pts = origins[:, None, :] + dirs[:, None, :] * t[:, :, None]  # (R,S,3)
    density, rgb = query_fn(pts.reshape(-1, 3))
    density = density.reshape(R, n_samples)
    rgb = rgb.reshape(R, n_samples, 3)

    delta = t[:, 1:] - t[:, :-1]
    last = torch.full_like(delta[:, :1], 1e10)
    delta = torch.cat([delta, last], dim=1)  # (R,S)

    alpha = 1.0 - torch.exp(-density * delta)  # (R,S)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=1),
        dim=1)[:, :-1]
    weights = alpha * trans  # (R,S)

    rgb_out = (weights[..., None] * rgb).sum(dim=1)  # (R,3)
    acc = weights.sum(dim=1, keepdim=True)           # (R,1)
    if white_bg:
        rgb_out = rgb_out + (1.0 - acc)
    return rgb_out, acc
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_renderer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/renderer.py tests/test_renderer.py
git commit -m "Faz B: volume_render - ray sample + integrate"
```

---

### Task 10: losses.py — MSE + LPIPS + mask

**Files:**
- Create: `scripts/lrm/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces: `LRMLoss(w_mse=1.0, w_lpips=1.0, w_mask=0.5, use_lpips=True)`.
  `forward(pred_rgb: (V,3,H,W), pred_acc: (V,1,H,W), gt_rgb: (V,3,H,W), gt_alpha: (V,1,H,W)) -> (total: scalar, dict of components)`. LPIPS lazy (ilk kullanımda yüklenir), CPU testinde `use_lpips=False`.

- [ ] **Step 1: Testi yaz**

```python
import torch
from lrm.losses import LRMLoss

def test_zero_loss_on_identical():
    loss = LRMLoss(use_lpips=False)
    rgb = torch.rand(2, 3, 16, 16)
    alpha = torch.rand(2, 1, 16, 16)
    total, parts = loss(rgb, alpha, rgb, alpha)
    assert total.item() < 1e-6
    assert "mse" in parts and "mask" in parts

def test_mask_loss_increases_with_silhouette_diff():
    loss = LRMLoss(use_lpips=False, w_mse=0.0, w_mask=1.0)
    rgb = torch.zeros(1, 3, 8, 8)
    a1 = torch.zeros(1, 1, 8, 8)
    a2 = torch.ones(1, 1, 8, 8)
    same, _ = loss(rgb, a1, rgb, a1)
    diff, _ = loss(rgb, a1, rgb, a2)
    assert diff.item() > same.item()
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementasyonu yaz**

```python
"""LRM loss: MSE + LPIPS (algisal) + mask/alpha."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRMLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_lpips=1.0, w_mask=0.5, use_lpips=True):
        super().__init__()
        self.w_mse, self.w_lpips, self.w_mask = w_mse, w_lpips, w_mask
        self.use_lpips = use_lpips
        self._lpips = None  # lazy

    def _lpips_fn(self, device):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="vgg").to(device)
            for p in self._lpips.parameters():
                p.requires_grad_(False)
        return self._lpips

    def forward(self, pred_rgb, pred_acc, gt_rgb, gt_alpha):
        parts = {}
        mse = F.mse_loss(pred_rgb, gt_rgb)
        parts["mse"] = mse.detach()
        mask = F.l1_loss(pred_acc, gt_alpha)
        parts["mask"] = mask.detach()
        total = self.w_mse * mse + self.w_mask * mask
        if self.use_lpips and self.w_lpips > 0:
            fn = self._lpips_fn(pred_rgb.device)
            lp = fn(pred_rgb * 2 - 1, gt_rgb * 2 - 1).mean()
            parts["lpips"] = lp.detach()
            total = total + self.w_lpips * lp
        parts["total"] = total.detach()
        return total, parts
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_losses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/losses.py tests/test_losses.py
git commit -m "Faz B: LRMLoss - mse + lpips + mask"
```

---

### Task 11: model.py — uçtan uca LRM

**Files:**
- Create: `scripts/lrm/model.py`
- Test: `tests/test_model_shapes.py`

**Interfaces:**
- Consumes: encoder, transformer, triplane, nerf, renderer, cameras.
- Produces: `LRM(dim=512, depth=12, heads=8, triplane_res=32, triplane_ch=32, nerf_hidden=64, encoder=None, bound=0.6, near=0.8, far=2.2, n_samples=64)`.
  `forward(input_imgs, input_c2w, input_K, render_c2w, render_K, render_hw) -> (rgb: (Vr,3,H,W), acc: (Vr,1,H,W))`.
  - `input_imgs (Vi,3,224,224)`, `input_c2w (Vi,4,4)`, `input_K (Vi,3,3)`.
  - `render_c2w (Vr,4,4)`, `render_K (Vr,3,3)`, `render_hw=(H,W)`.
  Test için encoder enjekte edilebilir (fake) — indirmeyi CPU testinde atlamak için.

- [ ] **Step 1: Testi yaz** (sahte encoder ile CPU, küçük)

```python
import torch
import torch.nn as nn
from lrm.model import LRM

class FakeEncoder(nn.Module):
    embed_dim = 384
    patch = 14
    def forward(self, imgs):  # (V,3,224,224) -> (V,256,384)
        v = imgs.shape[0]
        return torch.randn(v, 256, 384)

def test_forward_output_shapes():
    model = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
                nerf_hidden=16, encoder=FakeEncoder(), n_samples=8)
    Vi, Vr, H, W = 2, 3, 16, 16
    imgs = torch.rand(Vi, 3, 224, 224)
    ic2w = torch.eye(4)[None].expand(Vi, 4, 4).contiguous()
    iK = torch.tensor([[100., 0, 8], [0, 100, 8], [0, 0, 1]])[None].expand(Vi, 3, 3).contiguous()
    rc2w = torch.eye(4)[None].expand(Vr, 4, 4).contiguous()
    rK = torch.tensor([[50., 0, 8], [0, 50, 8], [0, 0, 1]])[None].expand(Vr, 3, 3).contiguous()
    rgb, acc = model(imgs, ic2w, iK, rc2w, rK, (H, W))
    assert rgb.shape == (Vr, 3, H, W)
    assert acc.shape == (Vr, 1, H, W)

def test_forward_single_input_view():
    model = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
                nerf_hidden=16, encoder=FakeEncoder(), n_samples=8)
    imgs = torch.rand(1, 3, 224, 224)
    ic2w = torch.eye(4)[None]; iK = torch.eye(3)[None] * 100; iK[0, 2, 2] = 1
    rc2w = torch.eye(4)[None]; rK = iK.clone()
    rgb, acc = model(imgs, ic2w, iK, rc2w, rK, (8, 8))
    assert rgb.shape == (1, 3, 8, 8)
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_model_shapes.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementasyonu yaz**

```python
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
                 bound=0.6, near=0.8, far=2.2, n_samples=64):
        super().__init__()
        self.encoder = encoder if encoder is not None else DinoEncoder()
        self.transformer = LRMTransformer(dim=dim, depth=depth, heads=heads,
                                          triplane_res=triplane_res,
                                          img_dim=self.encoder.embed_dim)
        self.triplane_head = TriplaneHead(dim=dim, out_channels=triplane_ch)
        self.nerf = TriplaneNeRF(in_dim=3 * triplane_ch, hidden=nerf_hidden)
        self.bound, self.near, self.far, self.n_samples = bound, near, far, n_samples
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
            return self.nerf(feats)

        rgb, acc = volume_render(o, d, self.near, self.far, self.n_samples, query)
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
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_model_shapes.py -v`
Expected: PASS (2 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/lrm/model.py tests/test_model_shapes.py
git commit -m "Faz B: LRM uctan uca forward (encoder->triplane->render)"
```

---

### Task 12: overfit_lrm.py — overfit sağlık testi (ilk kapı)

Tam eğitimden ÖNCE gelir: pipeline gerçekten öğreniyor mu?

**Files:**
- Create: `scripts/overfit_lrm.py`
- Test: `tests/test_overfit_smoke.py` (çok küçük, CPU, sahte encoder — loss düşüyor mu)

**Interfaces:**
- Consumes: `LRM`, `LRMLoss`, `LRMDataset`.
- Produces: `overfit(uids, renders_dir, steps, device, ...) -> list[float]` (adım başına loss). CLI: `python scripts/overfit_lrm.py --n_obj 2 --steps 500`.
  Beklenti: loss belirgin düşer; sonda `dataset/lrm_val_previews/overfit_<uid>.png` (tahmin vs GT) yazar.

- [ ] **Step 1: Testi yaz** (loss düşüyor mu — sentetik, sahte encoder, birkaç adım)

```python
import json, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.losses import LRMLoss

class FakeEncoder(nn.Module):
    embed_dim = 384; patch = 14
    def forward(self, imgs):
        return torch.randn(imgs.shape[0], 256, 384)

def _tiny_dataset(tmp_path):
    rroot = tmp_path / "renders"; rroot.mkdir()
    uid = "u0"; d = rroot / uid; d.mkdir()
    meta = {"intrinsic": [[400, 0, 256], [0, 400, 256], [0, 0, 1]],
            "canonical_indices": [0, 1, 2, 3], "views": []}
    for i in range(16):
        arr = np.zeros((64, 64, 4), np.uint8)
        arr[20:44, 20:44, :3] = 200; arr[20:44, 20:44, 3] = 255
        Image.fromarray(arr, "RGBA").save(d / f"{i:03d}.png")
        c2w = np.eye(4); c2w[2, 3] = 1.5
        meta["views"].append({"extrinsic": c2w.tolist()})
    json.dump(meta, open(d / "meta.json", "w", encoding="utf-8"))
    tl = tmp_path / "tl.json"
    json.dump({"train": [uid], "val": []}, open(tl, "w", encoding="utf-8"))
    return str(tl), str(rroot)

def test_overfit_loss_decreases(tmp_path):
    tl, rroot = _tiny_dataset(tmp_path)
    ds = LRMDataset(tl, rroot, split="train", input_res=224, render_res=32,
                    n_sup=2, augment=False)
    model = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
                nerf_hidden=16, encoder=FakeEncoder(), n_samples=16)
    loss_fn = LRMLoss(use_lpips=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(40):
        it = ds[0]
        rgb, acc = model(it["input_imgs"], it["input_c2w"], it["input_K"],
                         it["sup_c2w"], it["sup_K"], (32, 32))
        total, _ = loss_fn(rgb, acc, it["sup_rgb"], it["sup_alpha"])
        opt.zero_grad(); total.backward(); opt.step()
        losses.append(total.item())
    assert losses[-1] < losses[0] * 0.7  # belirgin dusus
```

Not: sahte encoder her adımda rastgele token üretir → mükemmel ezber beklenmez; test yalnızca **loss düşüşü** arar (pipeline gradyan akıtıyor). Gerçek koşuda encoder deterministiktir.

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_overfit_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: scripts.overfit_lrm`/`overfit_lrm`).

- [ ] **Step 3: Implementasyonu yaz**

```python
"""Overfit saglik testi: 1-2 objede bilerek ezberlet. Loss ~0'a inip render
GT'ye oturmali. Tam egitimden ONCE pipeline'i dogrular."""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.losses import LRMLoss

PREVIEW_DIR = "dataset/lrm_val_previews"


def _save_preview(pred_rgb, gt_rgb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pr = (pred_rgb[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    gt = (gt_rgb[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    grid = np.concatenate([gt, pr], axis=1)
    Image.fromarray(grid).save(path)


def overfit(train_list, renders_dir, n_obj=2, steps=500, render_res=64,
            device="cuda", lr=4e-4, use_lpips=True):
    ds = LRMDataset(train_list, renders_dir, split="train", input_res=224,
                    render_res=render_res, n_sup=4, augment=False)
    ds.uids = ds.uids[:n_obj]
    model = LRM(n_samples=48).to(device)
    model.transformer.enable_checkpointing()
    loss_fn = LRMLoss(use_lpips=use_lpips).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        it = ds[step % len(ds)]
        to = lambda t: t.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rgb, acc = model(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]),
                             to(it["sup_c2w"]), to(it["sup_K"]), (render_res, render_res))
            total, parts = loss_fn(rgb, acc, to(it["sup_rgb"]), to(it["sup_alpha"]))
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())
        if step % 50 == 0:
            print(f"step {step}: loss={total.item():.4f} "
                  f"mse={parts['mse'].item():.4f} mask={parts['mask'].item():.4f}")
            _save_preview(rgb, to(it["sup_rgb"]),
                          os.path.join(PREVIEW_DIR, f"overfit_{it['uid'][:8]}.png"))
    return losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list.json")
    ap.add_argument("--renders_dir", default="dataset/renders")
    ap.add_argument("--n_obj", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--render_res", type=int, default=64)
    ap.add_argument("--no_lpips", action="store_true")
    a = ap.parse_args()
    losses = overfit(a.train_list, a.renders_dir, a.n_obj, a.steps, a.render_res,
                     use_lpips=not a.no_lpips)
    print(f"ilk loss={losses[0]:.4f}  son loss={losses[-1]:.4f}")
    print("BASARILI: pipeline ogreniyor" if losses[-1] < losses[0] * 0.3
          else "DIKKAT: loss yeterince dusmedi - bug arastir")
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_overfit_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: GPU'da gerçek overfit koşusu (elle sağlık kapısı)**

Run: `python scripts/overfit_lrm.py --n_obj 2 --steps 800`
Expected: loss belirgin düşer (son < ilk×0.3), `dataset/lrm_val_previews/overfit_*.png`'de sağ (tahmin) sol (GT) benzer. Oturmazsa DUR ve systematic-debugging ile bug ara — tam eğitime geçme.

- [ ] **Step 6: Commit**

```bash
git add scripts/overfit_lrm.py tests/test_overfit_smoke.py
git commit -m "Faz B: overfit saglik testi (ilk kapi)"
```

---

### Task 13: train_lrm.py — tam eğitim döngüsü

**Files:**
- Create: `scripts/train_lrm.py`
- Test: `tests/test_train_helpers.py` (checkpoint save/load + preview grid — CPU)

**Interfaces:**
- Consumes: `LRM`, `LRMLoss`, `LRMDataset`, `lrm_collate`.
- Produces:
  - `save_checkpoint(path, model, opt, scheduler, step)` / `load_checkpoint(path, model, opt, scheduler) -> step`
  - `save_val_grid(model, dataset, uids, device, path, render_res)` — birkaç val objesi render edip PNG grid (Faz A montaj mantığı)
  - `train(...)` ana döngü. CLI: `python scripts/train_lrm.py --steps 40000 --micro_batch 2 --grad_accum 4 --render_res 128 --resume`.
  - Log: `dataset/renders/logs/` mantığına benzer zaman damgalı log; loss bileşenleri + lr + hız.

- [ ] **Step 1: Testi yaz** (checkpoint round-trip + val grid dosyası)

```python
import os
import torch
import torch.nn as nn
from lrm.model import LRM
from train_lrm import save_checkpoint, load_checkpoint

class FakeEncoder(nn.Module):
    embed_dim = 384; patch = 14
    def forward(self, imgs): return torch.randn(imgs.shape[0], 256, 384)

def test_checkpoint_roundtrip(tmp_path):
    m = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
            nerf_hidden=16, encoder=FakeEncoder())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    p = str(tmp_path / "ck.pt")
    save_checkpoint(p, m, opt, sched, step=123)
    m2 = LRM(dim=32, depth=2, heads=4, triplane_res=8, triplane_ch=8,
             nerf_hidden=16, encoder=FakeEncoder())
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.ConstantLR(opt2)
    step = load_checkpoint(p, m2, opt2, sched2)
    assert step == 123
    # triplane_head agirliklari esit yuklendi
    a = m.triplane_head.proj.weight
    b = m2.triplane_head.proj.weight
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Testi çalıştır, başarısız gör**

Run: `pytest tests/test_train_helpers.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementasyonu yaz**

```python
"""Tam LRM egitim dongusu: mikro-batch + gradient accumulation, bf16,
checkpoint/resume, val preview, zaman damgali log."""
import argparse
import logging
import os
import time

import numpy as np
import torch
from PIL import Image

from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.losses import LRMLoss

CKPT_DIR = "dataset/lrm_ckpts"
PREVIEW_DIR = "dataset/lrm_val_previews"
LOG_DIR = "dataset/lrm_logs"


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("train_lrm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{ts}.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    for h in (fh, ch):
        h.setFormatter(fmt); logger.addHandler(h)
    return logger


def save_checkpoint(path, model, opt, scheduler, step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": scheduler.state_dict(), "step": step}, path)


def load_checkpoint(path, model, opt, scheduler):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scheduler.load_state_dict(ck["sched"])
    return ck["step"]


def save_val_grid(model, dataset, uids, device, path, render_res):
    model.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    with torch.no_grad():
        for uid in uids:
            idx = dataset.uids.index(uid)
            it = dataset[idx]
            to = lambda t: t.to(device)
            rgb, _ = model(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]),
                           to(it["sup_c2w"][:1]), to(it["sup_K"][:1]), (render_res, render_res))
            pr = (rgb[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt = (it["sup_rgb"][0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            rows.append(np.concatenate([gt, pr], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    model.train()


def train(train_list, renders_dir, steps=40000, micro_batch=2, grad_accum=4,
          render_res=128, n_sup=4, lr=4e-4, warmup=500, ckpt_every=2000,
          val_every=1000, resume=False, device="cuda"):
    logger = setup_logging()
    train_ds = LRMDataset(train_list, renders_dir, split="train",
                          render_res=render_res, n_sup=n_sup, augment=True)
    val_ds = LRMDataset(train_list, renders_dir, split="val",
                        render_res=render_res, n_sup=n_sup, augment=False)
    val_uids = val_ds.uids[:6]

    model = LRM(n_samples=64).to(device)
    model.transformer.enable_checkpointing()
    loss_fn = LRMLoss(use_lpips=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        import math
        prog = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_step = 0
    last_ckpt = os.path.join(CKPT_DIR, "last.pt")
    if resume and os.path.isfile(last_ckpt):
        start_step = load_checkpoint(last_ckpt, model, opt, scheduler)
        logger.info(f"resume: step {start_step}")

    rng = np.random.default_rng(0)
    model.train()
    t0 = time.time()
    for step in range(start_step, steps):
        opt.zero_grad()
        agg = {"total": 0.0, "mse": 0.0, "mask": 0.0, "lpips": 0.0}
        for _ in range(grad_accum):
            for _ in range(micro_batch):
                it = train_ds[int(rng.integers(len(train_ds)))]
                to = lambda t: t.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    rgb, acc = model(to(it["input_imgs"]), to(it["input_c2w"]),
                                     to(it["input_K"]), to(it["sup_c2w"]), to(it["sup_K"]),
                                     (render_res, render_res))
                    total, parts = loss_fn(rgb, acc, to(it["sup_rgb"]), to(it["sup_alpha"]))
                (total / (grad_accum * micro_batch)).backward()
                for k in agg:
                    if k in parts:
                        agg[k] += parts[k].item() / (grad_accum * micro_batch)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        scheduler.step()

        if step % 20 == 0:
            speed = (step - start_step + 1) / (time.time() - t0)
            vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            logger.info(f"step {step}/{steps} loss={agg['total']:.4f} "
                        f"mse={agg['mse']:.4f} mask={agg['mask']:.4f} lpips={agg['lpips']:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} {speed:.2f}it/s vram={vram:.1f}GB")
        if step > 0 and step % val_every == 0:
            save_val_grid(model, val_ds, val_uids, device,
                          os.path.join(PREVIEW_DIR, f"val_{step:06d}.png"), render_res)
        if step > 0 and step % ckpt_every == 0:
            save_checkpoint(last_ckpt, model, opt, scheduler, step)
            logger.info(f"checkpoint kaydedildi: {last_ckpt}")
    save_checkpoint(last_ckpt, model, opt, scheduler, steps)
    logger.info("egitim bitti")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list.json")
    ap.add_argument("--renders_dir", default="dataset/renders")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--render_res", type=int, default=128)
    ap.add_argument("--n_sup", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(a.train_list, a.renders_dir, a.steps, a.micro_batch, a.grad_accum,
          a.render_res, a.n_sup, a.lr, resume=a.resume)
```

- [ ] **Step 4: Testi çalıştır, geç**

Run: `pytest tests/test_train_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Kısa duman koşusu (GPU, birkaç yüz adım)**

Run: `python scripts/train_lrm.py --steps 300 --render_res 64 --micro_batch 1 --grad_accum 2`
Expected: çöküş yok, OOM yok, loss düşüyor, `dataset/lrm_val_previews/val_*.png` yazılıyor, VRAM 16GB altında. VRAM taşarsa `--render_res 64` / `--n_sup 2` / n_samples düşür.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_lrm.py tests/test_train_helpers.py
git commit -m "Faz B: train_lrm - egitim dongusu + checkpoint/resume + val preview"
```

---

## Self-Review (plan yazarının kontrolü)

- **Spec kapsamı:** dataloader (T4), augmentation (T3), encoder (T5), transformer (T6),
  triplane (T7), nerf (T8), renderer (T9), loss (T10), model (T11), eğitim döngüsü +
  checkpoint/resume + val preview (T13), overfit sağlık kapısı (T12), ortam/PyTorch cu128 (T1),
  cameras/Plücker (T2) — tüm spec bölümleri bir task'a bağlı. ✓
- **Test stratejisi:** her modülün kendi TDD testi; ağır GPU/indirme testleri `-m slow` /
  elle koşu adımları. ✓
- **Tip tutarlılığı:** görüntü `(V,3,H,W)`, poz c2w `(V,4,4)`, intrinsic `(V,3,3)`;
  `make_triplane`/`render_view`/`volume_render`/`sample_triplane` imzaları task'lar arası uyumlu. ✓
- **Placeholder yok:** her adımda gerçek test + implementasyon kodu var. ✓
- **Bağımlılık sırası:** T1→T2→(T3,T4,T5)→T6→T7→T8→T9→T10→T11→T12→T13. ✓

## Riskler / İlk ayar düğmeleri (uygulama sırasında)

1. **VRAM:** ilk kısa koşuda taşarsa sırasıyla düşür: `render_res` 128→64, `n_sup` 4→2,
   `n_samples` 64→48, `micro_batch` → 1. Transformer checkpointing zaten açık.
2. **Convergence sinyali yok:** overfit testi (T12) geçmeden T13'e geçme.
3. **DINOv2 indirme:** T5 ilk koşuda internet ister; kurumsal proxy varsa `torch.hub` cache'i kontrol et.
4. **Blackwell:** T1 geçmeden hiçbir GPU adımına geçme.
