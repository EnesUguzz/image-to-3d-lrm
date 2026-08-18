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
    g = g / g.sum()
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
