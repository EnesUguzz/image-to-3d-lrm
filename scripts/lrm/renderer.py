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
