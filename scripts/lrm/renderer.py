"""NeRF volume rendering: isin boyunca ornekle, yogunluk+renk entegre et."""
import torch


def volume_render(origins, dirs, near, far, n_samples, query_fn,
                  white_bg=False, jitter=None, bg_color=None):
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
    return render_with_t(origins, dirs, t, query_fn,
                         white_bg=white_bg, bg_color=bg_color)


def render_with_t(origins, dirs, t, query_fn, white_bg=False, bg_color=None,
                  agirlik_dondur=False):
    """Verilen (SIRALI) t ornekleriyle hacim render'i.

    volume_render tekduze t uretip bunu cagirir; onem ornekleme tekduze
    OLMAYAN t uretir. Delta'lar t farkindan hesaplandigi icin ikisi de dogru.
    """
    R, n_samples = t.shape
    pts = origins[:, None, :] + dirs[:, None, :] * t[:, :, None]  # (R,S,3)
    density, rgb = query_fn(pts.reshape(-1, 3))
    density = density.reshape(R, n_samples)
    rgb = rgb.reshape(R, n_samples, 3)

    # Sinirli obje hacmi: son delta'yi son gercek aralikla tekrarla (SONLU).
    # 1e10 kullanmak her isini zorla opak yapar => seffaf arka plan ogrenilemez.
    delta = t[:, 1:] - t[:, :-1]
    delta = torch.cat([delta, delta[:, -1:]], dim=1)  # (R,S)

    alpha = 1.0 - torch.exp(-density * delta)  # (R,S)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=1),
        dim=1)[:, :-1]
    weights = alpha * trans  # (R,S)

    rgb_out = (weights[..., None] * rgb).sum(dim=1)  # (R,3)
    acc = weights.sum(dim=1, keepdim=True)           # (R,1)
    # bg_color: her adim rastgele renk => model sabit ciktiyla arka plani
    # tutturamaz, objeyi (renk+opaklik) gercekten kurmak zorunda kalir.
    if bg_color is not None:
        rgb_out = rgb_out + (1.0 - acc) * bg_color
    elif white_bg:
        rgb_out = rgb_out + (1.0 - acc)
    if agirlik_dondur:
        return rgb_out, acc, weights
    return rgb_out, acc


def volume_render_chunked(origins, dirs, near, far, n_samples, query_fn,
                          chunk=0, **kw):
    """volume_render, ama isinlari parcalayip her parcayi gradyan
    checkpoint'inden gecirir.

    NEDEN (2026-08-31): volume_render tum (isin x ornek) aktivasyonlarini
    ayni anda tutuyor. res 256 + n_samples 192 + 2 obje = 25M nokta => OOM.
    Bu duvar bugune kadar tarifeyi belirledi (n_sup 1'e mecbur kalindi,
    yuksek cozunurluk denenemedi). Parcalamayla bellek O(chunk)'a iner;
    bedeli geri yayilimda yeniden hesap (~x1.3-2 sure).

    jitter ACIKCA verilir: volume_render varsayilani torch.is_grad_enabled()
    ve checkpoint yeniden hesabinda bu bayrak degisebilir => ileri gecis ile
    yeniden hesap FARKLI ornekleme yapar ve gradyan sessizce bozulurdu.
    (checkpoint RNG durumunu korur, ama grad bayragini korumaz.)
    """
    if not chunk or origins.shape[0] <= chunk:
        return volume_render(origins, dirs, near, far, n_samples, query_fn, **kw)
    kw = dict(kw)
    kw.setdefault("jitter", torch.is_grad_enabled())
    rs, as_ = [], []
    for i in range(0, origins.shape[0], chunk):
        o, d = origins[i:i + chunk], dirs[i:i + chunk]
        if torch.is_grad_enabled():
            r, a = torch.utils.checkpoint.checkpoint(
                lambda o_, d_: volume_render(o_, d_, near, far, n_samples,
                                             query_fn, **kw),
                o, d, use_reentrant=False)
        else:
            r, a = volume_render(o, d, near, far, n_samples, query_fn, **kw)
        rs.append(r); as_.append(a)
    return torch.cat(rs, 0), torch.cat(as_, 0)


def _pdf_ornekle(t, weights, n_fine, det=False):
    """Agirlik PDF'inden ters-CDF ile n_fine ornek cek (NeRF hiyerarsik).

    weights DETACH edilir: gradyan ince gecisten akar, ornekleme kararindan
    degil (orijinal NeRF de boyle yapar; aksi halde gradyan ornekleme
    indeksleri uzerinden akmaya calisir ve gurultulu olur).
    """
    w = weights.detach()[:, 1:-1] + 1e-5
    pdf = w / w.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)
    mids = 0.5 * (t[:, 1:] + t[:, :-1])
    if det:
        u = torch.linspace(0., 1., n_fine, device=t.device).expand(t.shape[0], n_fine)
    else:
        u = torch.rand(t.shape[0], n_fine, device=t.device)
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    lo = (idx - 1).clamp(min=0)
    hi = idx.clamp(max=cdf.shape[-1] - 1)
    cdf_lo = torch.gather(cdf, 1, lo); cdf_hi = torch.gather(cdf, 1, hi)
    t_lo = torch.gather(mids, 1, lo.clamp(max=mids.shape[1] - 1))
    t_hi = torch.gather(mids, 1, hi.clamp(max=mids.shape[1] - 1))
    d = (cdf_hi - cdf_lo).clamp(min=1e-5)
    return t_lo + (u - cdf_lo) / d * (t_hi - t_lo)


def volume_render_importance(origins, dirs, near, far, n_coarse, n_fine,
                             query_fn, white_bg=False, bg_color=None,
                             jitter=None, chunk=0):
    """Iki gecisli: kaba tekduze -> agirliklara gore ince ornekleme.

    NEDEN (2026-08-31): tekduze ornekliyorduk. Isin [0.8, 2.2] araliginda
    ilerliyor ama obje bunun kucuk bir kismi; orneklerin cogu BOSLUKTA
    harcaniyor. Literatur (NeRF hiyerarsik ornekleme) bunun dogrudan
    "daha keskin doku" verdigini raporluyor.

    Kaba gecis no_grad: sadece NEREYE bakilacagina karar veriyor.
    """
    if jitter is None:
        jitter = torch.is_grad_enabled()
    # PARCALAMA (2026-09-02 kod incelemesi): bu fonksiyon `chunk` KABUL
    # ETMIYORDU, dolayisiyla `fit_teacher --n_fine N` verildiginde --ray_chunk
    # SESSIZCE dusuyordu. Ustelik ince gecis isin basina n_coarse+n_fine ornek
    # tasiyor, yani tam da parcalamaya en cok ihtiyac duyan yol. Belgelenen
    # kombinasyon (--res 256 --n_samples 192 --ray_chunk 32768 --n_fine ...)
    # OOM veriyordu.
    if chunk and origins.shape[0] > chunk:
        rs, as_ = [], []
        for i in range(0, origins.shape[0], chunk):
            o_, d_ = origins[i:i + chunk], dirs[i:i + chunk]
            if torch.is_grad_enabled():
                r, ac = torch.utils.checkpoint.checkpoint(
                    lambda oo, dd: volume_render_importance(
                        oo, dd, near, far, n_coarse, n_fine, query_fn,
                        white_bg=white_bg, bg_color=bg_color, jitter=jitter),
                    o_, d_, use_reentrant=False)
            else:
                r, ac = volume_render_importance(
                    o_, d_, near, far, n_coarse, n_fine, query_fn,
                    white_bg=white_bg, bg_color=bg_color, jitter=jitter)
            rs.append(r); as_.append(ac)
        return torch.cat(rs, 0), torch.cat(as_, 0)
    R = origins.shape[0]
    t = torch.linspace(near, far, n_coarse, device=origins.device)
    t = t.expand(R, n_coarse).clone()
    if jitter:
        mids = 0.5 * (t[:, 1:] + t[:, :-1])
        lo = torch.cat([t[:, :1], mids], 1); hi = torch.cat([mids, t[:, -1:]], 1)
        t = lo + (hi - lo) * torch.rand_like(t)
    with torch.no_grad():
        _, _, w = render_with_t(origins, dirs, t, query_fn, agirlik_dondur=True)
    t_f = _pdf_ornekle(t, w, n_fine, det=not jitter)
    t_all, _ = torch.sort(torch.cat([t, t_f], -1), dim=-1)
    return render_with_t(origins, dirs, t_all, query_fn,
                         white_bg=white_bg, bg_color=bg_color)
