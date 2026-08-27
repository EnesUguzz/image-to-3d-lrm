"""LRM loss: (foreground-agirlikli) MSE + LPIPS (algisal) + mask/alpha.

Arka plan objeden cok daha genis oldugu icin ('bos uret' tuzagi) obje pikselleri
(gt_alpha>0) fg_weight ile agirlandirilir; boylece bos sahne dusuk loss olmaz."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRMLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_lpips=1.0, w_mask=1.0, use_lpips=True,
                 fg_weight=5.0, mask_fg_weight=None):
        super().__init__()
        self.w_mse, self.w_lpips, self.w_mask = w_mse, w_lpips, w_mask
        self.use_lpips = use_lpips
        self.fg_weight = fg_weight
        # mask_fg_weight=None => eski davranis (RGB ile ayni agirlik). AYRI OLMASININ
        # SEBEBI (2026-08-27, olculdu): fg_weight TAM KARE denetimi icin eklenmisti,
        # orada arka plan alanca eziciydi. Bolge kirpmasiyla (fg_bias=0.75, 64^2 yama)
        # arka plan zaten azinlikta; ayni duzeltme cift sayilinca 'bosluk oy' sinyali
        # mask kaybinin sadece ~%14'une dusuyor => model bound kupunu doldurup oturuyor
        # (kol A: 6000 adim boyunca acc~0.50, mask 0.45'te cakili, PSNR 8.8 dB).
        self.mask_fg_weight = fg_weight if mask_fg_weight is None else mask_fg_weight
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
        # foreground-agirlik (obje pikselleri agir): MSE'de obje renkleri
        # 'bos uret'e cokmesin; mask'ta bos sahne cezalansin (empty->yuksek mask).
        # Uniform mask 'bos uret'i odullendirip cokusu tetikliyordu.
        w = 1.0 + self.fg_weight * gt_alpha            # (V,1,H,W)
        w_rgb = w.expand_as(pred_rgb)
        mse = (w_rgb * (pred_rgb - gt_rgb) ** 2).sum() / w_rgb.sum().clamp_min(1e-8)
        parts["mse"] = mse.detach()
        wm = (w if self.mask_fg_weight == self.fg_weight
              else 1.0 + self.mask_fg_weight * gt_alpha)
        mask = (wm * (pred_acc - gt_alpha).abs()).sum() / wm.sum().clamp_min(1e-8)
        parts["mask"] = mask.detach()
        total = self.w_mse * mse + self.w_mask * mask
        if self.use_lpips and self.w_lpips > 0:
            fn = self._lpips_fn(pred_rgb.device)
            lp = fn(pred_rgb * 2 - 1, gt_rgb * 2 - 1).mean()
            parts["lpips"] = lp.detach()
            total = total + self.w_lpips * lp
        parts["total"] = total.detach()
        return total, parts
