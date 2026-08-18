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
