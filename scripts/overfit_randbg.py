"""Hipotez testi: her adim RASTGELE arka plan rengi => model sabit ciktiyla
arka plani tutturamaz, objeyi gercekten kurmak zorunda kalir. Beyaz-cokme kalkmali.
Loss <0.3'e inip onizlemede renkli/ayirt edilebilir objeler cikmali."""
import argparse
import os

import numpy as np
import torch
from lrm import defaults
from PIL import Image

from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.losses import LRMLoss
from overfit_lrm import _pick_high_coverage

PREVIEW_DIR = "dataset/lrm_val_previews"
TAG = ""  # config etiketi (buyuk-model testinde baseline'i ezmemek icin)


def _save_preview(pred_rgb, gt_rgb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pr = (pred_rgb[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    gt = (gt_rgb[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([gt, pr], axis=1)).save(path)


def overfit(train_list, renders_dir, n_obj=4, steps=2500, render_res=64,
            device="cuda", lr=5e-4, use_lpips=True, amp=False,
            dim=512, triplane_res=32, triplane_ch=32):
    import contextlib
    # matematigi DEGISTIRMEYEN hizlandirmalar (TF32 matmul + en hizli kernel secimi);
    # bf16 BILEREK yok (rengi olduruyor, bu test tam da rengi/keskinligi olcuyor)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    ds = LRMDataset(train_list, renders_dir, split="train", input_res=224,
                    render_res=render_res, n_sup=4, augment=False)
    ds.uids = _pick_high_coverage(train_list, renders_dir, n_obj)
    model = LRM(dim=dim, triplane_res=triplane_res, triplane_ch=triplane_ch,
                n_samples=defaults.N_SAMPLES).to(device)
    print(f"model: dim={dim} triplane_res={triplane_res} triplane_ch={triplane_ch} "
          f"| params={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M",
          flush=True)
    model.transformer.enable_checkpointing()
    loss_fn = LRMLoss(use_lpips=use_lpips).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        it = ds[step % len(ds)]
        to = lambda t: t.to(device)
        alpha = to(it["sup_alpha"])                       # (V,1,H,W)
        # beyaza kompozit sup_rgb'den premultiplied obje rengini geri al:
        # sup_rgb = obj_premult + (1-alpha)*1  =>  obj_premult = sup_rgb - (1-alpha)
        obj_premult = to(it["sup_rgb"]) - (1.0 - alpha)   # (V,3,H,W)
        c = torch.rand(3, device=device)                  # rastgele bg (bu adim)
        target = obj_premult + (1.0 - alpha) * c[None, :, None, None]
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if amp else contextlib.nullcontext())
        with ctx:
            rgb, acc = model(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]),
                             to(it["sup_c2w"]), to(it["sup_K"]), (render_res, render_res),
                             bg_color=c)
            total, parts = loss_fn(rgb, acc, target, alpha)
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())
        if step % 50 == 0:
            lp = parts.get("lpips")
            lp = f" lpips={lp.item():.4f}" if lp is not None else ""
            print(f"step {step}: loss={total.item():.4f} mse={parts['mse'].item():.4f} "
                  f"mask={parts['mask'].item():.4f}{lp}", flush=True)
            # onizleme: BEYAZ bg'de render, beyaz-kompozit GT ile karsilastir
            with torch.no_grad():
                prgb, _ = model(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]),
                                to(it["sup_c2w"][:1]), to(it["sup_K"][:1]),
                                (render_res, render_res), bg_color=torch.ones(3, device=device))
            _save_preview(prgb, to(it["sup_rgb"][:1]),
                          os.path.join(PREVIEW_DIR, f"randbg_{TAG}{it['uid'][:8]}.png"))
    return losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--n_obj", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--render_res", type=int, default=64)
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--amp", action="store_true", help="bf16 mixed precision testi")
    ap.add_argument("--dim", type=int, default=512, help="transformer genisligi (tavan testi: 768)")
    ap.add_argument("--triplane_res", type=int, default=32, help="triplane token izgarasi (tavan testi: 64)")
    ap.add_argument("--triplane_ch", type=int, default=32)
    ap.add_argument("--tag", default="", help="onizleme dosya adi oneki (karsilastirma icin)")
    a = ap.parse_args()
    TAG = a.tag
    losses = overfit(a.train_list, a.renders_dir, a.n_obj, a.steps, a.render_res,
                     lr=a.lr, use_lpips=not a.no_lpips, amp=a.amp,
                     dim=a.dim, triplane_res=a.triplane_res, triplane_ch=a.triplane_ch)
    print(f"ilk loss={losses[0]:.4f}  son loss={losses[-1]:.4f}", flush=True)
    print("BASARILI: renk-cokme kalkti, model ogreniyor" if losses[-1] < 0.30
          else "DIKKAT: hala takili - baska kok neden var", flush=True)
