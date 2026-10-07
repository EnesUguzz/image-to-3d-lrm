"""Ogretmen triplane'leri kendileri ogreniyor mu? Checkpoint'teki ogretmeni
modelin NeRF'iyle render edip GT ile karsilastirir.
  - Ogretmen keskin  => onyukleme saglam, ogrenci takip edecek
  - Ogretmen gurultu => distilasyon hedefi cop, kosuyu kesmek gerek
"""
import json, math, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras
from lrm import defaults
from lrm.dataset import LRMDataset
from lrm.model import LRM
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render

DEV, RES = "cuda", 128
ck = torch.load("dataset/lrm_ckpts/last.pt", map_location="cpu", weights_only=False)
print(f"checkpoint step: {ck['step']}")
if "teacher" in ck:
    teacher = ck["teacher"]
    print("ogretmen: checkpoint'ten")
else:   # donuk ogretmen checkpoint'e yazilmaz; asama-1 dosyasindan oku
    teacher = torch.load("dataset/lrm_ckpts/teacher_init.pt", map_location="cpu", weights_only=False)["triplanes"]
    print("ogretmen: teacher_init.pt'den (donuk)")
print(f"ogretmen: {tuple(teacher.shape)}  std={teacher.std():.4f}  |mean|={teacher.mean().abs():.4f}")

model = LRM(n_samples=defaults.N_SAMPLES).to(DEV)
model.load_state_dict(ck["model"]); model.eval()
ds = LRMDataset("dataset/train_list.json", "dataset/renders", split="train",
                render_res=RES, n_sup=4, augment=False)

@torch.no_grad()
def render(tp, c2w, K):
    o, d = cameras.rays_from_camera(c2w, K, RES, RES)
    def q(pts):
        den, rgb = model.nerf(sample_triplane(tp, pts, bound=model.bound))
        ins = (pts.abs().amax(-1, keepdim=True) <= model.bound).to(den.dtype)
        return den*ins, rgb
    rgb, acc = volume_render(o.to(DEV), d.to(DEV), model.near, model.far, 64, q, jitter=False)
    return (rgb + (1-acc)).reshape(RES,RES,3).permute(2,0,1)

# ASIL OLCUM: ogrencinin triplane'i ogretmeninkine yaklasiyor mu?
# Render kalitesi yaniltici olabilir (NeRF olcegi uyusmazsa bos cikar);
# distilasyon triplane UZAYINDA calisir, o yuzden orada olcmek gerekir.
tinit = torch.load("dataset/lrm_ckpts/teacher_init.pt", map_location="cpu", weights_only=False)
tp_ref = tinit["triplanes"]
rel = []
with torch.no_grad():
    for i in range(8):
        it = ds[i]
        tp_s = model.make_triplane(it["input_imgs"].to(DEV), it["input_c2w"].to(DEV),
                                   it["input_K"].to(DEV)).cpu()
        t = tp_ref[i]
        rel.append(float(((tp_s - t) ** 2).mean() / t.var().clamp_min(1e-6)))
        if i == 0:
            print(f"ogrenci triplane std={tp_s.std():.4f}  ogretmen std={t.std():.4f}")
print(f"triplane bagil hata (1.0 = tamamen alakasiz): {np.mean(rel):.3f}")

rows, ps_t, ps_s = [], [], []
with torch.no_grad():
    for i in [0, 1, 2, 3, 4, 5]:
        it = ds[i]
        gt = it["sup_rgb"][0].to(DEV)
        c2w, K = it["sup_c2w"][:1].to(DEV), it["sup_K"][:1].to(DEV)
        t_img = render(teacher[i].to(DEV), c2w[0], K[0])
        s_rgb, _ = model(it["input_imgs"].to(DEV), it["input_c2w"].to(DEV),
                         it["input_K"].to(DEV), c2w, K, (RES, RES))
        ps_t.append(-10*math.log10(max(float(((t_img-gt)**2).mean()),1e-9)))
        ps_s.append(-10*math.log10(max(float(((s_rgb[0]-gt)**2).mean()),1e-9)))
        rows.append(torch.cat([gt, t_img, s_rgb[0].clamp(0,1)], -1).cpu())
print(f"\nOGRETMEN PSNR ort = {np.mean(ps_t):.2f} dB   {np.round(ps_t,1)}")
print(f"OGRENCI  PSNR ort = {np.mean(ps_s):.2f} dB   {np.round(ps_s,1)}")
Image.fromarray((torch.cat(rows,1).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
                ).save("dataset/lrm_val_previews/diag_ogretmen.png")
print("gorsel: dataset/lrm_val_previews/diag_ogretmen.png  [GT | OGRETMEN | OGRENCI]")
