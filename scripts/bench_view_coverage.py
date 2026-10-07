"""AYIRT EDICI TEST: gorunum KAPSAMASI yeterli mi?

Soru: mevcut sema (ring12) objenin ALTINI hic gormuyor (elevation -10..+80).
Alt gorunum eklemek (sphere20) gercekten bilgi katiyor mu, yoksa 16 yan/ust
gorunum objenin tamamini zaten belirliyor mu?

Encoder/transformer YOK -- her objeye kendi serbest triplane'i verilir (oracle).
Boylece olculen sey saf BILGI meselesi: "bu gorunum kumesi objeyi belirliyor mu?"

Tasarim (adil karsilastirma):
  eval  : el in (-70, -25) araligindaki 3 gorunum. HER IKI kosulda da fit'ten CIKARILIR.
  A_fit : K gorunum, HEPSI el >= -10   (ring12 kapsamasi: alt yok)
  B_fit : K gorunum, icinde alt kutup (el < -70) var, gerisi el >= -10
  -> Ikisinde de fit gorunum SAYISI ayni (K). Tek fark: B alti goruyor.

A ve B ayni held-out alt gorunumlerde olculur. Fark = alt kapsamanin degeri.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras
from lrm import defaults
from lrm.nerf import TriplaneNeRF
from lrm.triplane import sample_triplane
from lrm.renderer import volume_render
from lrm.losses import LRMLoss

DEV = "cuda"
OUT = "dataset/lrm_bench"


def load_view(render_dir, uid, vi, res):
    with open(os.path.join(render_dir, uid, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    v = meta["views"][vi]
    im = Image.open(os.path.join(render_dir, uid, v["file"])).convert("RGBA")
    if im.size[0] != res:
        im = im.resize((res, res), Image.BILINEAR)
    arr = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0
    K = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                 meta.get("resolution", 512), res)
    c2w = torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float32))
    return arr, c2w, K


def split_views(meta, k_fit, include_bottom, rng):
    """(fit_idx, eval_idx) dondurur; ikisi kesismez, fit boyu her kosulda ayni."""
    el = [float(v["elevation_deg"]) for v in meta["views"]]
    ev = [i for i, e in enumerate(el) if -70.0 < e < -25.0]
    if len(ev) < 2:
        return None
    rng.shuffle(ev)
    eval_idx = sorted(ev[:3])
    rest = [i for i in range(len(el)) if i not in eval_idx]
    upper = [i for i in rest if el[i] >= -10.0]
    bottom = [i for i in rest if el[i] <= -70.0]     # alt kutup (sphere20 kanonik)
    rng.shuffle(upper)
    rng.shuffle(bottom)
    if include_bottom:
        take_b = bottom[:1]
        fit = take_b + upper[:max(0, k_fit - len(take_b))]
    else:
        fit = upper[:k_fit]
    if len(fit) < k_fit:
        return None
    return sorted(fit), eval_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", default="dataset/_viewcov")
    ap.add_argument("--uids_json", default="", help="uid listesi json (yoksa klasorden okur)")
    ap.add_argument("--n_obj", type=int, default=40)
    ap.add_argument("--k_fit", type=int, default=12, help="her kosulda fit gorunum sayisi")
    ap.add_argument("--include_bottom", type=int, default=0, help="1 = alt kutbu fit'e kat")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2, help="adim basina obje")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--tp_res", type=int, default=64)
    ap.add_argument("--tp_ch", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=defaults.N_SAMPLES)
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--bound", type=float, default=0.6)
    ap.add_argument("--tag", default="viewcov")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    if a.uids_json:
        uids = json.load(open(a.uids_json, encoding="utf-8"))
        if isinstance(uids, dict):
            uids = uids.get("train", [])
    else:
        uids = sorted(d for d in os.listdir(a.render_dir)
                      if os.path.isdir(os.path.join(a.render_dir, d)))
    uids = uids[:a.n_obj]

    rng = np.random.default_rng(1234)
    data = []
    for u in uids:
        mp = os.path.join(a.render_dir, u, "meta.json")
        if not os.path.isfile(mp):
            continue
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        sp = split_views(meta, a.k_fit, a.include_bottom, rng)
        if sp is None:
            continue
        fit_idx, eval_idx = sp

        def pack(idxs):
            prem, alpha, c2w, K = [], [], [], []
            for i in idxs:
                arr, c, k = load_view(a.render_dir, u, i, a.res)
                prem.append(arr[:3] * arr[3:4]); alpha.append(arr[3:4])
                c2w.append(c); K.append(k)
            return dict(prem=torch.stack(prem).to(DEV), alpha=torch.stack(alpha).to(DEV),
                        c2w=torch.stack(c2w).to(DEV), K=torch.stack(K).to(DEV))

        d = pack(fit_idx)
        d["ev"] = pack(eval_idx)
        d["uid"] = u
        data.append(d)

    if not data:
        print("HATA: uygun obje yok (alt gorunum bulunamadi -- sphere20 render lazim)")
        return
    print(f"[{a.tag}] {len(data)} obje | k_fit={a.k_fit} alt_dahil={a.include_bottom} "
          f"| eval={data[0]['ev']['c2w'].shape[0]} alt gorunum | res {a.res}", flush=True)

    triplanes = nn.Parameter(torch.randn(len(data), 3, a.tp_ch, a.tp_res, a.tp_res,
                                         device=DEV) * 0.1)
    nerf = TriplaneNeRF(in_dim=3 * a.tp_ch, hidden=64).to(DEV)
    loss_fn = LRMLoss(use_lpips=True, w_lpips=a.w_lpips).to(DEV)
    opt = torch.optim.Adam([{"params": [triplanes], "lr": a.lr},
                            {"params": nerf.parameters(), "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def render(i, c2w, K, res, bg):
        tp = triplanes[i]
        os_, ds_ = [], []
        for v in range(c2w.shape[0]):
            o, d_ = cameras.rays_from_camera(c2w[v], K[v], res, res)
            os_.append(o); ds_.append(d_)
        o, d_ = torch.cat(os_), torch.cat(ds_)

        def q(pts):
            den, rgb = nerf(sample_triplane(tp, pts, bound=a.bound))
            ins = (pts.abs().amax(-1, keepdim=True) <= a.bound).to(den.dtype)
            return den * ins, rgb

        rgb, acc = volume_render(o, d_, defaults.NEAR, defaults.FAR, a.n_samples, q, bg_color=bg)
        V = c2w.shape[0]
        return (rgb.reshape(V, res, res, 3).permute(0, 3, 1, 2),
                acc.reshape(V, res, res, 1).permute(0, 3, 1, 2))

    @torch.no_grad()
    def eval_all(key):
        """key='ev' -> held-out ALT gorunumler; key=None -> fit gorunumleri."""
        fit_p, ev_p = [], []
        for i, d in enumerate(data):
            src = d["ev"] if key == "ev" else d
            rgb, acc = render(i, src["c2w"], src["K"], a.res, None)
            pred = rgb + (1 - acc)
            gt = src["prem"] + (1 - src["alpha"])
            ev_p.append((pred - gt).pow(2).mean().item())
        m = float(np.mean(ev_p))
        return -10 * math.log10(max(m, 1e-9)), m

    rng2 = np.random.default_rng(0); t0 = time.time(); hist = []
    for step in range(a.steps):
        opt.zero_grad()
        for _ in range(a.batch):
            i = int(rng2.integers(len(data))); d = data[i]
            c = torch.rand(3, device=DEV)
            target = d["prem"] + (1 - d["alpha"]) * c[None, :, None, None]
            rgb, acc = render(i, d["c2w"], d["K"], a.res, c)
            total, _ = loss_fn(rgb, acc, target, d["alpha"])
            (total / a.batch).backward()
        opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            f_psnr = eval_all(None)[0] if step == a.steps - 1 else float("nan")
            e_psnr, e_mse = eval_all("ev")
            hist.append(dict(step=step, fit_psnr=f_psnr, eval_psnr=e_psnr, eval_mse=e_mse))
            print(f"  step {step:5d} fitPSNR={f_psnr:.2f}dB  ALT-heldout={e_psnr:.2f}dB "
                  f"[{(step+1)/(time.time()-t0):.2f} it/s]", flush=True)

    with torch.no_grad():
        imgs = []
        for i, d in enumerate(data[:8]):
            rgb, acc = render(i, d["ev"]["c2w"][:1], d["ev"]["K"][:1], a.res, None)
            rgb = rgb + (1 - acc)
            gt = d["ev"]["prem"][0] + (1 - d["ev"]["alpha"][0])
            imgs.append(torch.cat([gt, rgb[0]], -1))
        Image.fromarray((torch.cat(imgs, 1).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                         * 255).astype(np.uint8)).save(f"{OUT}/{a.tag}.png")
    json.dump(dict(cfg=vars(a), n_obj=len(data), hist=hist),
              open(f"{OUT}/{a.tag}.json", "w", encoding="utf-8"), indent=1)
    print(f"[{a.tag}] BITTI  fitPSNR={hist[-1]['fit_psnr']:.2f}dB  "
          f"ALT-heldout={hist[-1]['eval_psnr']:.2f}dB", flush=True)


if __name__ == "__main__":
    main()
