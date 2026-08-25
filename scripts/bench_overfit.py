"""Standart hizli tezgah: N objeyi EZBERLEYEBILIYOR MU?
Sistem dogruysa kucuk N'de train PSNR yuksek + retrieval %100 olmali.
Tek degisken degistirip A/B karsilastirmak icin: --tag ile ayri metrik dosyasi."""
import argparse, json, os, sys, time, math
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras
from lrm.model import LRM
from lrm.losses import LRMLoss

DEV = "cuda"
IN_VIEWS = [0]
SUP_VIEWS = [4, 7, 10, 13]
# YENI GORUNUM: ne girdi ne supervision olarak kullanilir. Ezber ile gercek 3B
# rekonstruksiyonu ayirir -- egitimde gorulen acidan render etmek yaniltici olabilir.
NOVEL_VIEW = 8
OUT = "dataset/lrm_bench"
RENDER_DIR = "dataset/renders"   # --render_dir ile degistirilir


def load_view(uid, view, res, crop=1.0):
    """crop<1: kameralar hep orijine baktigi icin obje daima merkezde => merkezi
    crop oraninda kirp ve ayni cozunurlukte render et. Ayni maliyete ~1/crop^2 kat
    daha fazla obje pikseli. Intrinsic'te bu, odak uzakligini 1/crop ile carpmaktir."""
    meta = json.load(open(f"{RENDER_DIR}/{uid}/meta.json", encoding="utf-8"))
    v = meta["views"][view]
    im = Image.open(f"{RENDER_DIR}/{uid}/{v['file']}").convert("RGBA")
    a = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.
    if crop < 1.0:
        H = a.shape[-1]
        m = int(round(H * (1 - crop) / 2))
        a = a[..., m:H - m, m:H - m]
    a = F.interpolate(a[None], size=(res, res), mode="bilinear", align_corners=False)[0]
    K = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                 meta.get("resolution", 512), res)
    if crop < 1.0:
        K[0, 0] /= crop
        K[1, 1] /= crop
    c2w = torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float32))
    return a, c2w, K


def sample_fg_crop(d, res, rng, smin=0.45, smax=1.0, jitter=0.4):
    """TripoSR tarzi on-plana yanli rastgele kirpma.

    Referans pipeline'lar 512 master'i saklayip egitimde KIRPARAK denetler:
    LRM rastgele 128x128 bolge, InstantMesh 192x192 patch, TripoSR ise 'on plani
    kapsayan kirpmalarin secilme olasiligini artirarak' ilgi alanina agirlik verir.
    Bizim objelerimiz kareyi ortalama %9 kapladigi icin kaybin ~%85'i arka plana
    gidiyordu; kirpma bunu kaynaginda azaltir. Kameralar hep orijine baktigi icin
    obje merkezdedir => merkeze yakin kirpma dogal olarak on-plana yanlidir.

    Dondurur: (target_premult, target_alpha, K') -- K' kirpmaya gore olceklenmis.
    """
    prem_hi, alpha_hi, K_hi = d["prem_hi"], d["alpha_hi"], d["sk_hi"]
    HI = prem_hi.shape[-1]
    sc = rng.uniform(smin, smax)
    w = max(8, int(round(HI * sc)))
    base = (HI - w) // 2
    span = int(round(jitter * base))
    ox = base + (rng.integers(-span, span + 1) if span > 0 else 0)
    oy = base + (rng.integers(-span, span + 1) if span > 0 else 0)
    ox = int(min(max(ox, 0), HI - w)); oy = int(min(max(oy, 0), HI - w))
    pr = prem_hi[..., oy:oy + w, ox:ox + w]
    al = alpha_hi[..., oy:oy + w, ox:ox + w]
    pr = F.interpolate(pr, size=(res, res), mode="bilinear", align_corners=False)
    al = F.interpolate(al, size=(res, res), mode="bilinear", align_corners=False)
    f = res / w
    K = K_hi.clone()
    K[:, 0, 0] *= f; K[:, 1, 1] *= f
    K[:, 0, 2] = (K_hi[:, 0, 2] - ox) * f
    K[:, 1, 2] = (K_hi[:, 1, 2] - oy) * f
    return pr, al, K


def build(uids, res, crop=1.0, hi_res=0):
    data = []
    for u in uids:
        ii, ic, ik = [], [], []
        for v in IN_VIEWS:
            a, c, k = load_view(u, v, 224)
            ii.append(a[:3] * a[3:4] + (1 - a[3:4])); ic.append(c); ik.append(k)
        si, sc, sk, sa = [], [], [], []
        for v in SUP_VIEWS:
            a, c, k = load_view(u, v, res, crop)
            si.append(a[:3] * a[3:4]); sa.append(a[3:4]); sc.append(c); sk.append(k)
        na, nc, nk = load_view(u, NOVEL_VIEW, res, crop)
        rec = dict(uid=u, ii=torch.stack(ii).to(DEV), ic=torch.stack(ic).to(DEV),
                   ik=torch.stack(ik).to(DEV), prem=torch.stack(si).to(DEV),
                   alpha=torch.stack(sa).to(DEV), sc=torch.stack(sc).to(DEV),
                   sk=torch.stack(sk).to(DEV),
                   n_gt=(na[:3] * na[3:4] + (1 - na[3:4])).to(DEV),
                   n_c2w=nc[None].to(DEV), n_K=nk[None].to(DEV))
        if hi_res:   # rastgele kirpma icin yuksek cozunurluklu hedef sakla
            hp, ha, hk = [], [], []
            for v in SUP_VIEWS:
                a, c, k = load_view(u, v, hi_res, crop)
                hp.append(a[:3] * a[3:4]); ha.append(a[3:4]); hk.append(k)
            rec["prem_hi"] = torch.stack(hp).to(DEV)
            rec["alpha_hi"] = torch.stack(ha).to(DEV)
            rec["sk_hi"] = torch.stack(hk).to(DEV)
        data.append(rec)
    return data


@torch.no_grad()
def evaluate(model, data, res, novel=False):
    """novel=True: egitimde HIC kullanilmamis bir kameradan render et (NOVEL_VIEW).
    Ezber ile gercek 3B rekonstruksiyonu ayirir."""
    model.eval()
    preds, gts = [], []
    for d in data:
        if novel:
            rgb, _ = model(d["ii"], d["ic"], d["ik"], d["n_c2w"], d["n_K"], (res, res))
            preds.append(rgb[0]); gts.append(d["n_gt"])
        else:
            rgb, _ = model(d["ii"], d["ic"], d["ik"], d["sc"][:1], d["sk"][:1], (res, res))
            preds.append(rgb[0]); gts.append(d["prem"][0] + (1 - d["alpha"][0]))
    P, G = torch.stack(preds), torch.stack(gts)
    mse = ((P - G) ** 2).mean((1, 2, 3))
    psnr = (-10 * torch.log10(mse.clamp_min(1e-9))).mean().item()
    D = ((P[:, None] - G[None]) ** 2).mean((2, 3, 4))
    top1 = (D.argmin(1) == torch.arange(len(data), device=DEV)).float().mean().item()
    mean_mse = ((P.mean(0, keepdim=True) - G) ** 2).mean().item()
    model.train()
    return psnr, top1, mse.mean().item(), mean_mse, P, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obj", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--w_lpips", type=float, default=2.0)
    ap.add_argument("--lpips_start", type=float, default=0.0,
                    help="LPIPS'i egitimin bu orani gectikten SONRA devreye sok "
                         "(0=bastan acik). Olculdu: LPIPS bastan acikken model "
                         "'ortalama obje' havzasinda kaliyor (top1 %25 vs %59).")
    ap.add_argument("--fg_weight", type=float, default=5.0,
                    help="obje piksellerinin agirligi (kaplama %9 oldugu icin kritik)")
    ap.add_argument("--w_tv", type=float, default=5e-4)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--arch", default="joint", choices=["joint", "cross"])
    ap.add_argument("--train_encoder", action="store_true",
                    help="DINOv2'nin TAMAMINI egit")
    ap.add_argument("--unfreeze_last", type=int, default=0,
                    help="DINOv2'nin son N blogunu egit (LRM encoder'i egitir; biz "
                         "kucuk veride kismi cozmeyi deniyoruz)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--init_from", default="",
                    help="distile edilmis agirliklardan basla (hedef-vs-optimizasyon testi)")
    ap.add_argument("--init_nerf", default="",
                    help="oracle NeRF agirliklarini da yukle (_tp.pt)")
    ap.add_argument("--crop", type=float, default=1.0,
                    help="supervision hedefinde merkez-kirpma orani (0.6 => ~2.8x obje pikseli)")
    ap.add_argument("--uids", default="", help="virgulle ayrilmis uid listesi (kontrol koşulari icin)")
    ap.add_argument("--render_dir", default="dataset/renders",
                    help="render kok dizini (yeni set: dataset/renders_opp_score3)")
    ap.add_argument("--train_list", default="dataset/train_list.json",
                    help="uid listesi json (train/val anahtarli)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)

    global RENDER_DIR
    RENDER_DIR = a.render_dir

    all_uids = json.load(open(a.train_list, encoding="utf-8"))["train"]
    if a.uids:
        uids = [u for u in a.uids.split(",") if u]
    else:
        step_u = max(1, len(all_uids) // a.n_obj)
        uids = all_uids[::step_u][:a.n_obj]
    data = build(uids, a.res, a.crop)

    kw = {}
    if a.arch == "cross":
        kw["cross_attn"] = True
    model = LRM(n_samples=a.n_samples, **kw).to(DEV)
    if a.init_from:
        model.load_state_dict(torch.load(a.init_from, map_location="cpu")["model"])
        print(f"  baslangic agirliklari: {a.init_from}", flush=True)
    if a.init_nerf:
        model.nerf.load_state_dict(torch.load(a.init_nerf, map_location="cpu")["nerf"])
        print(f"  oracle NeRF yuklendi: {a.init_nerf}", flush=True)
    if a.train_encoder:
        for p in model.encoder.model.parameters():
            p.requires_grad_(True)
    elif a.unfreeze_last > 0:
        for blk in model.encoder.model.blocks[-a.unfreeze_last:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in model.encoder.model.norm.parameters():
            p.requires_grad_(True)
    if a.train_encoder or a.unfreeze_last > 0:
        model.encoder.forward = model.encoder.forward.__wrapped__.__get__(model.encoder)
    params = [p for p in model.parameters() if p.requires_grad]
    nparam = sum(p.numel() for p in params) / 1e6
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    loss_fn = LRMLoss(use_lpips=True, w_lpips=a.w_lpips, fg_weight=a.fg_weight).to(DEV)
    rng = np.random.default_rng(0)
    hist, t0 = [], time.time()
    print(f"[{a.tag}] arch={a.arch} n_obj={len(uids)} params={nparam:.1f}M "
          f"batch={a.batch} steps={a.steps} res={a.res} crop={a.crop} "
          f"w_lpips={a.w_lpips}@{a.lpips_start} fg={a.fg_weight} "
          f"enc_train={a.train_encoder or a.unfreeze_last}", flush=True)
    model.train()
    for step in range(a.steps):
        opt.zero_grad()
        agg = 0.0
        for _ in range(a.batch):
            d = data[int(rng.integers(len(data)))]
            c = torch.rand(3, device=DEV)
            target = d["prem"] + (1 - d["alpha"]) * c[None, :, None, None]
            rgb, acc = model(d["ii"], d["ic"], d["ik"], d["sc"], d["sk"],
                             (a.res, a.res), bg_color=c)
            loss_fn.w_lpips = (0.0 if step < a.lpips_start * a.steps else a.w_lpips)
            total, parts = loss_fn(rgb, acc, target, d["alpha"])
            if a.w_tv > 0:
                total = total + a.w_tv * model.tv_loss(model._last_triplane)
            (total / a.batch).backward()
            agg += float(total) / a.batch
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            psnr, top1, mse, mmse, P, G = evaluate(model, data, a.res)
            hist.append(dict(step=step, loss=agg, psnr=psnr, top1=top1, mse=mse, mean_mse=mmse))
            print(f"  step {step:5d} loss={agg:.4f} trainPSNR={psnr:.2f}dB top1={top1:.0%} "
                  f"mse={mse:.4f} ortalama-baseline={mmse:.4f} "
                  f"[{(step+1)/(time.time()-t0):.2f} it/s]", flush=True)
    psnr, top1, mse, mmse, P, G = evaluate(model, data, a.res)
    n = min(8, len(data))
    grid = torch.cat([torch.cat([G[i], P[i]], -1) for i in range(n)], 1)
    Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    ).save(f"{OUT}/{a.tag}.png")
    json.dump(dict(cfg=vars(a), params_M=nparam, hist=hist,
                   final=dict(psnr=psnr, top1=top1, mse=mse, mean_mse=mmse)),
              open(f"{OUT}/{a.tag}.json", "w"), indent=1)
    print(f"[{a.tag}] BITTI trainPSNR={psnr:.2f}dB top1={top1:.0%} "
          f"mse={mse:.4f} vs ortalama-baseline={mmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
