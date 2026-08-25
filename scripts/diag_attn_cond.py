"""Kanit toplama: (1) triplane token'lari image token'lara ne kadar dikkat ediyor?
(2) farkli objeler icin cikti ne kadar degisiyor (conditioning gucu)?"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from lrm.dataset import LRMDataset
from lrm.model import LRM

CKPT = "dataset/lrm_ckpts/last.pt"
DEV = "cuda"


@torch.no_grad()
def attn_mass(model, it):
    """Her blokta: triplane sorgularinin image token'lara giden attention kutlesi."""
    to = lambda t: t.to(DEV)
    tok = model.encoder(to(it["input_imgs"]))
    Vi, P, _ = tok.shape
    side = int(P ** 0.5)
    from lrm import cameras
    pls = []
    for i in range(Vi):
        Kp = cameras.scale_intrinsics(to(it["input_K"])[i], 224, side)
        pls.append(cameras.plucker_map(to(it["input_c2w"])[i], Kp, side, side).reshape(-1, 6))
    plucker = torch.stack(pls).reshape(Vi * P, 6)
    tr = model.transformer
    x = tr.img_proj(torch.cat([tok.reshape(Vi * P, -1), plucker], -1))
    x = torch.cat([x, tr.tp_tokens], 0)[None]
    M = Vi * P
    out = []
    for bi, blk in enumerate(tr.blocks):
        h = blk.n1(x)
        a, w = blk.attn(h, h, h, need_weights=True, average_attn_weights=True)
        # tp sorgularinin (son n_tp) image key'lerine (ilk M) giden kutle
        mass = w[0, M:, :M].sum(-1).mean().item()
        out.append(mass)
        x = x + a
        x = x + blk.mlp(blk.n2(x))
    return out, M, tr.n_tp


@torch.no_grad()
def cross_object_variance(model, ds, uids, res=64):
    """Ayni kameradan N farkli obje render et; pikselde objeler arasi std."""
    to = lambda t: t.to(DEV)
    it0 = ds[ds.uids.index(uids[0])]
    c2w, K = to(it0["sup_c2w"][:1]), to(it0["sup_K"][:1])
    preds, gts, tps = [], [], []
    for u in uids:
        it = ds[ds.uids.index(u)]
        tp = model.make_triplane(to(it["input_imgs"]), to(it["input_c2w"]), to(it["input_K"]))
        rgb, acc = model.render_view(tp, c2w[0], K[0], res, res)
        preds.append(rgb.cpu())
        tps.append(tp.cpu())
        gts.append(torch.nn.functional.interpolate(it["sup_rgb"][:1], size=(res, res))[0])
    P = torch.stack(preds); G = torch.stack(gts); T = torch.stack(tps)
    print(f"  pred: piksel-arasi-obje std={P.std(0).mean():.4f}  genel std={P.std():.4f} mean={P.mean():.4f}")
    print(f"  GT  : piksel-arasi-obje std={G.std(0).mean():.4f}  genel std={G.std():.4f} mean={G.mean():.4f}")
    print(f"  triplane: objeler-arasi std={T.std(0).mean():.4f}  kanal std={T.std():.4f} |mean|={T.mean().abs():.4f}")
    print(f"  ORAN pred/GT conditioning = {float(P.std(0).mean()/G.std(0).mean()):.3f}")


def main():
    ds = LRMDataset("dataset/train_list.json", "dataset/renders", split="train",
                    render_res=128, n_sup=4, augment=False)
    uids = ds.uids[:8]
    for tag, ck in [("FRESH", None), ("TRAINED", CKPT)]:
        torch.manual_seed(0)
        m = LRM(n_samples=48).to(DEV).eval()
        if ck:
            st = torch.load(ck, map_location="cpu")
            m.load_state_dict(st["model"]); print(f"\n=== {tag} (step {st['step']}) ===")
        else:
            print(f"\n=== {tag} ===")
        it = ds[ds.uids.index(uids[0])]
        mass, M, ntp = attn_mass(m, it)
        print(f"  token: {M} image + {ntp} triplane | uniform beklenti = {M/(M+ntp):.4f}")
        print("  blok basi tp->image attention kutlesi:")
        print("   " + " ".join(f"{v:.4f}" for v in mass))
        cross_object_variance(m, ds, uids)


if __name__ == "__main__":
    main()
