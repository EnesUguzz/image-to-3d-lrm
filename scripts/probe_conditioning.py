"""Phase 1 kanit toplama: model ciktisi GIRDIYE bagli mi?
Iki farkli objenin girdisiyle triplane + render ciktisini karsilastir.
- Egitimsiz model bile ayni ciktiyi veriyorsa => mimari kopukluk (bug).
- Egitimsizde farkli ama egitilmiste ayni => egitim cokmesi (collapse)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from lrm.dataset import LRMDataset
from lrm.model import LRM


def diff_stats(a, b):
    d = (a - b).abs()
    return float(d.mean()), float(d.max()), float(a.std()), float(b.std())


@torch.no_grad()
def probe(model, itA, itB, device, tag):
    to = lambda t: t.to(device)
    tpA = model.make_triplane(to(itA["input_imgs"]), to(itA["input_c2w"]), to(itA["input_K"]))
    tpB = model.make_triplane(to(itB["input_imgs"]), to(itB["input_c2w"]), to(itB["input_K"]))
    md, mx, sA, sB = diff_stats(tpA, tpB)
    print(f"[{tag}] triplane |A-B| mean={md:.5f} max={mx:.4f} | std_A={sA:.4f} std_B={sB:.4f}")
    # ayni kameradan iki objeyi render et, RGB farki
    c2w = to(itA["sup_c2w"][:1]); K = to(itA["sup_K"][:1])
    rgbA, accA = model.render_view(tpA, c2w[0], K[0], 64, 64)
    rgbB, accB = model.render_view(tpB, c2w[0], K[0], 64, 64)
    rmd, rmx, _, _ = diff_stats(rgbA, rgbB)
    amd, amx, _, _ = diff_stats(accA, accB)
    print(f"[{tag}] render RGB |A-B| mean={rmd:.5f} max={rmx:.4f} | acc |A-B| mean={amd:.5f} "
          f"| rgbA.mean={float(rgbA.mean()):.3f} rgbB.mean={float(rgbB.mean()):.3f} "
          f"accA.mean={float(accA.mean()):.3f} accB.mean={float(accB.mean()):.3f}")


def main():
    device = "cuda"
    ds = LRMDataset("dataset/train_list.json", "dataset/renders", split="train",
                    render_res=64, n_sup=4, augment=False)
    # iki gorsel olarak cok farkli obje (overfit testindekiler)
    itA = ds[ds.uids.index("056e0c89") if "056e0c89" in ds.uids else 0]  # yesil kutu
    itB = ds[ds.uids.index("04ef6175") if "04ef6175" in ds.uids else 1]  # beyaz yazici
    print(f"A={itA['uid'][:8]}  B={itB['uid'][:8]}")

    print("\n=== EGITIMSIZ (taze init) ===")
    torch.manual_seed(0)
    m_fresh = LRM(n_samples=48).to(device).eval()
    probe(m_fresh, itA, itB, device, "fresh")

    ckpt = "dataset/lrm_ckpts/last.pt"
    if os.path.isfile(ckpt):
        print("\n=== EGITILMIS (last.pt step 6500) ===")
        m_tr = LRM(n_samples=48).to(device).eval()
        m_tr.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        probe(m_tr, itA, itB, device, "trained")


if __name__ == "__main__":
    main()
