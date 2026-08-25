"""Render sonrası alpha-tabanlı kalite filtresi (sistem Python + Pillow/numpy).
Kanonik görünümlerin (000-003) alpha kaplamasına bakıp çöp/dejenere objeleri eler.
Çıktı: train_list.json (train/val split) + reject_log.jsonl."""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

CANONICAL = ["000.png", "001.png", "002.png", "003.png"]
# min_cov 0.03 -> 0.010 (2026-08-24): 0.03 esigi eski "sphere" cerceveleme
# icin ayarlanmisti. Yeni "fit" modunda ince/uzun objeler (kilic, yay,
# merdiven, ayakta insan) en kotu izdusumleri kareye oturtuldugu icin diger
# acilardan cizgiye iniyor; 0.03 bunlarin 661 tanesini HAKSIZ eliyordu.
# Gozle dogrulandi: <0.010 gercek coplerin bandi, >=0.010 saglam ince obje.
ALL_VIEWS = [f"{i:03d}.png" for i in range(16)]
ALPHA_THRESH = 16  # 0-255; bunun üstü "obje" sayılır


def alpha_coverage(png_path):
    """Opak piksellerin toplam piksele oranı (0-1)."""
    a = np.asarray(Image.open(png_path).convert("RGBA"))[:, :, 3]
    return float((a > ALPHA_THRESH).mean())


def touches_border(png_path, border_frac=0.4):
    """Obje çerçeve kenarına değiyor mu (taşma/kırpılma işareti)."""
    a = np.asarray(Image.open(png_path).convert("RGBA"))[:, :, 3] > ALPHA_THRESH
    edges = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    return bool(edges.mean() > border_frac)


def passes(render_dir, uid, min_cov=0.010, max_cov=0.90):
    """Obje eğitime uygun mu? (passed, reason) döner.
    Coverage 16 açının hepsinde ölçülür ve MAKSİMUM alınır: obje en az bir
    açıdan belirginse geçer (ince/düz objeler haksız elenmez). Gerçekten boş
    obje her açıda ~0 olduğu için yine elenir."""
    d = os.path.join(render_dir, uid)
    covs = []
    for f in ALL_VIEWS:
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            return False, f"missing:{f}"
        covs.append(alpha_coverage(p))
    max_c = max(covs)
    if max_c < min_cov:
        return False, f"coverage_low:{max_c:.3f}"
    if min(covs) > max_cov:                 # en küçük açı bile taşıyorsa → sürekli kırpılma
        return False, f"coverage_high:{min(covs):.3f}"
    if all(touches_border(os.path.join(d, f)) for f in CANONICAL):
        return False, "clipping"
    return True, "ok"


def filter_dataset(render_dir, uids, out_path, min_cov=0.010, max_cov=0.90,
                   val_frac=0.1, seed=42):
    kept, rejected = [], []
    for uid in uids:
        ok, reason = passes(render_dir, uid, min_cov, max_cov)
        (kept if ok else rejected).append(uid if ok else (uid, reason))
    kept_sorted = sorted(kept)
    rng = random.Random(seed)
    rng.shuffle(kept_sorted)
    n_val = int(len(kept_sorted) * val_frac)
    val, train = kept_sorted[:n_val], kept_sorted[n_val:]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"train": sorted(train), "val": sorted(val)}, f, indent=2)
    reject_log = os.path.join(os.path.dirname(out_path) or ".", "reject_log.jsonl")
    with open(reject_log, "w", encoding="utf-8") as f:
        for uid, reason in rejected:
            f.write(json.dumps({"uid": uid, "reason": reason}) + "\n")
    return kept, rejected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", default="dataset/renders")
    ap.add_argument("--subset", default="dataset/subset.json",
                    help="uid listesi (subset.json ya da meta içeren dizin)")
    ap.add_argument("--out", default="dataset/train_list.json")
    ap.add_argument("--min-cov", type=float, default=0.010)
    ap.add_argument("--max-cov", type=float, default=0.90)
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()
    with open(a.subset, encoding="utf-8") as f:
        uids = list(json.load(f).keys())
    kept, rejected = filter_dataset(a.render_dir, uids, a.out,
                                    a.min_cov, a.max_cov, a.val_frac)
    print(f"gecen: {len(kept)}  elenen: {len(rejected)}  -> {a.out}")
    reasons = {}
    for _, r in rejected:
        key = r.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("elenme nedenleri:", reasons)
