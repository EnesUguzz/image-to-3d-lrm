"""glb kökünden N obje seçip subset.json yazar (sistem Python)."""
import argparse
import glob
import json
import os
import random


def find_glbs(root):
    out = []
    for path in glob.glob(os.path.join(root, "**", "*.glb"), recursive=True):
        uid = os.path.splitext(os.path.basename(path))[0]
        out.append((uid, os.path.abspath(path)))
    return out


def load_curated_uids(cache_path="dataset/lvis_uids.json"):
    """LVIS kürate uid setini döndürür. Cache varsa okur; yoksa objaverse'ten
    çekip cache'ler (tek sefer indirir). objaverse yalnızca cache yokken gerekir."""
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return set(json.load(f))
    import objaverse
    lvis = objaverse.load_lvis_annotations()
    uids = set()
    for v in lvis.values():
        uids.update(v)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sorted(uids), f)
    return uids


def select_subset(pairs, n, seed=42, curated_uids=None):
    if curated_uids is not None:
        pairs = [p for p in pairs if p[0] in curated_uids]
    pairs = sorted(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def write_subset(pairs, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({u: p for u, p in pairs}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="dataset/subset.json")
    ap.add_argument("--curated", action="store_true",
                    help="Sadece LVIS kürate objeleri seç")
    ap.add_argument("--curated-cache", default="dataset/lvis_uids.json")
    a = ap.parse_args()
    curated = load_curated_uids(a.curated_cache) if a.curated else None
    if curated is not None:
        print(f"LVIS kürate uid: {len(curated)}")
    pairs = select_subset(find_glbs(a.root), a.n, curated_uids=curated)
    write_subset(pairs, a.out)
    print(f"subset yazildi: {len(pairs)} obje -> {a.out}")
