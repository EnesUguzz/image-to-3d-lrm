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


def select_subset(pairs, n, seed=42):
    pairs = sorted(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def write_subset(pairs, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({u: p for u, p in pairs}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="dataset/subset.json")
    a = ap.parse_args()
    pairs = select_subset(find_glbs(a.root), a.n)
    write_subset(pairs, a.out)
    print(f"subset yazildi: {len(pairs)} obje -> {a.out}")
