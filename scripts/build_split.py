"""Nihai train/val/test split'i uretir (CPU).

NEDEN VAR (docs/egitim-oncesi-hazirlik-plani.md, E4 + K12):
1) Mevcut split rastgele: bir obje ile BIREBIR AYNI kopyasi train ve val'e
   dagilirsa val iyimser olur. Olculdu: 38 val objesinin kopyasi train'de.
2) Tarife secimi ve raporlama ayni val setinde yapiliyordu => secim yanliligi.
   Ayri, DOKUNULMAMIS bir test seti gerekiyor.
3) Denetimde bulunan bozuk objeler (tek renk, karanlik) ve GIRDI GORUNUMU BOS
   objeler ayiklanmali.

KOPYA IMZASI -- ONEMLI:
Ilk denemede imza sadece kanonik[0] (on, az 0) uzerinden alinmisti ve 34 "kopya
kumesi" bulmustu. Gozle bakildiginda bunlar KOPYA DEGILDI: plastik kilifli
koleksiyon kartlari gibi DUZ objeler, on ve arka acidan kenardan gorunup ince bir
cizgiye iniyor, o yuzden hepsi ayni cikiyordu. Yan acilarda tamamen farklilar.
=> imza DORT kanonik acinin hepsinden alinir (hepsi tum objelerde ayni kamera).

Kullanim:
    python scripts/build_split.py --out dataset/train_list_v2.json
"""
import argparse
import collections
import hashlib
import io
import json
import os
import random

import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

CANON = ["000.png", "001.png", "002.png", "003.png"]
ALPHA_T = 16


def signature(render_dir, uid, cell=24):
    """DORT kanonik acidan birlesik kaba imza."""
    parts = []
    for v in CANON:
        p = os.path.join(render_dir, uid, v)
        if not os.path.isfile(p):
            return None
        a = np.asarray(Image.open(p).convert("RGBA").resize((cell, cell)))
        parts.append((a[:, :, :3].mean(2) // 32).astype(np.uint8).tobytes())
        parts.append((a[:, :, 3] > ALPHA_T).tobytes())
    return hashlib.md5(b"".join(parts)).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--train_list", default="dataset/train_list_opp_score3.json")
    ap.add_argument("--audit", default="dataset/audit_report.json")
    ap.add_argument("--bad_input", default="dataset/girdi_dejenere_uids.json")
    ap.add_argument("--out", default="dataset/train_list_v2.json")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep_bad", action="store_true",
                    help="denetimde sorunlu bulunanlari ELEME (varsayilan: ele)")
    a = ap.parse_args()

    tl = json.load(io.open(a.train_list, encoding="utf-8"))
    uids = sorted(set(tl.get("train", []) + tl.get("val", []) + tl.get("test", [])))
    print(f"havuz: {len(uids)} obje")

    drop = set()
    if not a.keep_bad:
        if os.path.isfile(a.audit):
            rep = json.load(io.open(a.audit, encoding="utf-8"))
            drop |= set(rep.get("problem_uids", {}))
            print(f"  denetim sorunlusu elendi : {len(rep.get('problem_uids', {}))}")
        if os.path.isfile(a.bad_input):
            bad = set(json.load(io.open(a.bad_input, encoding="utf-8")))
            print(f"  girdi gorunumu bos elendi: {len(bad - drop)}")
            drop |= bad
    uids = [u for u in uids if u not in drop]
    print(f"  kalan: {len(uids)}")

    print("kopya imzasi cikariliyor (4 kanonik aci)...", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        sigs = list(ex.map(lambda u: signature(a.render_dir, u), uids))

    groups = collections.defaultdict(list)
    for u, s in zip(uids, sigs):
        groups[s if s else f"__tek_{u}"].append(u)
    clusters = sorted(groups.values(), key=lambda c: (-len(c), c[0]))
    n_dup = sum(len(c) - 1 for c in clusters if len(c) > 1)
    print(f"  kume: {len(clusters)} | cok-uyeli kume: {sum(1 for c in clusters if len(c) > 1)}"
          f" | fazladan obje: {n_dup}")
    if n_dup:
        print(f"  en buyuk kumeler: {[len(c) for c in clusters[:5]]}")

    # KUME BUTUN atanir: bir kumenin tum uyeleri AYNI tarafa gider
    rng = random.Random(a.seed)
    rng.shuffle(clusters)
    test, val, train = [], [], []
    n_val_target = int(len(uids) * a.val_frac)
    for c in clusters:
        if len(test) < a.n_test:
            test += c
        elif len(val) < n_val_target:
            val += c
        else:
            train += c

    out = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    meta = {"seed": a.seed, "source": a.train_list, "render_dir": a.render_dir,
            "dropped": sorted(drop), "n_clusters": len(clusters),
            "n_duplicate_extra": n_dup,
            "counts": {k: len(v) for k, v in out.items()}}
    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump({**out, "_meta": meta}, f, indent=1)

    s_tr, s_va, s_te = set(train), set(val), set(test)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    # kume sizintisi kalmadi mi
    leak = sum(1 for c in clusters
               if sum(bool(set(c) & s) for s in (s_tr, s_va, s_te)) > 1)
    print(f"\ntrain {len(train)} | val {len(val)} | test {len(test)}")
    print(f"kume bolunmesi (sizinti): {leak}   <- 0 olmali")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
