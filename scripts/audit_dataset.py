"""Tam veri denetim raporu (CPU, GPU gerektirmez).

NEDEN VAR (docs/egitim-oncesi-hazirlik-plani.md, E3):
Kenar tasma ve alpha kaplama bugune kadar sadece 100 objede olculdu, 14.597'de
degil. Kopya obje, bos render, kamera tutarliligi ve gercek obje yaricapi hic
taranmadi. Uzun egitime girmeden once bunlarin TUM veride bilinmesi gerekiyor:
  - kopya kumeleri train/val'e bolunurse VAL SIZINTISI olur
  - `bound=0.6` kalibrasyonu gercek yaricapa dayanmali (hacmin %34'u kullaniliyor)
  - tek bir bozuk meta.json egitimi saatler sonra dusurur

Cikti: dataset/audit_report.json  +  dataset/audit_report.md

Kullanim:
    python scripts/audit_dataset.py --render_dir dataset/renders_opp_score3 \
        --train_list dataset/train_list_opp_score3.json --workers 8
    python scripts/audit_dataset.py --limit 500        # hizli on-bakis
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALPHA_T = 16          # 0-255; ustu "obje"
EMPTY_COV = 0.005     # bunun altinda kaplama = bos render
BORDER_FRAC = 0.02    # kenar pikselinin bu kadari doluysa tasma
DARK_MEAN = 0.05      # obje pikselleri bu kadar karanliksa suphel
FLAT_STD = 0.02       # obje rengi bu kadar tek duzeyse suphel


def _sig(a_small, alpha_small):
    """Kaba gorsel imza (kopya tespiti icin)."""
    q = (a_small // 32).astype(np.uint8).tobytes() + (alpha_small > ALPHA_T).tobytes()
    return hashlib.md5(q).hexdigest()


def audit_one(render_dir, uid, expect_views=None):
    d = os.path.join(render_dir, uid)
    out = {"uid": uid, "problems": []}
    mp = os.path.join(d, "meta.json")
    if not os.path.isfile(mp):
        out["problems"].append("meta_yok")
        return out
    try:
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        out["problems"].append(f"meta_bozuk:{type(e).__name__}")
        return out

    views = meta.get("views", [])
    out["n_views"] = len(views)
    if expect_views and len(views) != expect_views:
        out["problems"].append(f"gorunum_sayisi:{len(views)}")
    if not meta.get("canonical_indices"):
        out["problems"].append("kanonik_yok")

    master = meta.get("resolution", 512)
    radius = float(meta.get("camera", {}).get("radius", 0.0))
    out["radius"] = radius

    # --- kamera tutarliligi
    Ks, cam_norms = [], []
    for v in views:
        try:
            ext = np.array(v["extrinsic"], dtype=np.float64)
            c2w = np.linalg.inv(ext)
            cam_norms.append(float(np.linalg.norm(c2w[:3, 3])))
            Ks.append(np.array(v["intrinsic"], dtype=np.float64))
        except Exception:
            out["problems"].append("kamera_bozuk")
            break
    if cam_norms:
        out["cam_r_min"], out["cam_r_max"] = min(cam_norms), max(cam_norms)
        if radius and max(abs(c - radius) for c in cam_norms) > 1e-3:
            out["problems"].append("kamera_yaricapi_tutarsiz")
    if Ks and any(not np.allclose(K, Ks[0]) for K in Ks):
        out["problems"].append("intrinsic_tutarsiz")

    # --- goruntuler
    covs, borders, obj_means, obj_stds = [], [], [], []
    half_extent = 0.0
    fx = float(Ks[0][0, 0]) if Ks else 0.0
    sig = None
    for j, v in enumerate(views):
        p = os.path.join(d, v.get("file", f"{j:03d}.png"))
        if not os.path.isfile(p):
            out["problems"].append(f"png_yok:{os.path.basename(p)}")
            continue
        try:
            im = Image.open(p)
            if im.size != (master, master):
                out["problems"].append(f"cozunurluk:{im.size[0]}")
            arr = np.asarray(im.convert("RGBA"))
        except Exception as e:
            out["problems"].append(f"png_bozuk:{type(e).__name__}")
            continue
        al = arr[:, :, 3]
        m = al > ALPHA_T
        cov = float(m.mean())
        covs.append(cov)
        edges = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
        borders.append(float(edges.mean()))
        if m.any():
            rgb = arr[:, :, :3][m].astype(np.float32) / 255.0
            obj_means.append(float(rgb.mean()))
            obj_stds.append(float(rgb.std()))
            # OBJE YARICAPI: silüetin merkezden en uzak pikseli -> aci -> yaricap
            ys, xs = np.nonzero(m)
            c = master / 2.0
            rpix = float(np.max(np.hypot(xs - c, ys - c)))
            if fx > 0 and radius > 0:
                ang = math.atan(rpix / fx)
                half_extent = max(half_extent, radius * math.sin(ang))
        if j == 0:
            small = np.asarray(Image.fromarray(arr, "RGBA").resize((16, 16)))
            sig = _sig(small[:, :, :3].mean(2).astype(np.uint8), small[:, :, 3])

    if covs:
        out["cov_min"], out["cov_med"], out["cov_max"] = (
            float(np.min(covs)), float(np.median(covs)), float(np.max(covs)))
        if max(covs) < EMPTY_COV:
            out["problems"].append("bos_render")
        n_over = sum(1 for b in borders if b > BORDER_FRAC)
        out["tasma_gorunum"] = n_over
        if n_over:
            out["problems"].append(f"kenar_tasmasi:{n_over}")
    if obj_means:
        out["obj_mean"] = float(np.mean(obj_means))
        out["obj_std"] = float(np.mean(obj_stds))
        if out["obj_mean"] < DARK_MEAN:
            out["problems"].append("karanlik")
        if out["obj_std"] < FLAT_STD:
            out["problems"].append("tek_renk")
    out["half_extent"] = half_extent      # >= gercek obje yaricapi (silüet ust siniri)
    out["sig"] = sig
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--train_list", default="dataset/train_list_opp_score3.json")
    ap.add_argument("--out", default="dataset/audit_report")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="hizli on-bakis icin ilk N obje")
    ap.add_argument("--expect_views", type=int, default=16)
    a = ap.parse_args()

    with open(a.train_list, encoding="utf-8") as f:
        tl = json.load(f)
    uids = sorted(set(tl.get("train", []) + tl.get("val", []) + tl.get("test", [])))
    if a.limit:
        uids = uids[:a.limit]
    print(f"denetlenecek: {len(uids)} obje  ({a.render_dir})", flush=True)

    res, done = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda u: audit_one(a.render_dir, u, a.expect_views), uids):
            res.append(r)
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(uids)}", flush=True)

    # ---- ozet
    probs = collections.Counter()
    for r in res:
        for p in r.get("problems", []):
            probs[p.split(":")[0]] += 1
    bad = [r["uid"] for r in res if r.get("problems")]

    covs = [r["cov_med"] for r in res if "cov_med" in r]
    hx = [r["half_extent"] for r in res if r.get("half_extent")]
    sigs = collections.defaultdict(list)
    for r in res:
        if r.get("sig"):
            sigs[r["sig"]].append(r["uid"])
    dup_clusters = {k: v for k, v in sigs.items() if len(v) > 1}
    n_dup = sum(len(v) - 1 for v in dup_clusters.values())

    summary = {
        "n": len(res),
        "sorunlu": len(bad),
        "sorun_dagilimi": dict(probs),
        "kaplama_medyan": float(np.median(covs)) if covs else None,
        "kaplama_p05": float(np.percentile(covs, 5)) if covs else None,
        "kaplama_p95": float(np.percentile(covs, 95)) if covs else None,
        "yaricap_maks": float(np.max(hx)) if hx else None,
        "yaricap_p99": float(np.percentile(hx, 99)) if hx else None,
        "yaricap_medyan": float(np.median(hx)) if hx else None,
        "kopya_kume": len(dup_clusters),
        "kopya_obje": n_dup,
    }
    if hx:
        # bound onerisi: p99.9 yaricap + %5 pay (triplane hacmi bosa gitmesin)
        summary["onerilen_bound"] = round(float(np.percentile(hx, 99.9)) * 1.05, 3)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "duplicate_clusters": dup_clusters,
                   "problem_uids": {r["uid"]: r["problems"] for r in res if r.get("problems")},
                   "per_object": res}, f, indent=1)

    lines = ["# Veri denetim raporu", "",
             f"- obje: **{summary['n']}**, sorunlu: **{summary['sorunlu']}**",
             f"- kaplama medyan {summary['kaplama_medyan']:.4f} "
             f"(p05 {summary['kaplama_p05']:.4f}, p95 {summary['kaplama_p95']:.4f})",
             f"- obje yaricapi: medyan {summary['yaricap_medyan']:.3f}, "
             f"p99 {summary['yaricap_p99']:.3f}, maks {summary['yaricap_maks']:.3f}",
             f"- **onerilen bound: {summary.get('onerilen_bound')}** (su an 0.6)",
             f"- kopya kume: {summary['kopya_kume']}, fazladan obje: {summary['kopya_obje']}",
             "", "## Sorun dagilimi", ""]
    for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{k}`: {v}")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n-> {a.out}.json / {a.out}.md")


if __name__ == "__main__":
    main()
