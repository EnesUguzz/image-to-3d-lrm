"""KATMAN 3: GEOMETRI metrikleri -- tahmin mesh'i vs GERCEK glb.

NEDEN VAR (2026-08-27):
Katman 1 (PSNR/SSIM/LPIPS/CLIP/IoU) render uzayinda olcer. Bir model gorunuse
dogru ama GEOMETRIYE yanlis olabilir: NeRF bakis yonu almadigi icin "kartondan
kesme" tarzi ince bir kabuk da yuksek PSNR verebilir. Faz C mesh cikartacagi
icin geometrinin ayrica olculmesi gerekiyor.

METRIK SECIMI (arastirma, 2026-08-27):
- **F-score@d (BIRINCIL)**. Tatarchenko ve ark. (CVPR 2019) Chamfer ve IoU yerine
  bunu savunuyor: dogrudan yorumlanabilir ("yuzeyin yuzde kaci dogru kondu") ve
  aykiri degerlere dayanikli. TripoSR de bas tablosunda F-score raporluyor.
- Chamfer (IKINCIL). Aykiri degere asiri duyarli -- tek bir sacma ucgen sayiyi
  ucuruyor. Yine de referanslarla karsilastirilabilirlik icin raporlanir.
- Normal Consistency. Yuzey purruzlulugu/yonelimi.
- Volume IoU (TEMKINLI). Ici dolu hacimlerde ic bolge sayiyi sisirir; tek basina
  guvenilmez, digerleriyle birlikte okunur.

HIZALAMA -- BIZDE SERBEST, BU BIR AVANTAJ:
TripoSR rotasyon taramasi + ICP yapmak zorunda, Meshy render karsilastirmasini
"kamera hatasi geometri hatasina karisir" diye reddediyor. Bizde kameralar BILINIYOR
ve tahmin kanonik cercevede uretiliyor => hizalama gerekmiyor.
Bunun kosulu, GT mesh'in render ile AYNI normalizasyondan gecmesi:
`export_norm_mesh.py` bunu Blender'da ayni fonksiyonlari cagirarak yapar.

GECERLILIK KAPISI:
Normalize GT mesh'in siluetinin GT render alpha'siyla ortusmesi olculur.
`gt_siluet_iou < --min_gt_iou` (vars. 0.90) ise o obje ATLANIR ve raporda
sayilir -- yanlis hizalanmis bir GT'ye karsi olculen F-score copturur.

Kullanim:
    python scripts/eval_geometry.py --ckpt dataset/lrm_ckpts/distilled_v2.pt \
        --teacher dataset/lrm_ckpts/teacher_1024_tv.pt --n_obj 32 --tag blok2
"""
import argparse
import gzip
import io
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm import cameras, defaults, runstamp
from lrm.model import LRM
from extract_mesh import density_grid, to_mesh, clean, silhouette_iou, report

DEV = "cuda"
BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
GLB_ROOT = os.path.expanduser(r"~\.objaverse\hf-objaverse-v1")
PATHS_GZ = os.path.join(GLB_ROOT, "object-paths.json.gz")
# F-score esikleri: normalize cerceve kenari 1.0 (bbox en uzun kenar = 2*0.5).
# Tatarchenko "hacim kenarinin %1'i" diyor => 0.01. TripoSR 0.1/0.2/0.5 kullanir
# ama onun olcegi farkli; ikisini de raporlayalim.
FS_TAU = (0.01, 0.02, 0.05)


# ------------------------------------------------------------------ GT mesh
def uid_to_glb(uid, table):
    rel = table.get(uid)
    return os.path.join(GLB_ROOT, rel.replace("/", os.sep)) if rel else None


def gt_mesh(uid, table, cache_dir, resolution=512, force=False):
    """Render ile AYNI normalizasyondan gecmis GT mesh (Blender'da uretilir, cache'lenir)."""
    import trimesh
    out = os.path.join(cache_dir, f"{uid}.ply")
    if force or not os.path.isfile(out):
        glb = uid_to_glb(uid, table)
        if not glb or not os.path.isfile(glb):
            return None, "glb_yok"
        os.makedirs(cache_dir, exist_ok=True)
        r = subprocess.run(
            [BLENDER, "-b", "-P", os.path.join(os.path.dirname(__file__),
                                               "export_norm_mesh.py"),
             "--", "--object_path", glb, "--uid", uid, "--out", out,
             "--resolution", str(resolution)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=300)
        if not os.path.isfile(out):
            tail = (r.stdout or "")[-300:]
            return None, f"export_hata: {tail.strip()[-120:]}"
    try:
        m = trimesh.load(out, process=False, force="mesh")
    except Exception as e:
        return None, f"ply_okunamadi:{type(e).__name__}"
    if m.faces.shape[0] == 0:
        return None, "bos_mesh"
    return m, ""


# ------------------------------------------------------------------ metrikler
def sample_surface(mesh, n=10000, seed=0):
    """Yuzeyden n nokta + o noktalardaki normaller."""
    rng = np.random.default_rng(seed)
    pts, fid = mesh.sample(n, return_index=True)
    nrm = mesh.face_normals[fid]
    return np.asarray(pts, np.float64), np.asarray(nrm, np.float64)


def _nn(a, b):
    """a'daki her nokta icin b'deki en yakin komsunun (mesafe, indeks)'i."""
    from scipy.spatial import cKDTree
    d, i = cKDTree(b).query(a, k=1, workers=-1)
    return np.asarray(d), np.asarray(i)


def geom_metrics(P, Pn, G, Gn, taus=FS_TAU):
    """P=tahmin noktalari, G=GT noktalari (+normalleri). Cift yonlu."""
    d_pg, i_pg = _nn(P, G)      # tahmin -> GT  (precision / accuracy)
    d_gp, i_gp = _nn(G, P)      # GT -> tahmin  (recall / completeness)
    out = {
        "chamfer_L1": float(d_pg.mean() + d_gp.mean()),
        "chamfer_L2": float((d_pg ** 2).mean() + (d_gp ** 2).mean()),
        "dogruluk_med": float(np.median(d_pg)),      # aykiri degere dayanikli
        "tamlik_med": float(np.median(d_gp)),
    }
    for t in taus:
        pr = float((d_pg < t).mean())
        rc = float((d_gp < t).mean())
        out[f"fscore@{t}"] = float(2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0
        out[f"precision@{t}"] = pr
        out[f"recall@{t}"] = rc
    # Normal Consistency: cift yonlu, isaretten bagimsiz (|cos|)
    nc1 = np.abs((Pn * Gn[i_pg]).sum(1)).mean()
    nc2 = np.abs((Gn * Pn[i_gp]).sum(1)).mean()
    out["normal_consistency"] = float((nc1 + nc2) / 2)
    return out


def volume_iou(pred, gt, bound, res=64):
    """Voxel IoU. UYARI: ici dolu objelerde ic bolge sayiyi sisirir."""
    g = np.linspace(-bound, bound, res)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    try:
        a = pred.contains(pts)
        b = gt.contains(pts)
    except Exception:
        return None
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else None


def summarize(rows, keys):
    s = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=np.float64)
        if not len(v):
            continue
        # ANAHTARLAR SISTEM ADIYLA ON EKLI: 'ogrenci_fscore@0.01'.
        # Duz startswith("fscore") HICBIRINDE tutmuyordu => yuksek-iyi
        # metriklerin IYI kuyrugu (p90) KOTU kuyruk (p10) etiketiyle
        # raporlaniyordu (2026-08-27 bagimsiz denetim K4).
        _taban = k.split("_", 1)[-1] if "_" in k else k
        hi_iyi = _taban.startswith(("fscore", "precision", "recall", "normal",
                                    "iou", "siluet", "volume_iou"))
        s[k] = {"ort": float(v.mean()), "med": float(np.median(v)),
                ("p10" if hi_iyi else "p90"):
                    float(np.percentile(v, 10 if hi_iyi else 90))}
    return s


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dataset/lrm_ckpts/distilled_v2.pt")
    ap.add_argument("--teacher", default="dataset/lrm_ckpts/teacher_1024_tv.pt")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--split", default=None,
                    help="train/val/test. Verilirse uid'ler BURADAN gelir; "
                         "held-out geometri olcumu ancak boyle mumkun "
                         "(2026-08-27 denetim K3: eskiden uid'ler DAIMA ogretmen "
                         "bankasindan geliyordu, banka da %%100 train).")
    ap.add_argument("--uids_file", default="")
    ap.add_argument("--n_obj", type=int, default=32)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--n_pts", type=int, default=10000)
    ap.add_argument("--bound", type=float, default=defaults.BOUND)
    ap.add_argument("--level_steps", type=int, default=11,
                    help="siluete gore esik taramasindaki aday sayisi")
    ap.add_argument("--min_gt_iou", type=float, default=0.70,
                    help="GT mesh siluet dogrulamasi bunun altindaysa obje atlanir. DIKKAT: silhouette_iou nokta-splat tabanli, ince objelerde kenar etkisiyle DOGRU hizalamada bile ~0.89 verebilir; gercek hizalama hatasi ~0.1-0.4 olarak gorunur. O yuzden esik gevsek.")
    ap.add_argument("--no_teacher", action="store_true")
    ap.add_argument("--no_vol_iou", action="store_true",
                    help="voxel IoU pahali (mesh.contains); atlamak icin")
    ap.add_argument("--cache", default="dataset/gt_mesh_norm")
    ap.add_argument("--out", default="dataset/lrm_eval")
    ap.add_argument("--tag", default="geom")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("uid->glb haritasi yukleniyor...", flush=True)
    table = json.load(gzip.open(PATHS_GZ, "rt", encoding="utf-8"))

    tk = None if a.no_teacher else torch.load(a.teacher, map_location="cpu",
                                              weights_only=False)
    if a.uids_file:
        _u = json.load(io.open(a.uids_file, encoding="utf-8"))
        uids = (_u["uids"] if isinstance(_u, dict) else _u)[:a.n_obj]
    elif a.split is not None:
        uids = json.load(io.open(a.train_list, encoding="utf-8"))[a.split][:a.n_obj]
    elif tk is not None:
        uids = tk["uids"][:a.n_obj]
    else:
        uids = json.load(io.open(a.train_list, encoding="utf-8"))["train"][:a.n_obj]
    # TAVAN yalnizca bankada BULUNAN uid'ler icin verilebilir.
    tp_idx = None
    if tk is not None:
        _yer = {u: i for i, u in enumerate(tk["uids"])}
        tp_idx = [_yer.get(u) for u in uids]
        if any(i is None for i in tp_idx):
            print("UYARI: uid'lerin bir kismi ogretmen bankasinda yok -> "
                  "TAVAN satiri dusuruldu (held-out kumede beklenen).", flush=True)
            tk, tp_idx = None, None
    model = LRM(n_samples=defaults.N_SAMPLES, bound=a.bound).to(DEV).eval()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    print(f"geometri: {len(uids)} obje | izgara {a.grid}^3 | {a.n_pts} nokta "
          f"| ckpt adim {ck.get('step','?')}", flush=True)

    def pred_mesh(triplane, uid):
        """Marching cubes esigini SILUETE GORE sec.

        HATA VE DUZELTME (2026-08-27): once "pozitif yogunlugun medyani" kullanildi.
        Bu YANLIS: NeRF ciktisi SOFTPLUS, yani yogunluk HER NOKTADA > 0
        (olculdu: >0 orani 1.0000, medyan 0.000, maks 393.7). Pozitif degerlerin
        medyani ~0 cikiyor, marching cubes 0 seviyesinde TUM bound kupunu mesh'liyor:
        bbox [-0.6,0.6]^3, siluet IoU 0.08, F@1% = 0.000 -- OGRETMEN ICIN BILE.
        Yogunluk olcegi objeden objeye degisiyor (extract_mesh.py'de zaten yaziliydi),
        o yuzden esik VERIDEN secilmeli: GT siluetiyle ortusmeyi maksimize eden seviye.
        """
        dens, cols = density_grid(triplane, model.nerf, a.bound, res=a.grid)
        pos = dens[dens > 0]
        if pos.numel() < 100:
            return None, 0.0, 0.0
        lo, hi = float(pos.min()), float(pos.max())
        if not (hi > lo):
            return None, 0.0, 0.0
        cand = sorted({float(v) for v in torch.logspace(
            np.log10(max(lo, 1e-4)), np.log10(hi), a.level_steps)})
        best = (None, 0.0, -1.0)
        for lv in cand:
            mm = to_mesh(dens, cols, a.bound, lv)
            if mm is None or len(mm.faces) < 200:
                continue
            mm = clean(mm)
            iou = float(np.mean([silhouette_iou(mm, a.renders_dir, uid, v)
                                 for v in (0, 1, 2, 3)]))
            if iou > best[2]:
                best = (mm, lv, iou)
        return best

    rows, atlanan = [], {}
    for n, uid in enumerate(uids):
        gm, err = gt_mesh(uid, table, a.cache)
        if gm is None:
            atlanan[uid] = err
            continue
        # --- GECERLILIK KAPISI: GT mesh gercekten render'la ayni cercevede mi?
        try:
            gt_iou = float(np.mean([silhouette_iou(gm, a.renders_dir, uid, v)
                                    for v in (0, 1, 2, 3)]))
        except Exception as e:
            atlanan[uid] = f"siluet_hata:{type(e).__name__}"
            continue
        if gt_iou < a.min_gt_iou:
            atlanan[uid] = f"gt_hizalama_dustu:{gt_iou:.3f}"
            continue

        rgb, c2w, K = None, None, None
        with io.open(f"{a.renders_dir}/{uid}/meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        from eval_suite import load_view
        rgb, _, c2w, K = load_view(a.renders_dir, uid, 0, 224)
        with torch.no_grad():
            tp_s = model.make_triplane(rgb[None].to(DEV), c2w[None].to(DEV),
                                       K[None].to(DEV)).float()
        G, Gn = sample_surface(gm, a.n_pts)

        systems = {"ogrenci": tp_s}
        if tk is not None:
            systems["ogretmen"] = tk["triplanes"][tp_idx[n]].to(DEV)
        row = {"uid": uid, "gt_siluet_iou": gt_iou}
        for name, tp in systems.items():
            pm, lvl, sil = pred_mesh(tp, uid)
            if pm is None:
                row[f"{name}_bos"] = True
                continue
            row[f"{name}_esik"] = lvl
            row[f"{name}_siluet_iou"] = sil
            # TOPOLOJI SAGLIGI: Faz C .glb uretecek, mesh kapali ve makul olmali.
            # Daha once olculdugunde Euler = -92 (47 tunel) cikmisti -- "putur
            # putur" gorunum render artefakti degil GERCEK geometriydi.
            rep = report(pm)
            row[f"{name}_watertight"] = float(bool(rep["watertight"]))
            row[f"{name}_bilesen"] = float(rep["components"])
            try:
                row[f"{name}_euler"] = float(pm.euler_number)
            except Exception:
                pass
            P, Pn = sample_surface(pm, a.n_pts)
            for k, v in geom_metrics(P, Pn, G, Gn).items():
                row[f"{name}_{k}"] = v
            if not a.no_vol_iou:
                row[f"{name}_iou_hacim"] = volume_iou(pm, gm, a.bound)
        rows.append(row)
        if (n + 1) % 5 == 0:
            print(f"  {n+1}/{len(uids)} (gecerli {len(rows)}, atlanan {len(atlanan)})",
                  flush=True)

    if not rows:
        print("HIC GECERLI OBJE YOK -- atlanma sebepleri:", flush=True)
        for k, v in list(atlanan.items())[:10]:
            print(f"  {k}: {v}")
        return

    keys = [k for k in rows[0] if k not in ("uid",)]
    summary = summarize(rows, keys)
    stamp = runstamp.run_stamp(vars(a))
    path = os.path.join(a.out, f"geom_{a.tag}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_object": rows, "atlanan": atlanan,
                   "stamp": stamp}, f, indent=1)

    print(f"\n=== GEOMETRI ({len(rows)} gecerli obje, {len(atlanan)} atlandi) ===")
    print(f"GT hizalama dogrulamasi (siluet IoU): "
          f"ort {summary['gt_siluet_iou']['ort']:.3f} / p10 "
          f"{summary['gt_siluet_iou']['p10']:.3f}   <- 1.0'a yakin olmali")
    hdr = (f"{'sistem':<12}{'F@1%':>14}{'F@2%':>14}{'F@5%':>14}"
           f"{'Chamfer-L1':>14}{'NormCons':>14}{'IoU-hacim':>14}")
    print("\n" + hdr); print("-" * len(hdr))
    for name in ("ogretmen", "ogrenci"):
        if f"{name}_fscore@0.01" not in summary:
            continue
        def g(k, lo="p10"):
            d = summary.get(f"{name}_{k}")
            return "-".rjust(14) if not d else f"{d['ort']:>7.3f}/{d.get(lo, d.get('p90')):>6.3f}"
        print(f"{name:<12}{g('fscore@0.01')}{g('fscore@0.02')}{g('fscore@0.05')}"
              f"{g('chamfer_L1','p90')}{g('normal_consistency')}{g('iou_hacim')}")
    print("(her hucre: ortalama/p10; Chamfer'da ortalama/p90 -- dusuk iyi)")
    print("")
    print("TOPOLOJI (Faz C mesh cikisi icin):")
    for name in ("ogretmen", "ogrenci"):
        w = summary.get(f"{name}_watertight")
        if not w:
            continue
        eu = summary.get(f"{name}_euler", {})
        bl = summary.get(f"{name}_bilesen", {})
        nan = float("nan")
        print(f"  {name:<10} kapali %{w['ort']*100:>5.1f}"
              f" | Euler ort {eu.get('ort', nan):>9.1f} med {eu.get('med', nan):>8.1f}"
              f" | bilesen ort {bl.get('ort', nan):>5.1f}")
    print("  (Euler 2 = kure gibi tek kapali yuzey; cok negatif = tunel dolu)")
    if atlanan:
        from collections import Counter
        print("\natlanma sebepleri:", dict(Counter(v.split(":")[0] for v in atlanan.values())))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
