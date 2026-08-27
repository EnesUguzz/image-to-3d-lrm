"""Triplane -> yogunluk izgarasi -> marching cubes -> renkli .glb  (Faz C'nin kalbi)

NEDEN EGITIMDEN ONCE (docs/egitim-oncesi-hazirlik-plani.md, E1):
Nihai urun `.glb`. Modelin yogunluk alani "sisli/bulutsu" ise marching cubes cop
verir ve bunu duzeltmek EGITIM HEDEFINI degistirir (seyreklik/entropi terimi
eklemek gerekir). 27 saatlik bir kosudan sonra ogrenilecek sey degil.

Bu yuzden test ONCE ORACLE triplane uzerinde yapilir: obje basina serbest
triplane (fit_teacher / bench_triplane_fit ciktisi) en iyi halimiz -- 25 dB.
Oradan duzgun mesh cikmiyorsa LRM'den hic cikmaz.

KABUL OLCUTLERI (plan Blok 5):
  - bagli parca sayisi < 20
  - ucgen sayisi 10k - 200k
  - esik dayanikliligi: sigma araliginda hacim %30'dan az degissin

Kullanim:
  # oracle/ogretmen bankasindan i. objenin mesh'i
  python scripts/extract_mesh.py --teacher dataset/lrm_ckpts/teacher_1024_new.pt --index 0
  # egitilmis LRM'den, tek fotodan
  python scripts/extract_mesh.py --ckpt dataset/lrm_ckpts/last.pt --uid <uid>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm import cameras, defaults
from lrm.nerf import TriplaneNeRF
from lrm.triplane import sample_triplane

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def density_grid(triplane, nerf, bound, res=128, chunk=1 << 18):
    """bound kupunu res^3 orneklerle tarar; (density, rgb) izgaralarini dondurur."""
    lin = torch.linspace(-bound, bound, res, device=triplane.device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    dens = torch.empty(pts.shape[0], device=pts.device)
    cols = torch.empty(pts.shape[0], 3, device=pts.device)
    for i in range(0, pts.shape[0], chunk):
        p = pts[i:i + chunk]
        d, c = nerf(sample_triplane(triplane, p, bound=bound))
        dens[i:i + chunk] = d.squeeze(-1)
        cols[i:i + chunk] = c
    return dens.reshape(res, res, res), cols.reshape(res, res, res, 3)


def to_mesh(dens, cols, bound, level):
    from skimage import measure
    d = dens.detach().float().cpu().numpy()
    if not (d.min() < level < d.max()):
        return None
    verts, faces, _, _ = measure.marching_cubes(d, level=level)
    res = d.shape[0]
    # izgara indeksi -> dunya koordinati  (meshgrid sirasi: [z, y, x])
    idx = np.clip(np.round(verts).astype(int), 0, res - 1)
    vcol = cols.detach().float().cpu().numpy()[idx[:, 0], idx[:, 1], idx[:, 2]]
    world = verts[:, ::-1] / (res - 1) * (2 * bound) - bound     # (x, y, z)
    import trimesh
    m = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    m.visual.vertex_colors = np.concatenate(
        [np.clip(vcol, 0, 1) * 255, np.full((len(vcol), 1), 255)], axis=1).astype(np.uint8)
    return m


def silhouette_iou(mesh, renders_dir, uid, view_idx, res=128, n_pts=120000):
    """Mesh silüeti GT alpha maskesine ne kadar oturuyor (0-1).

    Marching cubes esigi OBJEYE GORE DEGISIYOR (yogunluk olcegi sabit degil):
    ayni obje icin IoU 0.589 (esik 1) -> 0.741 (esik 150) -> 0.545 (esik 260).
    Elle esik secmek Faz C'de imkansiz; bu olcut esigi VERIYE gore secmeyi saglar.
    Faz C'de de kullanilabilir: kullanici fotosunun arka plani zaten siliniyor,
    yani girdi maskesi elimizde olacak.
    """
    import numpy as np
    from PIL import Image
    with open(os.path.join(renders_dir, uid, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    v = meta["views"][view_idx]
    K = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                 meta.get("resolution", 512), res).numpy()
    ext = np.array(v["extrinsic"], dtype=np.float64)          # world->camera
    pts = mesh.sample(n_pts)
    P = (ext[:3, :3] @ pts.T).T + ext[:3, 3]
    z = -P[:, 2]                                              # OpenGL: -z ileri
    ok = z > 1e-6
    u = K[0, 0] * (P[ok, 0] / z[ok]) + K[0, 2]
    w = -K[1, 1] * (P[ok, 1] / z[ok]) + K[1, 2]
    m = np.zeros((res, res), bool)
    ui, wi = np.round(u).astype(int), np.round(w).astype(int)
    g = (ui >= 0) & (ui < res) & (wi >= 0) & (wi < res)
    m[wi[g], ui[g]] = True
    gt = np.asarray(Image.open(os.path.join(renders_dir, uid, v["file"])).convert("RGBA")
                    .resize((res, res)))[:, :, 3] > 16
    return float((m & gt).sum() / max((m | gt).sum(), 1))


def mean_iou(mesh, renders_dir, uid, views=(0, 1, 2, 3)):
    return sum(silhouette_iou(mesh, renders_dir, uid, v) for v in views) / len(views)


def clean(mesh, min_face_frac=0.01):
    """Kucuk yuzen parcalari at, dejenere ucgenleri temizle."""
    import trimesh
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        big = max(len(p.faces) for p in parts)
        keep = [p for p in parts if len(p.faces) >= min_face_frac * big]
        mesh = trimesh.util.concatenate(keep) if keep else mesh
    return mesh


def report(mesh):
    parts = mesh.split(only_watertight=False)
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "components": int(len(parts)), "watertight": bool(mesh.is_watertight),
            "volume": float(abs(mesh.volume)) if mesh.is_volume else None,
            "bbox": [round(float(v), 3) for v in mesh.extents]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="", help="fit_teacher ciktisi (oracle triplane bankasi)")
    ap.add_argument("--index", type=int, default=0, help="ogretmen bankasindaki obje indeksi")
    ap.add_argument("--ckpt", default="", help="egitilmis LRM checkpoint'i")
    ap.add_argument("--uid", default="", help="--ckpt ile: bu objenin fotosundan uret")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--bound", type=float, default=defaults.BOUND)
    ap.add_argument("--level", type=float, default=0.0,
                    help="marching cubes esigi (0 = otomatik tarama)")
    ap.add_argument("--out", default="dataset/mesh_out")
    ap.add_argument("--tag", default="mesh")
    ap.add_argument("--auto_level", action="store_true",
                    help="esigi GT silüet IoU'sunu maksimize ederek sec")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.teacher:
        tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
        triplane = tk["triplanes"][a.index].to(DEV)
        nerf = TriplaneNeRF(in_dim=triplane.shape[0] * triplane.shape[1]).to(DEV)
        nerf.load_state_dict(tk["nerf"])
        uid = tk["uids"][a.index]
        bound = tk.get("cfg", {}).get("bound", a.bound)
        print(f"oracle triplane: {uid} (index {a.index}), bound {bound}")
    elif a.ckpt:
        from lrm.model import LRM
        from lrm.dataset import _load_rgba
        model = LRM().to(DEV).eval()
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu",
                                         weights_only=False)["model"])
        uid = a.uid
        with open(os.path.join(a.renders_dir, uid, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        v = meta["views"][meta["canonical_indices"][0]]
        rgba = _load_rgba(os.path.join(a.renders_dir, uid, v["file"]), 224)
        img = (rgba[:3] * rgba[3:4] + (1 - rgba[3:4]))[None].to(DEV)
        K = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                     meta.get("resolution", 512), 224)[None].to(DEV)
        c2w = torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float32))[None].to(DEV)
        with torch.no_grad():
            triplane = model.make_triplane(img, c2w, K)
        nerf, bound = model.nerf, model.bound
        print(f"LRM tek-foto: {uid}")
    else:
        ap.error("--teacher ya da --ckpt ver")

    dens, cols = density_grid(triplane, nerf, bound, a.grid)
    # sample_triplane padding_mode="border" kullaniyor => kutu KENARINDA duzlem
    # kenar degeri tekrarlanip sahte bir kabuk olusturabiliyor. Renderer'da
    # `inside` maskesi bunu gizliyor, marching cubes ise goruyor. 1 voksel kirp.
    dens[0, :, :] = dens[-1, :, :] = 0
    dens[:, 0, :] = dens[:, -1, :] = 0
    dens[:, :, 0] = dens[:, :, -1] = 0
    dmin, dmax = float(dens.min()), float(dens.max())
    print(f"yogunluk: min {dmin:.4f}  medyan {float(dens.median()):.4f}  maks {dmax:.4f}")

    # ESIK SECIMI -- yogunluk alani COK SEYREK (olculdu: medyan 0.0000, maks 366).
    # Tum hacmin yuzdeliklerini almak hepsini 0'a dusuruyordu => gurultuyu de
    # mesh'liyordu (bbox tum kutu kadar cikiyordu). Esikler SIFIRDAN BUYUK
    # yogunluklarin dagilimindan alinir.
    flat = dens.flatten().float()
    pos = flat[flat > 1e-4]
    if pos.numel() == 0:
        print("MESH CIKMADI -- hacimde hic yogunluk yok")
        return
    print(f"  dolu voksel: {pos.numel()}/{flat.numel()} (%{100*pos.numel()/flat.numel():.3f}), "
          f"pozitif medyan {float(pos.median()):.3f}")
    levels = ([a.level] if a.level > 0 else
              sorted({round(float(torch.quantile(pos, q)), 4) for q in
                      (0.20, 0.40, 0.60, 0.75, 0.90)}))
    if a.auto_level and os.path.isdir(os.path.join(a.renders_dir, uid)):
        # esigi GT silüetine gore sec: yogunluk olcegi objeden objeye degisiyor
        lo, hi = float(pos.min()), float(pos.max())
        cand = sorted({round(float(v), 4) for v in
                       torch.logspace(np.log10(max(lo, 1e-4)), np.log10(hi), 9)})
        scored = []
        for lv in cand:
            mm = to_mesh(dens, cols, bound, lv)
            if mm is None or len(mm.faces) < 200:
                continue
            mm = clean(mm)
            scored.append((mean_iou(mm, a.renders_dir, uid), lv, mm))
        if scored:
            scored.sort(reverse=True, key=lambda t: t[0])
            print("  otomatik esik taramasi (IoU):")
            for iou, lv, mm in scored[:5]:
                print(f"    esik {lv:9.4f} -> IoU {iou:.3f}  ({len(mm.faces)} ucgen)")
            iou, lv, mesh = scored[0]
            r = report(mesh)
            path = os.path.join(a.out, f"{a.tag}_{uid[:8]}.glb")
            mesh.export(path)
            with open(os.path.join(a.out, f"{a.tag}_{uid[:8]}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"uid": uid, "level": lv, "iou": iou, "grid": a.grid,
                           "bound": bound, "mesh": r}, f, indent=1)
            print("")
            print(f"-> {path}")
            print(f"   IoU {iou:.3f} | {r}")
            return

    rows = []
    best = None
    for lv in levels:
        m = to_mesh(dens, cols, bound, lv)
        if m is None or len(m.faces) == 0:
            rows.append((lv, None))
            continue
        m = clean(m)
        r = report(m)
        rows.append((lv, r))
        print(f"  esik {lv:8.4f}: {r['faces']:7d} ucgen, {r['components']:3d} parca, "
              f"bbox {r['bbox']}")
        # KABUL: parca < 20 ve ucgen 10k-200k
        if r["components"] < 20 and 10_000 <= r["faces"] <= 200_000 and best is None:
            best = (lv, m, r)

    if best is None:                      # olcut tutmadiysa en cok ucgenli olani al
        cand = [(lv, r) for lv, r in rows if r]
        if not cand:
            print("MESH CIKMADI -- yogunluk alani yuzey olusturmuyor")
            return
        lv = max(cand, key=lambda t: t[1]["faces"])[0]
        best = (lv, clean(to_mesh(dens, cols, bound, lv)), None)
        print(f"  !! kabul olcutu tutmadi, en iyi aday esik {lv:.4f} kullanildi")

    lv, mesh, r = best
    path = os.path.join(a.out, f"{a.tag}_{uid[:8]}.glb")
    mesh.export(path)
    # esik dayanikliligi: hacim esikle ne kadar oynuyor
    vols = [rr["volume"] for _, rr in rows if rr and rr.get("volume")]
    stab = (max(vols) / min(vols)) if len(vols) > 1 and min(vols) > 0 else None
    summary = {"uid": uid, "level": lv, "grid": a.grid, "bound": bound,
               "density_min": dmin, "density_max": dmax,
               "mesh": r or report(mesh), "hacim_orani_esik_taramasi": stab}
    with open(os.path.join(a.out, f"{a.tag}_{uid[:8]}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"\n-> {path}")
    print(f"   {summary['mesh']}")
    if stab:
        print(f"   esik dayanikliligi (hacim maks/min): {stab:.2f}x  (<1.3 iyi)")


if __name__ == "__main__":
    main()
