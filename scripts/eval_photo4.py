"""4 kanonik acidan cekilmis GERCEK foto ile yeni-goruum degerlendirmesi.

Gercek fotoda 3B GT yoktur; ama ayni obje 4 kanonik acidan cekilirse
1'ini girdi verip DIGER 3'unu hedef yapabiliriz. Bu, in-domain
`eval_suite --split test --n_input 1` ile AYNI problem.

RAKIP = "girdiyi kopyala": yeni aci icin girdi fotosunu aynen basmak.
Derinlikte yayilmis bir blob bu rakibi GECEMEZ -- bu yuzden ayirt edici.
(Girdi goruumu siluet IoU'su ayirt edici DEGILDI: 2026-08-28.)
"""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from lrm import cameras, defaults
from lrm.model import LRM
from lrm import compat

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def canon(renders_dir, idx):
    u = sorted(os.listdir(renders_dir))[0]
    m = json.load(open(os.path.join(renders_dir, u, "meta.json"), encoding="utf-8"))
    v = m["views"][m["canonical_indices"][idx]]
    return v["intrinsic"], v["extrinsic"], m.get("resolution", 512)


def load(path, res):
    im = Image.open(path).convert("RGBA").resize((res, res), Image.LANCZOS)
    t = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.
    rgb = t[:3] * t[3:4] + (1 - t[3:4])          # beyaz kompozit
    return rgb, (t[3] > 0.5)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def iou(a, b):
    inter = float((a & b).sum()); union = float((a | b).sum())
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True, help="hazir/ klasoru (on,sag,arka,sol .png)")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--in_res", type=int, default=defaults.INPUT_RES,
                    help="encoder girdi cozunurlugu. TEK KAYNAK "
                         "lrm/defaults.py; sabit 224 yazmak, INPUT_RES "
                         "degistiginde Plucker haritasiyla sessiz "
                         "ayrismaya yol acar.")
    ap.add_argument("--mirror", action="store_true",
                    help="sag<->sol takasi (donme yonu belirsizligi)")
    ap.add_argument("--n_input", type=int, default=1, help="1 veya 4")
    ap.add_argument("--out", default="dataset/gercek_foto/kumanda4")
    ap.add_argument("--tag", default="kumanda4")
    a = ap.parse_args()

    # compat: eski (dar) ve yeni mimari checkpoint'lerin ikisini de yukler.
    # 2026-08-29: duz LRM()+load_state_dict eski ckpt'te RuntimeError veriyordu
    # ve zincirde cagri `|| true` ile sarili oldugu icin SESSIZCE yutulurdu.
    model, _arch, tk = compat.load_lrm(a.ckpt, device=DEV)
    print(f"mimari: {_arch}", flush=True)

    names = ["on", "sag", "arka", "sol"]
    if a.mirror: names = ["on", "sol", "arka", "sag"]
    gt_rgb, gt_m = [], []
    for n in names:
        r, m = load(os.path.join(a.dir, n + ".png"), a.res)
        gt_rgb.append(r); gt_m.append(m)

    # --- girdi ---
    n_in = a.n_input
    imgs, c2ws, Ks = [], [], []
    for i in range(n_in):
        rgb_i, _ = load(os.path.join(a.dir, names[i] + ".png"), a.in_res)
        Kr, Er, master = canon(a.renders_dir, i)
        imgs.append(rgb_i)
        Ks.append(cameras.scale_intrinsics(torch.tensor(Kr, dtype=torch.float32), master, a.in_res))
        c2ws.append(torch.linalg.inv(torch.tensor(Er, dtype=torch.float32)))
    img = torch.stack(imgs).to(DEV); K = torch.stack(Ks).to(DEV); c2w = torch.stack(c2ws).to(DEV)

    with torch.no_grad():
        tp = model.make_triplane(img, c2w, K)
        preds = []
        for i in range(4):
            Kr, Er, master = canon(a.renders_dir, i)
            Kv = cameras.scale_intrinsics(torch.tensor(Kr, dtype=torch.float32), master, a.res).to(DEV)
            cv = torch.linalg.inv(torch.tensor(Er, dtype=torch.float32)).to(DEV)
            rgb, acc = model.render_view(tp, cv, Kv, a.res, a.res)
            preds.append((rgb.float().cpu(), (acc[0].float().cpu() > 0.5)))

    novel = [i for i in range(4) if i >= n_in]
    print(f"[{a.tag}] girdi={n_in} gorunum  mirror={a.mirror}  ckpt adim {tk.get('step','?')}")
    print(f"{'aci':<12}{'MODEL psnr':>12}{'RAKIP psnr':>12}{'MODEL IoU':>11}{'RAKIP IoU':>11}")
    print("-" * 58)
    mp, rp, mi, ri = [], [], [], []
    base_rgb, base_m = gt_rgb[0], gt_m[0]           # RAKIP: girdiyi kopyala
    for i in range(4):
        p_rgb, p_m = preds[i]
        P, R = psnr(p_rgb, gt_rgb[i]), psnr(base_rgb, gt_rgb[i])
        I, J = iou(p_m, gt_m[i]), iou(base_m, gt_m[i])
        lbl = names[i] + (" (GIRDI)" if i < n_in else "")
        print(f"{lbl:<12}{P:>12.2f}{R:>12.2f}{I:>11.3f}{J:>11.3f}")
        if i in novel: mp.append(P); rp.append(R); mi.append(I); ri.append(J)
    print("-" * 58)
    os.makedirs(a.out, exist_ok=True)
    def im(t): return Image.fromarray((t.clamp(0,1).numpy().transpose(1,2,0)*255).astype(np.uint8))
    W = Image.new("RGB", (a.res*4, a.res*2), (255,255,255))
    for i in range(4):
        W.paste(im(gt_rgb[i]), (a.res*i, 0))
        W.paste(im(preds[i][0]), (a.res*i, a.res))
    g = os.path.join(a.out, f"{a.tag}_in{n_in}{'_mirror' if a.mirror else ''}.png")
    W.save(g); print(f"  -> {g}  (ust: GERCEK foto, alt: MODEL)")
    if not mp:      # n_input=4: held-out goruum YOK, sadece yeniden uretim
        print("  (girdi=4: held-out goruum yok -- yukaridaki satirlar YENIDEN"
              " URETIM kalitesi, genelleme DEGIL)")
        return
    MP, RP, MI, RI = np.mean(mp), np.mean(rp), np.mean(mi), np.mean(ri)
    print(f"{'YENI ACI ort':<12}{MP:>12.2f}{RP:>12.2f}{MI:>11.3f}{RI:>11.3f}")
    print(f"  in-domain referans (test, 1 gorunum): psnr 20.43  IoU 0.662")
    gecti = (MP > RP) and (MI > RI) and (MI >= 0.50)
    print(f"  ON-KAYITLI KAPI: {'GECTI' if gecti else 'KALDI'} "
          f"(model>rakip: psnr {MP>RP}, IoU {MI>RI}; IoU>=0.50: {MI>=0.50})")



if __name__ == "__main__":
    main()
