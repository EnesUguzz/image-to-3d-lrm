"""Asama 1 (ogretmen) ve Asama 2 (distile ogrenci) icin GORSEL KANIT.

Neden: fit_teacher tek bir onizleme yaziyor, distill_lrm HIC yazmiyor.
Sayilar iyi gorunurken cikti bozuk olabilir (proje dersi: "ciktiya bak,
metrige degil"). Bu betik ucunu yan yana koyar:  GT | OGRETMEN | OGRENCI.

Ayni kamera, ayni obje, tek fark hangi triplane'in render edildigi.
"""
import argparse, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm.dataset import LRMDataset
from lrm.nerf import TriplaneNeRF
from lrm.renderer import volume_render
from lrm.triplane import sample_triplane
from lrm import defaults, compat, cameras

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="dataset/lrm_ckpts/teacher_1024_v3.pt")
    ap.add_argument("--student", default="dataset/lrm_ckpts/distilled_v3.pt")
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--n_obj", type=int, default=8)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--out", default="dataset/lrm_val_previews/KANIT_asama12.png")
    ap.add_argument("--student_nerf", action="store_true",
                    help="3. asama ckpt'i icin ZORUNLU: ogrenci kendi NeRF'iyle "
                         "render edilir (render ince ayarinda NeRF de egitildi)")
    a = ap.parse_args()

    tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
    print(f"ogretmen: adim {tk['step']}, {len(tk['uids'])} obje, done={tk.get('done')}")

    # OGRETMEN DECODER'I KUNYEDEN KURULUR (2026-09-02 kod incelemesi).
    # Onceden in_dim=96/hidden=64 SABITTI ve bound defaults.BOUND varsayiliyordu.
    # `fit_teacher` artik --tp_ch / --nerf_hidden / --nerf_layers / --pos_enc /
    # --bound yaziyor: farkli kanalla fit edilmis bir ogretmen YUKLENEMEZ (gurultulu),
    # ama farkli --bound ile fit edilmis bir ogretmen SORUNSUZ yuklenip YANLIS
    # ornekleme uzayinda render edilirdi. Bu betigin tek isi GT|ogretmen|ogrenci
    # karsilastirmasi uretmek; o karsilastirma sessizce gecersiz olurdu.
    _cfg = tk.get("cfg", {}) or {}
    _ch = int(_cfg.get("tp_ch", 32))
    T_BOUND = float(_cfg.get("bound", defaults.BOUND))
    nerf = TriplaneNeRF(in_dim=3 * _ch, hidden=int(_cfg.get("nerf_hidden", 64)),
                        layers=int(_cfg.get("nerf_layers", 4)),
                        pos_enc=int(_cfg.get("pos_enc", 0)),
                        pos_bound=T_BOUND).to(DEV).eval()
    nerf.load_state_dict(tk["nerf"])
    print(f"  ogretmen decoder: tp_ch {_ch}, hidden {_cfg.get('nerf_hidden', 64)}, "
          f"layers {_cfg.get('nerf_layers', 4)}, pos_enc {_cfg.get('pos_enc', 0)}, "
          f"bound {T_BOUND}, tp_res {_cfg.get('tp_res', 64)}")
    model, arch, sk = compat.load_lrm(a.student, device=DEV)
    print(f"ogrenci: adim {sk.get('step')}, mimari {arch}")

    ds = LRMDataset(a.train_list, a.renders_dir, split="train",
                    render_res=a.res, n_sup=1, augment=False, deterministic=True)
    assert ds.uids[:len(tk["uids"])] == tk["uids"], "uid sirasi uyusmuyor"

    def rend(tp, c2w, K, dec=None, bnd=None):
        # fit_teacher.py:77 ile AYNI yol -- ogretmen ve ogrenci ayni
        # renderer'dan gecmezse karsilastirma gecersiz olur.
        # bound SISTEM BASINA (2026-09-02): ogretmen kendi kunyesindeki bound'la,
        # ogrenci modelin bound'uyla ornekleniyor. Sabit defaults.BOUND yazmak,
        # --bound 0.552 ile fit edilmis bir ogretmeni HATASIZ ama YANLIS uzayda
        # render ederdi.
        _dec = nerf if dec is None else dec
        _b = T_BOUND if bnd is None else bnd
        o, d = cameras.rays_from_camera(c2w[0], K[0], a.res, a.res)
        o, d = o.to(DEV), d.to(DEV)
        def q(pts):
            # pts DE gecilir: pos_enc>0 ile fit edilmis bir decoder pts'siz
            # cagrilinca ValueError atar (nerf.py), sessizce kodlamasiz calismaz.
            den, rgb = _dec(sample_triplane(tp, pts, bound=_b), pts)
            ins = (pts.abs().amax(-1, keepdim=True) <= _b).to(den.dtype)
            return den * ins, rgb
        rgb, acc = volume_render(o, d, defaults.NEAR, defaults.FAR,
                                 defaults.N_SAMPLES, q, bg_color=None)
        rgb = rgb.reshape(a.res, a.res, 3).permute(2, 0, 1)
        acc = acc.reshape(a.res, a.res, 1).permute(2, 0, 1)
        return (rgb + (1 - acc)).clamp(0, 1)

    # 3. asamada NeRF de egitildi -> ogrenci triplane'i ogretmenin decoder'indan
    # gecirmek gecersiz olur. Her sistem kendi decoder'iyla render edilir.
    s_nerf = model.nerf if a.student_nerf else nerf
    print("ogrenci decoder: " + ("kendi NeRF'i" if a.student_nerf else "ogretmen NeRF'i"))

    rows, psnr_t, psnr_s = [], [], []
    with torch.no_grad():
        for i in range(a.n_obj):
            it = ds[i]
            gt = it["sup_rgb"][0].to(DEV)
            c2w, K = it["sup_c2w"][:1].to(DEV), it["sup_K"][:1].to(DEV)
            t_tp = tk["triplanes"][i].to(DEV)
            s_tp = model.make_triplane(it["input_imgs"].to(DEV),
                                       it["input_c2w"].to(DEV), it["input_K"].to(DEV))
            t_img = rend(t_tp, c2w, K)
            s_img = rend(s_tp, c2w, K, s_nerf,
                         float(model.bound) if a.student_nerf else T_BOUND)
            mse = lambda x: float(((x - gt) ** 2).mean())
            psnr_t.append(-10 * np.log10(max(mse(t_img), 1e-9)))
            psnr_s.append(-10 * np.log10(max(mse(s_img), 1e-9)))
            rows.append(torch.cat([gt, t_img, s_img], -1).cpu())

    grid = (torch.cat(rows, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im = Image.new("RGB", (grid.shape[1], grid.shape[0] + 18), "white")
    im.paste(Image.fromarray(grid), (0, 18))
    d = ImageDraw.Draw(im)
    _sl = "OGRENCI (as.3)" if a.student_nerf else "OGRENCI (as.2)"
    for j, t in enumerate(["GERCEK", "OGRETMEN (as.1)", _sl]):
        d.text((j * a.res + 4, 4), t, fill="black")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    im.save(a.out)
    print(f"\n{'obje':>5} {'OGRETMEN dB':>12} {'OGRENCI dB':>11} {'fark':>7}")
    for i, (t, s) in enumerate(zip(psnr_t, psnr_s)):
        print(f"{i:>5} {t:>12.2f} {s:>11.2f} {s-t:>7.2f}")
    print(f"{'ORT':>5} {np.mean(psnr_t):>12.2f} {np.mean(psnr_s):>11.2f} "
          f"{np.mean(psnr_s)-np.mean(psnr_t):>7.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
