"""KATMAN 1+2+4 degerlendirme paketi: render uzayinda TAM metrik seti.

NEDEN VAR (2026-08-26):
Blok 2 kapisinin esikleri `rel` (triplane ozellik uzayinda MSE) uzerine yazilmisti.
`rel` KALIBRE DEGIL: triplane kanallarinin gorsel karsiligi yok, bir kanaldaki
0.1 hata gorunmez olabilir, baskasindaki ayni hata objeyi yok edebilir. Yani
"rel < 0.25 ise gecti" demek, isaretlenmemis bir cetvele bakmakti.

Bu paket sayilari GORSEL uzaya tasir ve -- kritik olan -- her sayiyi
TAVAN / TABAN / RAKIP ucgeni icinde raporlar:

  TAVAN  ogretmen triplane render'i  -> temsilin verebilecegi en iyi
  TABAN  ortalama-obje               -> cokus cizgisi (blob)
  RAKIP  en-yakin-komsu retrieval    -> Tatarchenko ve ark. (CVPR 2019):
         "tek-gorunum 3B aglari rekonstruksiyon degil siniflandirma yapiyor".
         Model, girdiye en cok benzeyen BASKA egitim objesinin GT'sini
         dondurmekten iyi degilse, rekonstruksiyon yapmiyordur.

Metrikler (referanslardan):
  LRM      : PSNR, SSIM, LPIPS, CLIP-similarity  (TUM ablasyonlarda ayni 4'lu)
  TripoSR  : PSNR/SSIM/LPIPS + Chamfer/F-score (geometri icin: eval_geometry.py)
  bize ozgu: siluet IoU, top-1 retrieval, ortalama-baseline orani, obje-arasi std

ONEMLI AYRINTI -- LPIPS agi:
Egitimde LPIPS-VGG kayip olarak kullaniliyor. Degerlendirmede de VGG kullanmak
kendi kayip fonksiyonumuzu metrik diye raporlamak olur. Burada ALEXNET kullanilir.

DAGILIM, ORTALAMA DEGIL:
Ortalama PSNR objelerin %20'sinin coktugunu gizler. Her metrik icin
ortalama + medyan + p10 raporlanir (Tatarchenko'nun dagilim onerisi).

Kullanim:
    python scripts/eval_suite.py --ckpt dataset/lrm_ckpts/distilled_v2.pt \
        --teacher dataset/lrm_ckpts/teacher_1024_tv.pt \
        --train_list dataset/train_list_v2.json \
        --renders_dir dataset/renders_opp_score3 --n_obj 128 --tag blok2
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrm import cameras, defaults, guards, runstamp, imutil
from lrm.model import LRM

DEV = "cuda"
INPUT_VIEW = 0          # kanonik "on" -- egitimde girdi olarak kullanilan
NOVEL_VIEW = 8          # halka gorunumu: GERCEK 3B testi
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- veri
def load_view(render_dir, uid, view, res):
    """Tek gorunum: (RGB beyaz-zeminli, alpha, c2w, K)."""
    with io.open(f"{render_dir}/{uid}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    v = meta["views"][view]
    # 2026-09-02: burada takma-adli + premultiply-sirasi-ters kucultme vardi.
    # BIRINCIL kapi olcum araci oldugu icin V3 tablosundaki her PSNR/SSIM/LPIPS
    # sayisi bozuk bir GT'ye karsi olculmustu. Bkz. lrm/imutil.py.
    a = imutil.yukle_rgba(f"{render_dir}/{uid}/{v['file']}", res)
    rgb = a[:3] * a[3:4] + (1 - a[3:4])          # beyaz zemine kompozit
    K = cameras.scale_intrinsics(torch.tensor(v["intrinsic"], dtype=torch.float32),
                                 meta.get("resolution", 512), res)
    c2w = torch.linalg.inv(torch.tensor(v["extrinsic"], dtype=torch.float32))
    return rgb, a[3:4], c2w, K


def build(render_dir, uids, res, in_res=224, n_input=1, normalize_cams=False):
    """n_input: girdi olarak kullanilacak KANONIK gorunum sayisi (1..4).

    NEDEN PARAMETRE (2026-08-27): projenin hedefi "kullanici 1 foto YA DA
    4 kanonik aci verir". Egitim gercekten 1-4 arasi rastgele gorunum
    kullaniyor (dataset.py: rng.randint(1, hi)) ama BUGUNE KADAR HER
    DEGERLENDIRME TEK GORUNUMLE yapildi -- yani vaat edilen iki moddan biri
    hic olculmedi. 4 gorunum 1'den KOTU bile olabilir (kamera ozelliklerinin
    fuzyonu bozuksa); olcmeden bilinmez.
    """
    data = []
    for u in uids:
        with io.open(f"{render_dir}/{u}/meta.json", encoding="utf-8") as f:
            canon = list(json.load(f).get("canonical_indices") or [0, 1, 2, 3])
        sel = canon[:max(1, min(n_input, len(canon)))]
        imgs, c2ws, Ks = [], [], []
        for vi in sel:
            a, _, c, k = load_view(render_dir, u, vi, in_res)
            imgs.append(a); c2ws.append(c); Ks.append(k)
        in_c2w = torch.stack(c2ws)
        tgt = {}
        for name, vi in (("girdi", INPUT_VIEW), ("yeni", NOVEL_VIEW)):
            tgt[name] = load_view(render_dir, u, vi, res)
        if normalize_cams:
            # Model kanoniklestirilmis kameralarla egitildiyse degerlendirme de
            # AYNI donusumden gecmeli; yoksa dagitim disi kamera gomulmesi.
            ref = in_c2w[0].clone()
            in_c2w = cameras.canonicalize(ref, in_c2w)
            for name in tgt:
                g, al, c2w, K = tgt[name]
                tgt[name] = (g, al, cameras.canonicalize(ref, c2w[None])[0], K)
        rec = {"uid": u,
               "in_img": torch.stack(imgs).to(DEV),
               "in_c2w": in_c2w.to(DEV),
               "in_K": torch.stack(Ks).to(DEV)}
        for name, (g, al, c2w, K) in tgt.items():
            rec[f"gt_{name}"] = g.to(DEV)
            rec[f"al_{name}"] = al.to(DEV)
            rec[f"c2w_{name}"] = c2w.to(DEV)
            rec[f"K_{name}"] = K.to(DEV)
        data.append(rec)
    return data


# --------------------------------------------------------------------------- metrik
class Metrics:
    def __init__(self, use_clip=True):
        import lpips
        # ALEXNET -- egitimdeki VGG DEGIL (bkz. dosya basligi)
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
        self.clip = None
        if use_clip:
            try:
                import open_clip
                m, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai")
                self.clip = m.to(DEV).eval()
                self.cm = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1)
                self.cs = torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1)
            except Exception as e:
                print(f"  [uyari] CLIP yuklenemedi ({type(e).__name__}), atlanacak",
                      flush=True)

    @torch.no_grad()
    def clip_sim(self, P, G):
        if self.clip is None:
            return None
        def emb(x):
            # OLCEK-DENETIMI: buyutme (deger cozunurlugu <= 224; CLIP girisi 224)
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
            f = self.clip.encode_image((x.clamp(0, 1) - self.cm) / self.cs)
            return F.normalize(f.float(), dim=-1)
        return (emb(P) * emb(G)).sum(-1)                       # (N,)

    @torch.no_grad()
    def per_object(self, P, G, A_pred=None, A_gt=None):
        """P,G: (N,3,H,W) [0,1]. Obje BASINA metrik sozlugu dondurur."""
        from skimage.metrics import structural_similarity as ssim_fn
        mse = ((P - G) ** 2).mean((1, 2, 3))
        out = {"psnr": (-10 * torch.log10(mse.clamp_min(1e-9))).cpu().numpy(),
               "mse": mse.cpu().numpy()}
        lp = self.lpips(P * 2 - 1, G * 2 - 1).flatten()
        out["lpips"] = lp.cpu().numpy()
        cs = self.clip_sim(P, G)
        if cs is not None:
            out["clip"] = cs.cpu().numpy()
        pn, gn = P.cpu().numpy(), G.cpu().numpy()
        # PARAMETRELER lrm/metrics.py:ssim_per_object ILE AYNI OLMALI
        # (2026-08-27 denetim K1): skimage varsayilani 7x7 DUZGUN pencere +
        # ORNEKLEM kovaryansi; metrics.py 11x11 GAUSS + POPULASYON kovaryansi.
        # Olculdu: ayni goruntude 0.8869 vs 0.8344 (0.053 fark, hep ayni yonde)
        # => egitim egrisindeki SSIM ile kapi raporundaki SSIM yan yana konamazdi.
        out["ssim"] = np.array([
            ssim_fn(gn[i].transpose(1, 2, 0), pn[i].transpose(1, 2, 0),
                    channel_axis=2, data_range=1.0, gaussian_weights=True,
                    sigma=1.5, use_sample_covariance=False) for i in range(len(pn))])
        if A_pred is not None and A_gt is not None:
            p = (A_pred > 0.5).float()
            g = (A_gt > 0.5).float()
            inter = (p * g).sum((1, 2, 3))
            union = ((p + g) > 0).float().sum((1, 2, 3)).clamp_min(1.0)
            out["iou"] = (inter / union).cpu().numpy()
        return out


def top1(P, G):
    """Her tahmin KENDI hedefine mi en yakin? (conditioning testi, sans = 1/N)"""
    D = ((P[:, None] - G[None]) ** 2).mean((2, 3, 4))
    return float((D.argmin(1) == torch.arange(len(P), device=P.device)).float().mean())


def summarize(per_obj):
    """ortalama + medyan + p10 -- dagilimi gizlememek icin (Tatarchenko)."""
    s = {}
    for k, v in per_obj.items():
        v = np.asarray(v, dtype=np.float64)
        lo = 10 if k in ("psnr", "ssim", "clip", "iou") else 90   # kotu uc hangi yonde
        s[k] = {"ort": float(v.mean()), "med": float(np.median(v)),
                "p10" if lo == 10 else "p90": float(np.percentile(v, lo))}
    return s


# --------------------------------------------------------------------------- sistemler
@torch.no_grad()
def render_bank(model, triplanes, data, view, res, nerf=None):
    """Verilen triplane bankasini hedef gorunumde render et.

    nerf: TAVAN icin ogretmenin KENDI decoder'i (denetim K5). None = modelinki."""
    P, A = [], []
    for i, d in enumerate(data):
        rgb, acc = model.render_view(triplanes[i], d[f"c2w_{view}"], d[f"K_{view}"],
                                     res, res, nerf=nerf)
        P.append(rgb); A.append(acc)
    return torch.stack(P), torch.stack(A)


@torch.no_grad()
def student_triplanes(model, data, amp=False):
    import contextlib
    tps = []
    for d in data:
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp
               else contextlib.nullcontext())
        with ctx:
            tp = model.make_triplane(d["in_img"], d["in_c2w"], d["in_K"])
        tps.append(tp.float())
    return tps


def neighbor_baseline(data, view):
    """RAKIP: her obje icin, GIRDI goruntusu en cok benzeyen BASKA objenin
    hedef-gorunum GT'si. 'Tani ve getir' stratejisinin ta kendisi."""
    # RAKIP baseline'i DAIMA ilk (kanonik on) gorunumden hesaplanir ki
    # n_input degisince baseline degismesin -- kollar karsilastirilabilir kalsin.
    # 224 -> 32 = 7x kucultme; takma adli bilinear burada RAKIP eslesmesini
    # kaydiriyordu (lrm/imutil.py).
    X = torch.stack([imutil.kucult(d["in_img"][:1], 32)[0] for d in data])
    D = ((X[:, None] - X[None]) ** 2).mean((2, 3, 4))
    D.fill_diagonal_(float("inf"))
    j = D.argmin(1)
    P = torch.stack([data[int(k)][f"gt_{view}"] for k in j])
    A = torch.stack([data[int(k)][f"al_{view}"] for k in j])
    return P, A


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dataset/lrm_ckpts/distilled_v2.pt")
    ap.add_argument("--teacher", default="dataset/lrm_ckpts/teacher_1024_tv.pt")
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--split", default=None,
                    help="train/val/test. ACIKCA verilirse ogretmen bankasini "
                         "EZER (denetim K2). Verilmezse: banka varsa banka, "
                         "yoksa train.")
    ap.add_argument("--uids_file", default="")
    ap.add_argument("--n_obj", type=int, default=128)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--in_res", type=int, default=224)
    ap.add_argument("--normalize_cams", action="store_true",
                    help="model --normalize_cams ile egitildiyse SART")
    ap.add_argument("--n_input", type=int, default=1,
                    help="girdi olarak kac KANONIK gorunum (1=tek foto, 4=dort aci)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_teacher", action="store_true")
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--bound", type=float, default=defaults.BOUND)
    ap.add_argument("--density_bias", type=float, default=0.0)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--out", default="dataset/lrm_eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- uid secimi: ogretmen bankasi varsa ONUN sirasi baglayicidir
    tk = None
    if not a.no_teacher and os.path.isfile(a.teacher):
        tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
    # UID SECIM ONCELIGI (2026-08-27 denetim K2): eskiden ogretmen dosyasi
    # diskte varsa --split SESSIZCE YOK SAYILIYORDU. Banka %100 train oldugu
    # icin `--split val` train objelerini olcuyor, kunyeye ise 'val' yaziyordu.
    # Artik ACIKCA verilen --split her zaman kazanir.
    if a.uids_file:
        uids = json.load(io.open(a.uids_file, encoding="utf-8"))
        uids = (uids["uids"] if isinstance(uids, dict) else uids)[:a.n_obj]
    elif a.split is not None:
        tl = json.load(io.open(a.train_list, encoding="utf-8"))
        uids = tl[a.split][:a.n_obj]
    elif tk is not None:
        uids = tk["uids"][:a.n_obj]
    else:
        tl = json.load(io.open(a.train_list, encoding="utf-8"))
        uids = tl["train"][:a.n_obj]
    # Ogretmen (TAVAN) yalnizca bankada BULUNAN uid'ler icin verilebilir.
    tidx = None
    if tk is not None:
        _yer = {u: i for i, u in enumerate(tk["uids"])}
        tidx = [_yer.get(u) for u in uids]
        _bulunan = sum(1 for i in tidx if i is not None)
        if _bulunan < len(uids):
            print("UYARI: %d/%d obje ogretmen bankasinda YOK -> TAVAN satiri "
                  "dusuruldu (held-out kumede beklenen)." % (len(uids) - _bulunan,
                                                             len(uids)), flush=True)
            tk, tidx = None, None
    print(f"degerlendirme: {len(uids)} obje | res {a.res} | GIRDI GORUNUM SAYISI "
          f"{a.n_input} | yeni gorunum {NOVEL_VIEW}", flush=True)

    model = LRM(n_samples=defaults.N_SAMPLES, bound=a.bound,
                density_bias=a.density_bias).to(DEV).eval()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    missing = model.load_state_dict(sd, strict=False)
    print(f"  ckpt: {a.ckpt} (adim {ck.get('step','?')}) "
          f"eksik={len(missing.missing_keys)} fazla={len(missing.unexpected_keys)}",
          flush=True)

    data = build(a.renders_dir, uids, a.res, a.in_res, a.n_input,
                 normalize_cams=a.normalize_cams)
    M = Metrics(use_clip=not a.no_clip)

    tp_student = student_triplanes(model, data, amp=a.amp)
    systems = {"ogrenci": tp_student}
    if tk is not None and tidx is not None:
        systems["ogretmen"] = [tk["triplanes"][i].to(DEV) for i in tidx[:len(uids)]]
        # Ogretmenin kendi decoder'i (fit_teacher triplane + NeRF'i BIRLIKTE
        # optimize ediyor). Yoksa TAVAN, ogrenci egitildikce kayar.
        from lrm.nerf import TriplaneNeRF
        _tn = TriplaneNeRF(in_dim=96, hidden=64).to(DEV).eval()
        _tn.load_state_dict(tk["nerf"])
        for _p in _tn.parameters():
            _p.requires_grad_(False)
        tk["nerf_mod"] = _tn

    results, previews = {}, {}
    per_obj_kayit = {}          # denetim K8: ham diziler
    for view in ("girdi", "yeni"):
        G = torch.stack([d[f"gt_{view}"] for d in data])
        Ag = torch.stack([d[f"al_{view}"] for d in data])
        results[view] = {}

        for name, tps in systems.items():
            # TAVAN kendi decoder'iyla render edilir (denetim K5): ogretmen
            # triplane'i + ogrencinin NeRF'i = egitildikce KAYAN bir cetvel.
            _nerf = tk["nerf_mod"] if (name == "ogretmen" and tk) else None
            P, A = render_bank(model, tps, data, view, a.res, nerf=_nerf)
            po = M.per_object(P, G, A, Ag)
            per_obj_kayit.setdefault(view, {})[name] = {
                k: np.asarray(v).tolist() for k, v in po.items()}
            g = guards.collapse_flags(preds=P, acc_mean=float(A.mean()))
            results[view][name] = {
                **summarize(po),
                "top1": top1(P, G), "sans": 1.0 / len(data),
                "acc": float(A.mean()), "obj_std": guards.inter_object_std(P),
                "dejenere": bool(g.get("degenerate", False)),
            }
            if name == "ogrenci":
                previews[view] = (P[:6].cpu(), G[:6].cpu())

        # TABAN: ortalama-obje
        Pm = G.mean(0, keepdim=True).expand_as(G).contiguous()
        Am = Ag.mean(0, keepdim=True).expand_as(Ag).contiguous()
        _po_taban = M.per_object(Pm, G, Am, Ag)
        per_obj_kayit.setdefault(view, {})['ortalama(TABAN)'] = {
            k: np.asarray(v).tolist() for k, v in _po_taban.items()}
        results[view]["ortalama(TABAN)"] = {
            **summarize(_po_taban),
            "top1": 0.0, "sans": 1.0 / len(data), "acc": float(Am.mean()),
            "obj_std": 0.0, "dejenere": True}

        # RAKIP: en-yakin-komsu
        Pn, An = neighbor_baseline(data, view)
        _po_rakip = M.per_object(Pn, G, An, Ag)
        per_obj_kayit.setdefault(view, {})['komsu(RAKIP)'] = {
            k: np.asarray(v).tolist() for k, v in _po_rakip.items()}
        results[view]["komsu(RAKIP)"] = {
            **summarize(_po_rakip),
            "top1": top1(Pn, G), "sans": 1.0 / len(data), "acc": float(An.mean()),
            "obj_std": guards.inter_object_std(Pn), "dejenere": False}

    # ---- rapor
    stamp = runstamp.run_stamp(vars(a))
    path = os.path.join(a.out, f"eval_{a.tag}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        # OBJE-BASINA DIZILER (2026-08-27 denetim K8): eskiden sadece ozet
        # yaziliyordu. 'Hangi objeler coktu, ortak paydalari ne?' ve iki
        # checkpoint arasinda ESLESTIRILMIS test bunlar olmadan imkansiz,
        # ve modeli tekrar kosturmak gerekirdi (last_*.pt ezilmis olur).
        json.dump({"summary": results, "per_object": per_obj_kayit,
                   "uids": uids, "stamp": stamp}, f, indent=1)

    order = ["ogretmen", "ogrenci", "komsu(RAKIP)", "ortalama(TABAN)"]
    for view in ("girdi", "yeni"):
        print(f"\n=== {view.upper()} GORUNUM (res {a.res}, {len(data)} obje) ===")
        hdr = f"{'sistem':<18}{'PSNR':>17}{'SSIM':>15}{'LPIPS':>15}{'CLIP':>15}{'IoU':>15}{'top-1':>9}{'obj_std':>9}"
        print(hdr); print("-" * len(hdr))
        for name in order:
            r = results[view].get(name)
            if not r:
                continue
            def c(k, lo="p10"):
                if k not in r:
                    return f"{'-':>15}"
                d = r[k]
                return f"{d['ort']:>7.3f}/{d.get(lo, d.get('p90')):>6.3f}"
            ps = r["psnr"]
            flag = "  <<< DEJENERE" if r["dejenere"] else ""
            ostd = r["obj_std"] if r["obj_std"] is not None else float("nan")
            print(f"{name:<18}{ps['ort']:>9.2f}/{ps['p10']:>6.2f}"
                  f"{c('ssim'):>15}{c('lpips','p90'):>15}{c('clip'):>15}{c('iou'):>15}"
                  f"{r['top1']:>9.1%}{ostd:>9.4f}{flag}")
        print(f"{'sans (top-1)':<18}{results[view]['ogrenci']['sans']:>9.1%}"
              "        (her hucre: ortalama/p10, LPIPS'te ortalama/p90)")

    # ---- gorsel izgara
    try:
        import torchvision.utils as vutils
        for view, (P, G) in previews.items():
            grid = vutils.make_grid(torch.cat([G, P]), nrow=len(G), padding=2)
            Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).clip(0, 255)
                            .astype(np.uint8)).save(
                os.path.join(a.out, f"onizleme_{a.tag}_{view}.png"))
    except Exception as e:
        print(f"[uyari] onizleme yazilamadi: {type(e).__name__}")

    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
