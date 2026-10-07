"""KAFES (blok) ARTEFAKTI TESHISI -- triplane'de periyot-2 yapisi var mi?

NEDEN (2026-09-03, ZINCIR3 olcumu + gorseli):
`last_V3` GT'nin %29'u kadar HF ENERJISI uretiyor ama `kor` 0.008,
`kor_orta` 0.024 -- yani enerjinin tamami YANLIS YERDE. Gorselde bu, kayanin
uzerindeki gozle gorulur dokuma/kafes deseni olarak cikiyor.

HIPOTEZ: `TriplaneHead` tek bir `ConvTranspose2d(kernel=2, stride=2)`.
`k=2, s=2` ORTUSMESIZDIR: her token bagimsiz bir 2x2 blok uretir, komsu
karisimi YOKTUR. Model tasiyamadigi yuksek frekansi uretmeye zorlandiginda
blok SINIRLARI gorunur hale gelir => triplane'de PERIYOT-2 (Nyquist) yapisi.

OLCUM: triplane duzlemlerinin 2B guc spektrumunda Nyquist frekansindaki
(periyot 2 = piksel bazinda dama tahtasi) gucun, komsu frekanslara oranli
fazlasi. Blok artefakti varsa bu oran 1'den belirgin buyuk cikar.

KONTROL KOLU SART: ogretmen triplane'leri SERBEST PARAMETRE (head yok).
Onlarda oran ~1 cikmali; cikmiyorsa olcum yontemi bozuk demektir.

CPU'da kosar (GPU dolu olabilir). Ogrenci icin tek obje ileri gecisi yeter.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from lrm import compat  # noqa: E402
from lrm.dataset import LRMDataset  # noqa: E402


def blok_sinir_orani(duzlem, blok=2):
    """BIRINCIL METRIK: hata/yapi 2x2 BLOK SINIRLARINDA mi birikiyor?

    NEDEN BU, `nyquist_orani` DEGIL (2026-09-03, bagimsiz denetim):
    `nyquist_orani` AYIRT EDICI DEGIL. CPU tezgahinda gercek calisma
    noktasinda (dim=4): k2s2 Nyq 9.92, k4s2 Nyq 10.31 -- ORTUSMELI kol DAHA
    YUKSEK. Ona dayali on-kayitli kapim ("k4s2 < 2 ise dogrulandi") DOGRU
    hipotezi REDDEDERDI. Ayrica tek FFT kutusunun ~11 kutuluk medyana orani
    agir kuyruklu (std, ortalamanin 2 kati).

    `blok_sinir` ayni noktada: k2s2 8.67 <-> k4s2 1.01 <-> up_conv 1.02
    (hedef/kontrol 1.046). Ayrim 8x ve kararli.

    ConvTranspose2d(k=2,s=2) ORTUSMESIZ: her token bagimsiz bir 2x2 blok
    uretir. Satir ciftleri (0,1),(2,3),... blok ICI; (1,2),(3,4),... blok
    SINIRI. Oran ~1 ise blok yapisi gorunmuyor demektir.
    """
    x = duzlem.float()
    d_sat = (x[..., 1:, :] - x[..., :-1, :]).abs()
    d_sut = (x[..., :, 1:] - x[..., :, :-1]).abs()
    # indis i = satir i ile i+1 arasi fark. i tek ise blok SINIRI.
    ic = torch.cat([d_sat[..., 0::blok, :].reshape(-1),
                    d_sut[..., :, 0::blok].reshape(-1)])
    sinir = torch.cat([d_sat[..., 1::blok, :].reshape(-1),
                       d_sut[..., :, 1::blok].reshape(-1)])
    return float(sinir.mean() / ic.mean().clamp_min(1e-12))


def nyquist_orani(duzlem):
    """Periyot-2 gucunun komsu bantlara gore fazlaligi.

    duzlem: (C, H, W). Kanal ortalamasi alinmis 2B FFT gucu uzerinde:
      pay   = en yuksek frekans kosesindeki (Nyquist x Nyquist) guc
      payda = onun cevresindeki halkanin medyan gucu
    Blok (dama tahtasi) artefakti tam Nyquist'te tepe yapar.
    """
    x = duzlem.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    P = (torch.fft.fft2(x).abs() ** 2).mean(0)          # (H,W) kanal ort.
    P = torch.fft.fftshift(P)
    H, W = P.shape
    cy, cx = H // 2, W // 2
    # fftshift sonrasi Nyquist DC'den en uzak nokta: (0,0) kosesi
    ny = float(P[0, 0])
    # kosenin cevresindeki 5x5 halka (kendisi haric) -> yerel taban
    # simetrik komsuluk: fftshift sonrasi dort kose de Nyquist civari
    kom = torch.cat([P[0:3, 0:3].reshape(-1), P[0:3, -2:].reshape(-1),
                     P[-2:, 0:3].reshape(-1), P[-2:, -2:].reshape(-1)])
    kom = kom[kom != P[0, 0]]
    taban = float(kom.median())
    # ayrica: yarim-Nyquist referansi (periyot 4)
    ref = float(P[cy // 2, cx // 2])
    return ny / max(taban, 1e-20), ny, taban, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=[])
    ap.add_argument("--teachers", nargs="*", default=[])
    ap.add_argument("--train_list", default="dataset/train_list_v2.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--n_obj", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    DEV = a.device

    ds = LRMDataset(a.train_list, a.renders_dir, split="train", render_res=64,
                    n_sup=1, augment=False, deterministic=True)

    def ozet(ad, vals, kontrol=False):
        bs = np.array([v[0] for v in vals]); ny = np.array([v[1] for v in vals])
        sem = bs.std(ddof=1) / np.sqrt(bs.size) if bs.size > 1 else float("nan")
        yorum = ("KONTROL" if kontrol else
                 ("KAFES VAR" if bs.mean() >= 3.0 else
                  ("kafes yok" if bs.mean() < 1.5 else "sinirda")))
        print(f"{ad:<38} {bs.mean():>10.2f} {1.96*sem:>7.2f} "
              f"{ny.mean():>9.2f} {bs.size:>4}   {yorum}")

    print(f"\nkafes teshisi: {a.n_obj} obje | cihaz {DEV}")
    print("BIRINCIL: blok_sinir = ort|fark| BLOK SINIRINDA / BLOK ICINDE.")
    print("  ~1.0 : blok yapisi gorunmuyor | >=3.0 : ortusmesiz deconv imzasi")
    print("IKINCIL: Nyq (periyot-2 gucu / komsu bant). AYIRT EDICI DEGIL --")
    print("  CPU tezgahinda k2s2 9.92 <-> k4s2 10.31 (ortusmeli kol DAHA yuksek).")
    print("  Sadece referans icin basiliyor; KARAR blok_sinir'den verilir.")
    print(f"\n{'sistem':<38} {'blok_sinir':>10} {'+-95%':>7} {'Nyq':>9} {'n':>4}   yorum")
    print("-" * 84)

    for p in a.teachers:
        tk = torch.load(p, map_location="cpu", weights_only=False)
        tp = tk["triplanes"]
        vals = [(blok_sinir_orani(tp[i, d]), nyquist_orani(tp[i, d])[0])
                for i in range(min(a.n_obj, tp.shape[0])) for d in range(tp.shape[1])]
        ozet(os.path.basename(p)[:-3] + " [OGRETMEN/serbest]", vals, kontrol=True)

    for p in a.students:
        mdl, arch, sk = compat.load_lrm(p, device=DEV)
        mdl.eval()
        vals = []
        with torch.no_grad():
            for i in range(a.n_obj):
                it = ds[i]
                tp = mdl.make_triplane(it["input_imgs"].to(DEV),
                                       it["input_c2w"].to(DEV),
                                       it["input_K"].to(DEV)).float()
                for d in range(tp.shape[0]):
                    vals.append((blok_sinir_orani(tp[d]), nyquist_orani(tp[d])[0]))
        ozet(os.path.basename(p)[:-3] + f" [ogrenci {sk.get('step')}]", vals)

    print("\nHead yapisi: TriplaneHead = ConvTranspose2d(k=upsample, s=upsample).")
    print("k==s ise ORTUSME YOK: her token bagimsiz bir blok uretir, komsu")
    print("karisimi olmaz. Kafes cikarsa cozum ortusmeli cekirdek (or. k=4,s=2)")
    print("ya da deconv sonrasi yumusatma katmani.")


if __name__ == "__main__":
    main()
