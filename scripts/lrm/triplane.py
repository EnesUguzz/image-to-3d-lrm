"""Triplane: token izgarasi -> 3 duzlem (upsample) + nokta ornekleme."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneHead(nn.Module):
    """Token izgarasi -> 3 duzlem.

    RUTBE DARBOGAZI (2026-08-29, bagimsiz denetim -- OLCULDU):
    Onceki sira `Linear(dim -> C)` sonra `ConvTranspose2d(C -> C, k=2, s=2)` idi.
    `k=2, s=2` deconv ORTUSMESIZDIR: her hucre bagimsiz bir 2x2 blok uretir,
    komsu karisimi yoktur. Yani 2x2 blogun 4*C sayisinin TAMAMI, Linear'in
    indirdigi C boyutlu alt uzayda yatiyordu.

      etkin serbestlik  biz: 3*32^2*32  =  98.304
                   OpenLRM: 3*32^2*128 = 393.216     -> 4x dar

    Ve darbogaz tam olarak yuksek frekansin uretilmesi gereken yerde. Referans
    (OpenLRM `openlrm/models/rendering/`) Linear'i HIC kullanmaz, deconv'a
    dogrudan `dim` boyutlu token'i verir. Ayni siraya geciyoruz.
    """

    # Ogretmen bankasindan OLCULDU (teacher_1024_tv.pt, 1024 obje):
    # triplane degerlerinin std'si 0.2416. Ogrencinin uretmeyi ogrenecegi
    # dagilim bu; head baslangicta oraya yakin cikmali.
    HEDEF_STD = 0.24

    # ORTUSMELI TABAN (2026-09-03, opt-in; varsayilan DEGISMEDI)
    # OLCULDU (scripts/bench_head_kafes.py, serbest token + head, transformer
    # ve encoder YOK; hedef TAVAN2_K64; birincil metrik `blok_sinir` =
    # ort|fark| BLOK SINIRINDA / BLOK ICINDE, kontrol degeri 1.05):
    #
    #   dim  rel(k2s2) rel(k4s2)   blok_sinir(k2s2)  blok_sinir(k4s2)
    #    32     0.0669    0.0450        1.26              1.01
    #    16     0.1475    0.1018        1.66              1.00
    #     8     0.2423    0.1877        5.85              0.98
    #     4     0.3506    0.3145        8.67              1.01   <- gercek nokta
    #
    # Iki sonuc: (1) k2s2'de hata BLOK SINIRLARINDA birikiyor ve rel yukseldikce
    # patliyor; ortusmeli tabanda 1.00'da sabit. (2) Ortusmenin BEDELI YOK --
    # her dim'de k4s2 DAHA IYI uyuyor (%10-32). Takas degil, bedava kazanc.
    # ⚠️ RUTBE DEGISMIYOR (3 x 32^2 x 512 sinir aynidir); degisen TABAN.
    # ⚠️ Kafes BASLANGICTA YOK (egitimsiz k2s2 Nyq 1.08); OGRENILIYOR.
    def __init__(self, dim, out_channels=32, upsample=2, tip="k2s2"):
        super().__init__()
        self.tip = tip
        if tip == "k2s2":                      # VARSAYILAN -- ortusme YOK
            self.up = nn.ConvTranspose2d(dim, out_channels,
                                         kernel_size=upsample, stride=upsample)
            fan = dim
        elif tip == "k4s2":                    # her cikis pikseli 4 token'dan
            self.up = nn.ConvTranspose2d(dim, out_channels,
                                         kernel_size=2 * upsample,
                                         stride=upsample, padding=upsample // 2)
            fan = dim * 4
        else:
            raise ValueError(f"bilinmeyen head tipi: {tip}")
        # BASLANGIC OLCEGI -- KRITIK.
        # PyTorch'un ConvTranspose2d varsayilan init'i fan_in'i OUT_channels'tan
        # (32) hesaplar, oysa toplam IN_channels (512) uzerinden aliniyor =>
        # cikti std'si 1.16 ciktı, eski mimarinin 0.177'sine karsi 6.5x.
        # Buyuk triplane degeri -> NeRF'te softplus doygunlugu -> acc->0 cokusu,
        # ki bu projede UC KEZ isirdi (M_base, M_enc4, K2_bf16).
        # Birim varyansli token icin: std_cikti = sqrt(dim) * std_w  =>
        # cikti std'si fan_in'e gore olceklenir; k4s2'de fan 4x =>
        # ayni HEDEF_STD'yi tutturmak icin std yarilanir.
        nn.init.normal_(self.up.weight, std=self.HEDEF_STD / (fan ** 0.5))
        nn.init.zeros_(self.up.bias)

    def forward(self, tp_grid):
        x = tp_grid.permute(0, 3, 1, 2)              # (3, dim, r, r)
        return self.up(x)                            # (3, C, r*up, r*up)


def sample_triplane(triplane, points, bound=0.6):
    p = (points / bound).clamp(-1, 1)   # (N,3)
    planes_coords = [p[:, [0, 1]], p[:, [0, 2]], p[:, [1, 2]]]  # XY, XZ, YZ
    feats = []
    for plane, coords in zip(triplane, planes_coords):
        grid = coords.view(1, -1, 1, 2)  # (1, N, 1, 2)
        f = F.grid_sample(plane[None], grid, mode="bilinear",
                          align_corners=True, padding_mode="border")  # (1,C,N,1)
        feats.append(f.squeeze(0).squeeze(-1).T)  # (N, C)
    return torch.cat(feats, dim=-1)  # (N, 3C)


def tv_loss(triplane):
    """Toplam-varyasyon: komsu triplane hucreleri arasi ani sicramalari cezalar.

    NEDEN ORTAK YERDE (2026-08-26): `fit_teacher.py` bunu HIC uygulamiyordu,
    `train_lrm.py`/`bench_overfit.py` ise `w_tv=5e-4` ile uyguluyordu. Sonuc:
    ogretmen triplane'lerinin varyansinin **%40.5'i** yuksek frekansti ve o
    yuksek frekans render kalitesine sadece **0.34 dB** katkı veriyordu -- yani
    bilgi degil gurultu. Bu gurultu iki yerden vuruyordu:
      1) mesh: Euler -92 (47 tunel), komsu ucgen aci p90 38 derece
      2) DISTILASYON HEDEFI: rel = ((tp-hedef)^2)/hedef.var() icin ~0.40'lik
         OGRENILEMEZ taban olusturuyor => "kapasite siniri" kapisi gecersiz oluyor
    """
    dh = (triplane[..., 1:, :] - triplane[..., :-1, :]).abs().mean()
    dw = (triplane[..., :, 1:] - triplane[..., :, :-1]).abs().mean()
    return dh + dw
