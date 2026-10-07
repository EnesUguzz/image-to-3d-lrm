"""Eski mimarili checkpoint'leri yukleme.

2026-08-29'da iki mimari degisikligi yapildi:
  - TriplaneHead: Linear(dim->C) + ConvT(C->C)  =>  ConvT(dim->C)
  - TriplaneNeRF: 2 gizli katman              =>  4

Bu, eski checkpoint'leri (last_TAM_asama3.pt, distilled_v2.pt, teacher_*.pt)
yeni kodla uyumsuz yapar. Karsilastirma yapabilmek icin (eski kosu <-> yeni
kosu) eski mimariyi state_dict'ten TESPIT edip kurabilmemiz gerekiyor.

ONEMLI: bu bir "sessizce uydur" katmani DEGIL. Mimari state_dict'ten okunur ve
`arch` alaninda RAPORLANIR; cagiran hangi mimariyi yukledigini bilir.
"""


# Render sonucunu DEGISTIREN ve state_dict'te GORUNMEYEN her sey.
# Bunlar parametre degil (density_bias yogunluga eklenen sabit, bound ornekleme
# uzayi, n_samples ornekleme sikligi) => uyusmazlik hicbir sekil hatasi vermez,
# sadece sessizce baska bir modeli olcersin.
KUNYE_ALANLARI = ("density_bias", "noise_std", "bound", "n_samples")

# 2026-09-02: olcek yontemi kunyeye girdi. Ortam degiskeni checkpoint'le
# SEYAHAT ETMEZ; kunyeye yazilmazsa eski yolla oturtulmus bir ogretmene
# yeni yolla uretilmis GT ile bakilir ve bunu hicbir sey soylemez.
OLCEK_ALANI = "olcek_yontemi"

# 2026-09-03: `input_res` KUNYE_ALANLARI'na KONULAMAZ cunku `LRM.__init__`
# kwarg'i degil, SINIF NITELIGI (`LRM.INPUT_RES`, model.py:19) -- kwarg olarak
# gecmek TypeError verir. Ayrica import aninda `defaults.INPUT_RES`'ten
# baglaniyor, yani ortam degiskeni checkpoint'le SEYAHAT ETMEZ.
# Uyusmazligin bedeli sessiz: `model.py:52` Plucker haritasini
# `scale_intrinsics(K, INPUT_RES, side)` ile kuruyor; `side` patch sayisindan
# (yani GERCEK girdiden) geliyor, `INPUT_RES` varsayilandan. 448'de egitilmis
# bir checkpoint 224 varsayilaniyla olculurse HICBIR HATA ALINMAZ, sadece
# baska bir model olculur -- ustelek saatlerce suren egitimin ARDINDAN.
GIRDI_RES_ALANI = "input_res"

# Kunyesiz (eski format) ckpt'ler icin zincirin FIILEN kullandigi degerler.
# fit_teacher.py TriplaneNeRF'i density_bias/noise_std vermeden kuruyor (=0.0),
# distill_lrm.py `LRM(n_samples=defaults.N_SAMPLES)` diyor, bound defaults.BOUND.
KUNYESIZ_VARSAYIM = {"density_bias": 0.0, "noise_std": 0.0}


def kunye_kwargs(tk, lrm_kwargs):
    """Checkpoint kunyesini LRM kwargs'ina cevirir.

    Doner: (kwargs, uyari_metni_veya_None)

    NEDEN (2026-09-02 kod incelemesi): `load_lrm` kunyeyi HIC okumuyordu, LRM
    kutuphane varsayilanlariyla kuruluyordu. Bunu dort arac kullaniyor
    (bench_keskinlik --students, eval_elevation, eval_photo4, kanit_asama12).
    `--density_bias 1.0` ile egitilmis bir model 0.0 ile render edilince
    tamamen baska bir yogunluk alani cikar ve HICBIR HATA ALINMAZ.
    Ve density_bias, bu projede uc kez yasanan `acc->0` cokusunun olculmus
    panzehiri -- yani tuzak, tam da tavsiye edilen ayari kullandigimiz an kurulur.

    Cagiranin acikca verdigi deger kazanir (ornegin adil karsilastirma icin
    n_samples sabitlemek), ama SESSIZ kalmaz: uyari metninde raporlanir.
    """
    from lrm import defaults
    kunye = tk.get("render_cfg") if isinstance(tk, dict) else None
    notlar = []
    kw = dict(lrm_kwargs)
    if kunye is None:
        varsayim = dict(KUNYESIZ_VARSAYIM)
        varsayim.setdefault("bound", defaults.BOUND)
        varsayim["n_samples"] = defaults.N_SAMPLES
        for k, v in varsayim.items():
            kw.setdefault(k, v)
        return kw, ("KUNYESIZ ckpt (eski format): render ayarlari DOGRULANAMIYOR. "
                    "Zincirin fiili varsayilanlari kabul edildi -> "
                    + ", ".join(f"{k}={kw[k]}" for k in varsayim))
    for k in KUNYE_ALANLARI:
        if k not in kunye or kunye[k] is None:
            continue
        if k in lrm_kwargs and lrm_kwargs[k] != kunye[k]:
            notlar.append(f"{k}: kunye {kunye[k]} EZILDI -> {lrm_kwargs[k]}")
            continue
        kw[k] = kunye[k]
    return kw, ("; ".join(notlar) if notlar else None)


def mimari_tespit(sd):
    """state_dict'ten mimariyi cikar. Doner: dict(head_linear, nerf_layers)."""
    head_linear = any(k.startswith("triplane_head.proj.") for k in sd)
    # HEAD TABANI (2026-09-03): `TriplaneHead` artik iki tabani destekliyor --
    # k2s2 (ortusmesiz, varsayilan) ve k4s2 (ortusmeli, k=4 s=2 p=1). Cekirdek
    # boyutu state_dict'ten OKUNMALI; yoksa k4s2 bir checkpoint k2s2 olarak
    # kurulur ve `load_state_dict` sekil uyusmazligiyla patlar -- ustelik
    # SAATLERCE suren egitimin ARDINDAN, olcum adiminda.
    _hw = sd.get("triplane_head.up.weight")
    head_tip = "k2s2"
    if _hw is not None and not head_linear and _hw.shape[2] == 4:
        head_tip = "k4s2"
    # backbone.0/2/4/6 -> Linear katmanlari (aralarinda ReLU var)
    # anahtar bicimi: nerf.backbone.<idx>.weight  -> idx 2. konumda
    lin = sorted({int(k.split(".")[2]) for k in sd
                  if k.startswith("nerf.backbone.") and k.endswith(".weight")})
    return {"head_linear": head_linear, "head_tip": head_tip,
            "nerf_layers": max(1, len(lin))}


def load_lrm(ckpt_path, device="cuda", **lrm_kwargs):
    """Checkpoint'i mimarisini tespit ederek yukle.

    Doner: (model, arch, tk)  -- arch mimari bilgisini TASIR, sessiz degildir.
    """
    import os
    import torch
    from lrm.model import LRM
    from lrm import nerf as nerf_mod
    from lrm import triplane as tri_mod

    tk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = tk["model"] if "model" in tk else tk
    arch = mimari_tespit(sd)
    lrm_kwargs, _uyari = kunye_kwargs(tk, lrm_kwargs)
    if _uyari:
        import sys as _s
        print(f"[compat] {os.path.basename(str(ckpt_path))}: {_uyari}",
              file=_s.stderr, flush=True)
    arch = dict(arch, **{k: lrm_kwargs[k] for k in KUNYE_ALANLARI
                         if k in lrm_kwargs})

    if arch["head_linear"]:
        # ESKI TriplaneHead'i gecici olarak geri kur
        import torch.nn as nn

        class _EskiHead(nn.Module):
            def __init__(self, dim, out_channels=32, upsample=2):
                super().__init__()
                self.proj = nn.Linear(dim, out_channels)
                self.up = nn.ConvTranspose2d(out_channels, out_channels,
                                             kernel_size=upsample, stride=upsample)

            def forward(self, tp_grid):
                return self.up(self.proj(tp_grid).permute(0, 3, 1, 2))

        _yeni = tri_mod.TriplaneHead
        tri_mod.TriplaneHead = _EskiHead
        import lrm.model as _m
        _m.TriplaneHead = _EskiHead
    try:
        # head_tip mimariden geliyor; cagiran acikca vermediyse uygula.
        if not arch["head_linear"]:
            lrm_kwargs.setdefault("head_tip", arch["head_tip"])
        model = LRM(**lrm_kwargs)
        if arch["nerf_layers"] != 4:
            # KUNYE BURADA DA GECMELI. Aksi halde kunyeden okunan density_bias/
            # noise_std yeniden kurulan decoder'da 0.0'a duser ve `arch` hala
            # kunye degerini raporlar => cagirana YUKLENENIN TERSI soylenir.
            # Bu dal tam olarak compat.py'nin var olma sebebi olan eski
            # 2-katmanli checkpoint'lerde tetiklenir.
            model.nerf = nerf_mod.TriplaneNeRF(
                in_dim=model.nerf.backbone[0].in_features,
                hidden=model.nerf.backbone[0].out_features,
                layers=arch["nerf_layers"],
                density_bias=lrm_kwargs.get("density_bias", 0.0),
                noise_std=lrm_kwargs.get("noise_std", 0.0))
        model.load_state_dict(sd)          # strict: uyusmazlik GURULTULU patlar
    finally:
        if arch["head_linear"]:
            tri_mod.TriplaneHead = _yeni
            import lrm.model as _m
            _m.TriplaneHead = _yeni

    # GIRDI COZUNURLUGU: kunyeden geri kur (bkz. GIRDI_RES_ALANI yorumu).
    # ORNEK niteligi olarak yaziliyor => sinif niteligini golgeler, ayni
    # surecte farkli cozunurlukte iki checkpoint yan yana olculebilir.
    from lrm import defaults as _d
    _kunye = tk.get("render_cfg") if isinstance(tk, dict) else None
    _ires = (_kunye or {}).get(GIRDI_RES_ALANI)
    model.INPUT_RES = int(_ires) if _ires else int(_d.INPUT_RES)
    arch[GIRDI_RES_ALANI] = model.INPUT_RES
    if _ires and int(_ires) != int(_d.INPUT_RES):
        import sys as _s
        print(f"[compat] {os.path.basename(str(ckpt_path))}: girdi cozunurlugu "
              f"kunyeden {int(_ires)} (surecin varsayilani {int(_d.INPUT_RES)}) "
              f"-- cagiran GIRDIYI DE bu cozunurlukte vermeli.",
              file=_s.stderr, flush=True)
    return model.to(device).eval(), arch, tk
