"""Kosu kunyesi + agirlik-degisti kontrolu.

NEDEN VAR:
1) Bu projede iki kez A/B kollari sessizce ayni/yanlis veriyi okudu (bkz.
   PROJE-DEVIR-BELGESI 7.2) ve bir kez "encoder acildi" iddiasi dogrulanamadi
   (M_enc4). Her sonuc dosyasi, onu ureten kosuyu tek basina tanimlamali:
   git sha, tam argv, torch surumu, GPU, config hash.
2) "Su bayrak acildi" demek yetmez -- AGIRLIKLARIN GERCEKTEN DEGISTIGI
   olculmelidir. weight_snapshot/weight_delta bunu yapar.
"""
import hashlib
import json
import os
import subprocess
import sys
import time


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=10,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.dirname(os.path.abspath(__file__)))))
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def config_hash(cfg, n=12):
    """Config sozlugunun kararli kisa hash'i. Anahtar sirasi onemsiz;
    serilestirilemeyen degerler repr'ine dusurulur (patlamaz)."""
    blob = json.dumps(cfg or {}, sort_keys=True, default=repr, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:n]


def run_stamp(cfg=None):
    """Sonuc json'una gomulecek kunye."""
    try:
        import torch
        # str() SART: torch.__version__ bir TorchVersion NESNESI, str degil.
        # Kunyeye oldugu gibi konursa checkpoint torch.load(weights_only=True)
        # ile okunamaz (PyTorch 2.6+ varsayilani) => --resume/--init_from duser.
        tv = str(torch.__version__)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cuda = str(getattr(torch.version, "cuda", None) or "")
    except Exception:
        tv, gpu, cuda = "", "", None
    return {
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": list(sys.argv),
        "torch": tv,
        "cuda": cuda,
        "gpu": gpu,
        "python": sys.version.split()[0],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": config_hash(cfg),
    }


# --------------------------------------------------------------- agirlik kontrolu
def weight_snapshot(model, prefix="", only_trainable=False):
    """Parametrelerin CPU kopyasi. prefix verilirse sadece o alt agac."""
    snap = {}
    for name, p in model.named_parameters():
        if prefix and not name.startswith(prefix):
            continue
        if only_trainable and not p.requires_grad:
            continue
        snap[name] = p.detach().float().cpu().clone()
    return snap


def weight_delta(model, snapshot):
    """Her parametre icin |son - baslangic| ortalamasi."""
    cur = dict(model.named_parameters())
    out = {}
    for name, before in snapshot.items():
        p = cur.get(name)
        if p is None:
            continue
        out[name] = float((p.detach().float().cpu() - before).abs().mean().item())
    return out


def changed_count(delta, eps=0.0):
    """Gercekten degismis tensor sayisi."""
    return sum(1 for v in delta.values() if v > eps)


def format_delta(delta, label="agirlik", eps=0.0):
    """Log satiri: kac tensor degisti, ortalama ve en buyuk degisim."""
    if not delta:
        return f"[{label}] snapshot bos"
    n = changed_count(delta, eps)
    vals = list(delta.values())
    worst = max(delta.items(), key=lambda kv: kv[1])
    return (f"[{label}] degisen {n}/{len(delta)} tensor | "
            f"ort |delta|={sum(vals)/len(vals):.2e} | "
            f"en cok {worst[0]}={worst[1]:.2e}")
