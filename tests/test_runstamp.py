import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lrm import runstamp


def test_kunye_zorunlu_alanlari_tasir():
    s = runstamp.run_stamp({"lr": 4e-4})
    for k in ("git_sha", "git_dirty", "argv", "torch", "gpu", "time", "config_hash"):
        assert k in s, k


def test_kunye_json_serilestirilebilir():
    json.dumps(runstamp.run_stamp({"a": 1, "b": [1, 2]}))


def test_ayni_config_ayni_hash_farkli_config_farkli_hash():
    a = runstamp.config_hash({"lr": 4e-4, "batch": 8})
    b = runstamp.config_hash({"batch": 8, "lr": 4e-4})     # anahtar sirasi onemsiz
    c = runstamp.config_hash({"lr": 4e-4, "batch": 16})
    assert a == b
    assert a != c


def test_config_hash_serilestirilemeyen_degeri_yutar():
    h = runstamp.config_hash({"fn": object(), "lr": 1e-3})
    assert isinstance(h, str) and len(h) == 12


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)
        for p in self.b.parameters():
            p.requires_grad_(False)


def test_agirlik_degisti_kontrolu_degisimi_yakalar():
    """M_enc4 dersi: 'encoder acildi' iddiasi ancak agirliklar gercekten
    degistiyse gecerlidir."""
    m = _Tiny()
    snap = runstamp.weight_snapshot(m)
    with torch.no_grad():
        m.a.weight.add_(0.5)
    d = runstamp.weight_delta(m, snap)
    assert d["a.weight"] > 0
    assert d["b.weight"] == 0.0
    assert runstamp.changed_count(d) == 1


def test_agirlik_degismezse_delta_sifir():
    m = _Tiny()
    snap = runstamp.weight_snapshot(m)
    d = runstamp.weight_delta(m, snap)
    assert runstamp.changed_count(d) == 0
    assert max(d.values()) == 0.0


def test_snapshot_sadece_istenen_on_eki_alir():
    m = _Tiny()
    snap = runstamp.weight_snapshot(m, prefix="a.")
    assert set(snap) == {"a.weight", "a.bias"}
