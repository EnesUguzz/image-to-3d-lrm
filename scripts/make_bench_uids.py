"""Sabit tezgah uid kumelerini uretir (bir kez calisir, cikti REPOYA COMMIT EDILIR).

NEDEN VAR (PROJE-DEVIR-BELGESI 7.1):
32-obje kapisi HANGI 32 objeye asiri duyarli -- ayni veri, ayni ayar, farkli alt
kume: top-1 %75 / %19 / %3. `--n_obj 32` ile "listeden adimlayarak" secmek
train_list her yeniden uretildiginde farkli objeler veriyor, yani gecmis kosularla
karsilastirilamiyor. Cozum: kumeleri BIR KEZ uret, dosyaya yaz, commit et,
her A/B kolunu `--uids` ile bunlara sabitle.

TASARIM:
- 32'lik kume = dataset/_ab_uids.json (AB_YENI kosusunda 20.11 dB / top-1 %75
  ile COKMEDIGI dogrulanmis kume; M_base'in coktugu kume degil).
- Kumeler IC ICE: 32 subset 250 subset 1024. Boylece ayni tarife iki olcekte
  kosuldugunda fark gercekten olcekten gelir, obje degisiminden degil.
- Sadece train split'ten secilir (val/test'e dokunulmaz).
- Secim deterministik (sabit seed + sirali liste).

Kullanim:
    python scripts/make_bench_uids.py --train_list dataset/train_list_opp_score3.json \
        --renders_dir dataset/renders_opp_score3
"""
import argparse
import json
import os
import random

SIZES = (32, 250, 1024)


def build_nested(all_uids, seed_uids, sizes=SIZES, seed=1337):
    """Ic ice kumeler dondurur: seed_uids en kucugun cekirdegi olur."""
    pool = sorted(set(all_uids))
    seed_list = [u for u in seed_uids if u in set(pool)]
    rest = [u for u in pool if u not in set(seed_list)]
    random.Random(seed).shuffle(rest)
    ordered = seed_list + rest
    out = {}
    for n in sizes:
        if n > len(ordered):
            raise ValueError(f"{n} obje istendi ama havuzda {len(ordered)} var")
        out[n] = ordered[:n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default="dataset/train_list_opp_score3.json")
    ap.add_argument("--renders_dir", default="dataset/renders_opp_score3")
    ap.add_argument("--seed_uids", default="dataset/_ab_uids.json")
    ap.add_argument("--out_dir", default="dataset")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    with open(a.train_list, encoding="utf-8") as f:
        tl = json.load(f)
    train = tl["train"]

    seed_uids = []
    if os.path.isfile(a.seed_uids):
        with open(a.seed_uids, encoding="utf-8") as f:
            s = json.load(f)
        seed_uids = s if isinstance(s, list) else s.get("train", [])

    # render'i gercekten diskte olanlar (yarim kalmis obje tezgahi bozmasin)
    def ok(u):
        return os.path.isfile(os.path.join(a.renders_dir, u, "meta.json"))

    train = [u for u in train if ok(u)]
    seed_uids = [u for u in seed_uids if ok(u)]
    print(f"havuz: {len(train)} train objesi (meta.json dogrulandi), "
          f"cekirdek: {len(seed_uids)} uid")

    sets = build_nested(train, seed_uids, seed=a.seed)
    for n, uids in sets.items():
        path = os.path.join(a.out_dir, f"bench_uids_{n}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"n": n, "seed": a.seed, "source": a.train_list,
                       "renders_dir": a.renders_dir, "uids": uids}, f, indent=1)
        print(f"  {path}: {len(uids)} uid")

    # ic ice olma garantisi
    s32, s250, s1024 = (set(sets[n]) for n in SIZES)
    assert s32 <= s250 <= s1024, "kumeler ic ice degil"
    print("ic ice dogrulandi: 32 subset 250 subset 1024")


if __name__ == "__main__":
    main()
