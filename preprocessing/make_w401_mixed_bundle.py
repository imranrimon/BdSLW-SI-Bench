"""Signer-MIXED (signer-dependent) split for BdSLW401 — the SMG counterpart of
data/bdslw401_si, mirroring how data/bdsl_sd was built for BdSLW60
(preprocessing/generate_random_split_npy.py).

The SAME clip pool as the SI bundle (train+val+test of data/bdslw401_si, 51,098
front-view clips, 401 classes) is re-split RANDOMLY, stratified by class (seed 42),
to the SI split's proportions, so a signer's repetitions of a word land on both
sides — including the SI test signers S04/S08. Reads the existing SI bundles
(no pose-cache re-read), so preprocessing is byte-identical to the SI run.

Usage:
  python preprocessing/make_w401_mixed_bundle.py --si data/bdslw401_si --out data/bdslw401_sd --seed 42
"""
import argparse
import json
import os
import pickle
import shutil

import numpy as np

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--si", default="data/bdslw401_si")
    ap.add_argument("--out", default="data/bdslw401_sd")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()

    arrs, src = {}, []          # src: (split, row, name, label) over the whole pool
    for s in SPLITS:
        arrs[s] = np.load(os.path.join(a.si, f"{s}_data.npy"), mmap_mode="r")
        names, labels = pickle.load(open(os.path.join(a.si, f"{s}_label.pkl"), "rb"))
        assert len(names) == arrs[s].shape[0], s
        src += [(s, i, n, int(l)) for i, (n, l) in enumerate(zip(names, labels))]
    n_si = {s: arrs[s].shape[0] for s in SPLITS}
    total = sum(n_si.values())
    frac = {"val": n_si["val"] / total, "test": n_si["test"] / total}

    rng = np.random.default_rng(a.seed)
    by_cls = {}
    for k, (_, _, _, l) in enumerate(src):
        by_cls.setdefault(l, []).append(k)
    assign = {s: [] for s in SPLITS}
    for l in sorted(by_cls):
        idx = by_cls[l]
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * frac["val"])))
        n_test = max(1, int(round(n * frac["test"])))
        assign["val"] += idx[:n_val]
        assign["test"] += idx[n_val:n_val + n_test]
        assign["train"] += idx[n_val + n_test:]

    os.makedirs(a.out, exist_ok=True)
    shape = arrs["train"].shape[1:]
    for s in SPLITS:
        ks = sorted(assign[s])
        out = np.lib.format.open_memmap(os.path.join(a.out, f"{s}_data.npy"), mode="w+",
                                        dtype=np.float32, shape=(len(ks),) + shape)
        for c in range(0, len(ks), a.chunk):
            for j, k in enumerate(ks[c:c + a.chunk]):
                sp, row, _, _ = src[k]
                out[c + j] = arrs[sp][row]
        out.flush(); del out
        pickle.dump(([src[k][2] for k in ks], [src[k][3] for k in ks]),
                    open(os.path.join(a.out, f"{s}_label.pkl"), "wb"))
        print(f"[{s}] {len(ks)} clips (SI {n_si[s]})")
    for f in ("classes.json", "classes_romanized.json"):
        if os.path.exists(os.path.join(a.si, f)):
            shutil.copy(os.path.join(a.si, f), os.path.join(a.out, f))
    json.dump({"seed": a.seed, "source": a.si, "sizes": {s: len(assign[s]) for s in SPLITS},
               "si_sizes": n_si}, open(os.path.join(a.out, "split_meta.json"), "w"), indent=1)
    open(os.path.join(a.out, ".READY"), "w").write("ok\n")
    print("done.")


if __name__ == "__main__":
    main()
