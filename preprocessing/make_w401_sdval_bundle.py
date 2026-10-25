#!/usr/bin/env python
"""Build a signer-disjoint-validation BdSLW401 bundle (data/bdslw401_si_sdval).

The canonical data/bdslw401_si split keeps test signers S04/S08 held out, but its
validation set is drawn from the 16 training signers, so LR and checkpoint
selection happen on seen signers. This bundle holds out one full-coverage
training signer (S15 by default: the highest-numbered full-coverage signer)
entirely for validation:

  val   = every S15 clip from the old train + val
  train = old train + val minus S15 (15 signers)
  test  = unchanged S04/S08 (symlinked to the canonical files)

Clip names, label format ((names, labels) pickle) and array layout
(N, C, T, V, M) float32 are unchanged. Arrays are copied through memmaps in
chunks, so peak RAM stays small. Writes splits/bdslw401_sdval.json.

Usage:
  python preprocessing/make_w401_sdval_bundle.py [--val-signer 15]
"""
import argparse
import collections
import json
import os
import pickle
import re
import shutil

import numpy as np

SRC = "data/bdslw401_si"
DST = "data/bdslw401_si_sdval"
SIG = re.compile(r"S(\d+)")
CHUNK = 1024


def signer(name):
    return int(SIG.search(name).group(1))


def load(split):
    names, labels = pickle.load(open(os.path.join(SRC, f"{split}_label.pkl"), "rb"))
    data = np.load(os.path.join(SRC, f"{split}_data.npy"), mmap_mode="r")
    assert len(names) == len(labels) == data.shape[0], split
    return list(names), list(labels), data


def write(split, picks, srcs):
    """picks: list of (source_index_in_srcs, row). Copies rows in order via memmap."""
    shape = (len(picks),) + srcs[0].shape[1:]
    out = np.lib.format.open_memmap(os.path.join(DST, f"{split}_data.npy"), mode="w+",
                                    dtype=srcs[0].dtype, shape=shape)
    for start in range(0, len(picks), CHUNK):
        block = picks[start:start + CHUNK]
        for k, (s, r) in enumerate(block):
            out[start + k] = srcs[s][r]
    out.flush()
    del out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-signer", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(DST, exist_ok=True)

    tr_n, tr_y, tr_x = load("train")
    va_n, va_y, va_x = load("val")
    te_n, te_y, _ = load("test")
    test_signers = sorted({signer(n) for n in te_n})

    pool = [(0, i, n, y) for i, (n, y) in enumerate(zip(tr_n, tr_y))] + \
           [(1, i, n, y) for i, (n, y) in enumerate(zip(va_n, va_y))]
    new_tr = [p for p in pool if signer(p[2]) != a.val_signer]
    new_va = [p for p in pool if signer(p[2]) == a.val_signer]

    tr_sig = {signer(p[2]) for p in new_tr}
    assert a.val_signer not in tr_sig, "val signer leaked into train"
    assert not (tr_sig & set(test_signers)), "test signer leaked into train"
    assert {signer(p[2]) for p in new_va} == {a.val_signer}
    assert len({p[2] for p in new_tr} & {p[2] for p in new_va}) == 0, "clip overlap train/val"
    assert len(new_tr) + len(new_va) == len(tr_n) + len(va_n)

    srcs = [tr_x, va_x]
    for split, rows in (("train", new_tr), ("val", new_va)):
        write(split, [(s, r) for s, r, _, _ in rows], srcs)
        pickle.dump(([p[2] for p in rows], [p[3] for p in rows]),
                    open(os.path.join(DST, f"{split}_label.pkl"), "wb"))
        print(f"{split}: {len(rows)} clips")

    for f in ("test_data.npy", "test_label.pkl"):
        dst = os.path.join(DST, f)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(os.path.join(SRC, f), DST), dst)
    for f in ("classes.json", "classes_romanized.json"):
        if os.path.exists(os.path.join(SRC, f)):
            shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))

    def counts(names):
        c = collections.Counter(signer(n) for n in names)
        return {f"S{k:02d}": v for k, v in sorted(c.items())}

    meta = {
        "source": SRC, "bundle": DST, "val_signer": f"S{a.val_signer:02d}",
        "rule": "val = all clips of one held-out full-coverage training signer "
                "(highest-numbered full-coverage signer); train = remaining training "
                "signers (old train+val); test unchanged",
        "train_signers": sorted(counts([p[2] for p in new_tr])),
        "val_signers": [f"S{a.val_signer:02d}"],
        "test_signers": [f"S{s:02d}" for s in test_signers],
        "counts": {"train": len(new_tr), "val": len(new_va), "test": len(te_n)},
        "per_signer": {"train": counts([p[2] for p in new_tr]),
                       "val": counts([p[2] for p in new_va]), "test": counts(te_n)},
        "num_classes": len(set(tr_y) | set(va_y)),
        "builder": "preprocessing/make_w401_sdval_bundle.py",
    }
    os.makedirs("splits", exist_ok=True)
    json.dump(meta, open("splits/bdslw401_sdval.json", "w"), indent=2)
    print(json.dumps(meta["counts"]), "val signer", meta["val_signer"])


if __name__ == "__main__":
    main()
