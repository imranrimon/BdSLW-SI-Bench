"""Build the participant-MIXED (random) ASL Citizen split for the Signer-Mixing
Gap (SMG) diagnostic.

Companion to the OFFICIAL participant-disjoint bundle (data/aslcitizen_si). This
takes the SAME clip population (train+val+test bundles concatenated) and re-splits
it RANDOMLY, stratified by class (fixed seed), so a participant's videos land in
both train and test. The participant-disjoint minus participant-mixed Top-1 gap
IS the Signer-Mixing Gap — showing the signer-identity shortcut reproduces on ASL
Citizen, exactly as it does on BdSLW60.

No video access / MediaPipe: operates purely on the already-extracted NPY bundles.
Preserves the <user>__<basename> sample names, so per-signer analysis still works.

Usage:
  python preprocessing/generate_aslcitizen_mixed_split.py \
      --si-dir data/aslcitizen_si --out-dir data/aslcitizen_mixed --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np


def _load(si_dir, split):
    data = np.load(os.path.join(si_dir, "{}_data.npy".format(split)), mmap_mode="r")
    with open(os.path.join(si_dir, "{}_label.pkl".format(split)), "rb") as f:
        names, labels = pickle.load(f)
    return data, names, list(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--si-dir", default="data/aslcitizen_si")
    ap.add_argument("--out-dir", default="data/aslcitizen_mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="which SI bundles to pool before re-splitting")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) pool every clip from the SI bundles.
    arrays, all_names, all_labels = [], [], []
    for s in args.splits:
        d, names, labels = _load(args.si_dir, s)
        arrays.append(np.asarray(d))
        all_names.extend(names)
        all_labels.extend(labels)
        print("[pool] {}: {} clips".format(s, len(names)))
    data = np.concatenate(arrays, axis=0)
    labels = np.asarray(all_labels)
    n = len(all_names)
    print("[pool] total {} clips, shape {}".format(n, data.shape))

    # target sizes match the OFFICIAL split proportions (train/val/test).
    sizes = {s: len(_load(args.si_dir, s)[1]) for s in ["train", "val", "test"]}
    frac_val = sizes["val"] / float(sum(sizes.values()))
    frac_test = sizes["test"] / float(sum(sizes.values()))

    # 2) class-stratified random assignment (fixed seed) -> participant-mixed.
    rng = np.random.RandomState(args.seed)
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    assign = {}  # idx -> split
    for y, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_c = len(idxs)
        n_val = int(round(n_c * frac_val))
        n_test = int(round(n_c * frac_test))
        # ensure train non-empty for tiny classes
        n_val = min(n_val, max(0, n_c - 1))
        n_test = min(n_test, max(0, n_c - 1 - n_val))
        for j, idx in enumerate(idxs):
            if j < n_val:
                assign[idx] = "val"
            elif j < n_val + n_test:
                assign[idx] = "test"
            else:
                assign[idx] = "train"

    # 3) write bundles.
    buckets = defaultdict(list)
    for i in range(n):
        buckets[assign[i]].append(i)
    # copy the class map for downstream label alignment.
    src_classes = os.path.join(args.si_dir, "classes.json")
    if os.path.exists(src_classes):
        with open(src_classes) as f:
            cj = json.load(f)
        with open(os.path.join(args.out_dir, "classes.json"), "w") as f:
            json.dump(cj, f, indent=2)

    stats = {}
    for s in ["train", "val", "test"]:
        idxs = np.array(sorted(buckets[s]), dtype=np.int64)
        sub = np.ascontiguousarray(data[idxs])
        names = [all_names[i] for i in idxs]
        labs = [int(labels[i]) for i in idxs]
        np.save(os.path.join(args.out_dir, "{}_data.npy".format(s)), sub)
        with open(os.path.join(args.out_dir, "{}_label.pkl".format(s)), "wb") as f:
            pickle.dump((names, labs), f)
        # participant leakage stat: users appearing in multiple splits (expected here).
        users = set(nm.split("__")[0] for nm in names)
        stats[s] = {"clips": len(idxs), "n_users": len(users)}
        print("[mixed] {}: {} clips, {} users".format(s, len(idxs), len(users)))

    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump({"note": "participant-MIXED random split (seed {}) for SMG".format(args.seed),
                   "seed": args.seed, "per_split": stats}, f, indent=2)
    print("wrote participant-mixed bundle to", args.out_dir)


if __name__ == "__main__":
    main()
