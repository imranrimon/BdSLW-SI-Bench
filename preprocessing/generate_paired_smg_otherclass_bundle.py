"""Paired-SMG decomposition: focus signer seen ONLY through OTHER classes (P2 fix, 2026-10).

generate_paired_smg_bundle.py's `seen_matched` puts half of the focus signer's clips of
EVERY class into training and tests on the other half. In BdSLW60 each signer x class is
ONE recording of ~12 trials, so the training clips include sibling trials of the very
recording being tested: the paired gap mixes signer familiarity with same-recording
(same-class) familiarity.

This script builds the cleaner condition. The 60 classes are split into two fixed halves
H in {A, B} (seeded permutation of the sorted class list). For `seenother_<H>`:
  train = the 8 canonical train signers, with a class-stratified matched swap that injects
          ALL focus-signer clips of the classes in H (dropping an equal number of
          other-signer clips of the same class, so per-class counts and |train| equal
          `unseen`'s; ~the same number of injected focus clips as `seen_matched`);
  test  = the IDENTICAL frozen focus clips of generate_paired_smg_bundle.py (even-index rule)
          restricted to the classes NOT in H;
  val   = signer U15.
So the focus signer is seen in training, but never in the tested class / recording. The
union of seenother_A's and seenother_B's test sets is exactly the original frozen set, so
the per-clip comparison with `unseen` and `seen_matched` (same frozen clips) is exact:
    paired gap = [seenother - unseen]  (signer familiarity via other classes)
               + [seen_matched - seenother]  (same-class / same-recording familiarity)

  python preprocessing/generate_paired_smg_otherclass_bundle.py --focus-signer 2 \
      --output-root data/bdsl_si_pairsmg_seenother
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.generate_paired_smg_bundle import (  # noqa: E402
    TRAIN_SIGNERS, VAL_SIGNER, _scan_cache, _save_split)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/bdsl_cache")
    ap.add_argument("--output-root", default="data/bdsl_si_pairsmg_seenother")
    ap.add_argument("--focus-signer", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0, help="matched-swap RNG (as the original)")
    ap.add_argument("--half-seed", type=int, default=0, help="class-half permutation RNG")
    ap.add_argument("--check-against", default="data/bdsl_si_pairsmg",
                    help="original paired bundles; the frozen test set must match U<F>_unseen/test exactly")
    a = ap.parse_args()
    focus = a.focus_signer

    by_signer_class = defaultdict(lambda: defaultdict(list))
    class_set = set()
    for path, cls, signer, _w, _t, _h in _scan_cache(a.cache_dir):
        by_signer_class[signer][cls].append(path)
        class_set.add(cls)
    classes = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    frozen_test, focus_all_by_class = [], defaultdict(list)
    for cls in classes:
        paths = sorted(by_signer_class[focus].get(cls, []))
        for i, p in enumerate(paths):
            if i % 2 == 0:
                frozen_test.append((p, cls))
            focus_all_by_class[cls].append((p, cls))

    # sanity: identical frozen test instances to the original paired bundle
    ref = Path(a.check_against) / f"U{focus:02d}_unseen" / "test_label.pkl"
    if ref.exists():
        ref_names, _ = pickle.load(open(ref, "rb"))
        mine = [os.path.basename(p).replace(".npz", ".mp4") for p, _ in frozen_test]
        if sorted(ref_names) != sorted(mine):
            raise SystemExit(f"frozen test set differs from {ref}")
        print(f"frozen test set == {ref} ({len(mine)} clips)")

    base_by_class = defaultdict(list)
    for s in TRAIN_SIGNERS:
        for cls in classes:
            base_by_class[cls].extend((p, cls) for p in by_signer_class[s].get(cls, []))
    val_clips = [(p, cls) for cls in classes for p in by_signer_class[VAL_SIGNER].get(cls, [])]

    perm = np.random.RandomState(a.half_seed).permutation(len(classes))
    halves = {"A": sorted(classes[i] for i in perm[: len(classes) // 2]),
              "B": sorted(classes[i] for i in perm[len(classes) // 2:])}

    for h, seen_classes in halves.items():
        rng = np.random.RandomState(a.seed)
        seen = set(seen_classes)
        train, injected = [], 0
        for cls in classes:
            base = list(base_by_class[cls])
            if cls in seen:
                add = list(focus_all_by_class.get(cls, []))
                k = min(len(add), len(base))
                drop = set(rng.choice(len(base), size=k, replace=False).tolist()) if k else set()
                base = [c for i, c in enumerate(base) if i not in drop] + add[:k]
                injected += k
            train.extend(base)
        test = [(p, cls) for p, cls in frozen_test if cls not in seen]
        out = os.path.join(a.output_root, f"U{focus:02d}_seenother_{h}")
        os.makedirs(out, exist_ok=True)
        print(f"\nU{focus:02d}_seenother_{h}: focus seen in {len(seen)} classes ({injected} clips injected); "
              f"train={len(train)} val={len(val_clips)} test={len(test)} (frozen clips of the other "
              f"{len(classes) - len(seen)} classes)")
        with open(os.path.join(out, "classes.json"), "w") as f:
            json.dump({"classes": classes, "class_to_idx": class_to_idx,
                       "pairsmg_otherclass": {"focus_signer": focus, "half": h,
                                              "focus_seen_classes": seen_classes,
                                              "test_classes": [c for c in classes if c not in seen],
                                              "injected_focus_clips": injected,
                                              "half_seed": a.half_seed, "swap_seed": a.seed}}, f, indent=2)
        for split, clips in (("train", train), ("val", val_clips), ("test", test)):
            _save_split(clips, split, out, class_to_idx)


if __name__ == "__main__":
    main()
