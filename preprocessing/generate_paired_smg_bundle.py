"""E1 -- Paired-test-instance Signer-Mixing-Gap (SMG) bundles.

The reviewer's objection to the original SMG figure: it compared a
signer-disjoint split against a signer-mixed split whose TEST sets were
different clips, so the gap conflated "signer seen in training" with "different
(easier) test instances". This script removes that confound by FREEZING one
focus signer's evaluation clips and only changing whether that signer's OTHER
clips appear in training. The two conditions are then evaluated on the
*identical* frozen test instances -> the accuracy difference isolates signer
familiarity alone.

For a focus signer F (one of the canonical TEST signers {2, 13}) we
deterministically, class-stratified split F's cached clips 50/50 into:
  FROZEN_TEST(F)  -- frozen evaluation set, used by EVERY condition below
  POOL(F)          -- F's remaining clips, eligible to enter training

Then we emit three train/val/test bundles that all TEST on FROZEN_TEST(F):

  unseen        train = 8 canonical train signers {1,4,5,6,8,9,11,12}
                (F entirely absent from training -> the true SI condition)
  seen_matched  class-stratified swap: start from `unseen`'s train, and for
                each class add POOL(F) clips while removing an equal number of
                other-signer clips of that class. Per-class train frequencies
                and total size stay identical to `unseen`; the ONLY difference
                is that some training clips now come from F.  <-- headline
  seen_aug      train = 8 canonical + all of POOL(F) (larger train; secondary,
                measures seen-signer + extra-data together)

Paired SMG (identical instances) = acc(seen_matched) - acc(unseen).

Val is always signer U15 (the canonical val signer), disjoint from every
train/test condition.  Reads the per-clip pose cache built once by
generate_signer_split_npy.py; no MediaPipe.

Usage (one focus signer):
    python preprocessing/generate_paired_smg_bundle.py \\
        --cache-dir data/bdsl_cache \\
        --focus-signer 2 \\
        --output-root data/bdsl_si_pairsmg

`tools/run_paired_smg.py` / the sbatch driver iterate the focus signers and
train BlockGCN under the shared recipe on each condition.
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
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.bdsl_signer_split import SIGNER_SPLIT, parse_filename  # noqa: E402

TRAIN_SIGNERS = list(SIGNER_SPLIT["train"])   # 1,4,5,6,8,9,11,12
VAL_SIGNER = SIGNER_SPLIT["val"][0]           # 15


def _scan_cache(cache_dir):
    """Yield (cache_path, class_name, signer, word, trial, hand) per cached clip."""
    for class_name in sorted(os.listdir(cache_dir)):
        cls_dir = os.path.join(cache_dir, class_name)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith(".npz"):
                continue
            parsed = parse_filename(fn.replace(".npz", ".mp4"))
            if parsed is None:
                continue
            signer, word, trial, hand = parsed
            yield (os.path.join(cls_dir, fn), class_name, signer, word, trial, hand)


def _save_split(clip_list, split_name, output_dir, class_to_idx,
                max_frames=300, num_channels=3, num_joints=27):
    """clip_list: list of (cache_path, class_name). Writes <split>_{data,label}."""
    if not clip_list:
        print(f"  [{split_name}] no clips, skipping")
        return 0
    n = len(clip_list)
    large = np.zeros((n, num_channels, max_frames, num_joints, 1), dtype=np.float32)
    names, labels = [], []
    for i, (cache_path, cls) in enumerate(tqdm(clip_list, desc=f"  [{split_name}]")):
        with np.load(cache_path) as h:
            data = h["data"]
        t = data.shape[1]
        large[i, :, :min(t, max_frames), :, :] = data[:, :max_frames, :, :]
        names.append(os.path.basename(cache_path).replace(".npz", ".mp4"))
        labels.append(class_to_idx[cls])
    np.save(os.path.join(output_dir, f"{split_name}_data.npy"), large)
    with open(os.path.join(output_dir, f"{split_name}_label.pkl"), "wb") as f:
        pickle.dump((names, labels), f)
    print(f"  [{split_name}] wrote {n} clips, shape={large.shape}")
    return n


def _write_condition(name, train_clips, val_clips, test_clips, output_root,
                     focus, class_to_idx, classes, dry_run):
    out_dir = os.path.join(output_root, f"U{focus:02d}_{name}")
    print(f"\ncondition '{name}' -> {out_dir}")
    print(f"  train={len(train_clips)}  val={len(val_clips)}  test={len(test_clips)}")
    if dry_run:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx,
                   "pairsmg": {"focus_signer": focus, "condition": name}}, f, indent=2)
    for split, clips in (("train", train_clips), ("val", val_clips), ("test", test_clips)):
        _save_split(clips, split, out_dir, class_to_idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/bdsl_cache")
    ap.add_argument("--output-root", default="data/bdsl_si_pairsmg")
    ap.add_argument("--focus-signer", type=int, required=True,
                    help="signer whose clips are frozen as the shared TEST set (2 or 13)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the matched-composition swap (test split is "
                         "deterministic class-stratified, independent of seed)")
    ap.add_argument("--conditions", nargs="+",
                    default=["unseen", "seen_matched", "seen_aug"],
                    choices=["unseen", "seen_matched", "seen_aug"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    focus = a.focus_signer
    if not os.path.isdir(a.cache_dir):
        raise FileNotFoundError(a.cache_dir)

    # Group cached clips.
    by_signer_class = defaultdict(lambda: defaultdict(list))  # signer -> class -> [path]
    class_set = set()
    for path, cls, signer, _w, _t, _h in _scan_cache(a.cache_dir):
        by_signer_class[signer][cls].append(path)
        class_set.add(cls)
    classes = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if focus not in by_signer_class:
        raise ValueError(f"focus signer U{focus:02d} not in cache")

    # Deterministic, class-stratified 50/50 split of the focus signer's clips.
    frozen_test, focus_pool = [], []          # each: (path, class)
    for cls in classes:
        paths = sorted(by_signer_class[focus].get(cls, []))  # stable order
        for i, p in enumerate(paths):
            (frozen_test if i % 2 == 0 else focus_pool).append((p, cls))
    print(f"focus U{focus:02d}: frozen_test={len(frozen_test)}  pool={len(focus_pool)}")

    # Base training pool = the 8 canonical train signers.
    base_train = []                            # (path, class)
    base_train_by_class = defaultdict(list)
    for s in TRAIN_SIGNERS:
        for cls in classes:
            for p in by_signer_class[s].get(cls, []):
                base_train.append((p, cls))
                base_train_by_class[cls].append((p, cls))
    val_clips = [(p, cls) for cls in classes
                 for p in by_signer_class[VAL_SIGNER].get(cls, [])]

    pool_by_class = defaultdict(list)
    for p, cls in focus_pool:
        pool_by_class[cls].append((p, cls))

    rng = np.random.RandomState(a.seed)

    conds = {}
    conds["unseen"] = list(base_train)
    conds["seen_aug"] = list(base_train) + list(focus_pool)

    # matched: per class, add pool(F) clips and drop an equal count of base clips.
    matched = []
    dropped = added = 0
    for cls in classes:
        base_cls = list(base_train_by_class[cls])
        add_cls = list(pool_by_class.get(cls, []))
        k = min(len(add_cls), len(base_cls))   # cannot drop more than we have
        if k < len(add_cls):
            print(f"  [warn] class '{cls}': only {len(base_cls)} base clips, "
                  f"can inject {k}/{len(add_cls)} focus clips (size still matched)")
        keep = list(base_cls)
        drop_idx = rng.choice(len(keep), size=k, replace=False) if k else []
        keep = [c for i, c in enumerate(keep) if i not in set(drop_idx)]
        matched.extend(keep + add_cls[:k])
        dropped += k
        added += k
    conds["seen_matched"] = matched
    print(f"seen_matched: swapped {added} focus clips in for {dropped} base clips "
          f"(|train|={len(matched)} vs unseen |train|={len(conds['unseen'])})")

    for name in a.conditions:
        _write_condition(name, conds[name], val_clips, frozen_test, a.output_root,
                         focus, class_to_idx, classes, a.dry_run)

    print("\ndone." + ("  (dry-run, nothing written)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
