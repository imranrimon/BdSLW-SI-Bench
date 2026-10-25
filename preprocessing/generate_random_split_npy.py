"""Signer-DEPENDENT (random) split bundler for BdSLW60 — the shortcut baseline.

Companion to `generate_signer_split_npy.py` (the signer-INDEPENDENT split). This
builds the *signer-dependent* counterpart used to quantify the identity shortcut:
the SAME labeled-signer clip population as SI, re-split RANDOMLY (stratified by
class, seed 42) so a signer's repetitions of a word land in both train and test.
The SI-vs-SD accuracy gap per model/representation IS the shortcut measurement.

Two representations, one shared clip->split assignment so pose and DINOv2 are
directly comparable:
  * pose  -> read per-clip .npz from data/bdsl_cache/<word>/<base>.npz  (no
    MediaPipe re-run; the cache already holds every clip).
  * DINOv2 -> remap the existing SI DINOv2 bundles (data/bdsl_si_dino) by
    clip basename (no feature re-extraction).

Usage:
  python preprocessing/generate_random_split_npy.py \
      --pose-cache data/bdsl_cache --out-pose data/bdsl_sd \
      --si-dino data/bdsl_si_dino --out-dino data/bdsl_sd_dino --seed 42
"""
import argparse
import json
import os
import pickle
import re
import time

import numpy as np

from preprocessing.bdsl_signer_split import SIGNER_SPLIT

# clip basename (no extension) -> signer id, e.g. "U4W213F_trial_2_R"
_BASE_RE = re.compile(r"^U(\d+)W(\d+)", re.IGNORECASE)

# same target sizes as the SI split (5748 / 655 / 1365 = 7768 labeled clips)
_FRAC = {"val": 655 / 7768.0, "test": 1365 / 7768.0}  # train = remainder

LABELED = set(SIGNER_SPLIT["train"]) | set(SIGNER_SPLIT["val"]) | set(SIGNER_SPLIT["test"])


def _signer_of(base):
    m = _BASE_RE.match(base)
    return int(m.group(1)) if m else None


def build_assignment(pose_cache, seed):
    """Return {clip_base: split} for a stratified random split over labeled signers."""
    classes = sorted(d for d in os.listdir(pose_cache)
                     if os.path.isdir(os.path.join(pose_cache, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)
    assignment = {}
    per_class = {}
    for cls in classes:
        cdir = os.path.join(pose_cache, cls)
        bases = [os.path.splitext(f)[0] for f in os.listdir(cdir) if f.endswith(".npz")]
        bases = [b for b in bases if _signer_of(b) in LABELED]
        rng.shuffle(bases)
        n = len(bases)
        n_val = int(round(n * _FRAC["val"]))
        n_test = int(round(n * _FRAC["test"]))
        # guarantee each class present in every split
        n_val = max(1, n_val); n_test = max(1, n_test)
        val = bases[:n_val]; test = bases[n_val:n_val + n_test]; train = bases[n_val + n_test:]
        for b in train: assignment[b] = "train"
        for b in val: assignment[b] = "val"
        for b in test: assignment[b] = "test"
        per_class[cls] = (len(train), len(val), len(test))
    return classes, class_to_idx, assignment, per_class


def _save(results, split, out_dir, max_frames=300, C=3, V=27):
    n = len(results)
    large = np.zeros((n, C, max_frames, V, 1), dtype=np.float32)
    names, labels = [], []
    for i, (data, label, base) in enumerate(results):
        t = data.shape[1]
        large[i, :, :min(t, max_frames), :, :] = data[:, :max_frames, :, :]
        names.append(base); labels.append(label)
    np.save(os.path.join(out_dir, f"{split}_data.npy"), large)
    with open(os.path.join(out_dir, f"{split}_label.pkl"), "wb") as f:
        pickle.dump((names, labels), f)
    print(f"[pose:{split}] wrote {large.shape}")


def _load_npz_data(path, retries=5):
    """np.load with retry on transient lustre I/O errors; raises if persistently unreadable."""
    for i in range(retries):
        try:
            with np.load(path) as h:
                return np.array(h["data"])
        except OSError:
            if i == retries - 1:
                raise  # persistent I/O error: fail loudly rather than drop the clip
            time.sleep(2 * (i + 1))
    return None


def bundle_pose(pose_cache, out_dir, classes, class_to_idx, assignment):
    os.makedirs(out_dir, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    skipped = 0
    for cls in classes:
        cdir = os.path.join(pose_cache, cls)
        for f in os.listdir(cdir):
            if not f.endswith(".npz"):
                continue
            base = os.path.splitext(f)[0]
            split = assignment.get(base)
            if split is None:
                continue
            data = _load_npz_data(os.path.join(cdir, f))
            if data is None:
                skipped += 1
                print(f"[pose] WARN unreadable, skipped: {cls}/{f}")
                continue
            buckets[split].append((data, class_to_idx[cls], base))
    if skipped:
        print(f"[pose] total skipped (unreadable): {skipped}")
    for split in ("train", "val", "test"):
        _save(buckets[split], split, out_dir)
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)


def _norm(name):
    return os.path.splitext(os.path.basename(str(name)))[0]


def remap_dino(si_dino, out_dir, assignment):
    """Re-split the existing SI DINOv2 bundles by clip basename (no re-extraction)."""
    os.makedirs(out_dir, exist_ok=True)
    feats, names, labels = [], [], []
    for s in ("train", "val", "test"):
        dp = os.path.join(si_dino, f"{s}_data.npy")
        lp = os.path.join(si_dino, f"{s}_label.pkl")
        if not (os.path.exists(dp) and os.path.exists(lp)):
            print(f"[dino] missing {s}; skip dino remap"); return
        d = np.load(dp)
        with open(lp, "rb") as f:
            nm, lb = pickle.load(f)
        feats.append(d); names += list(nm); labels += list(lb)
    feats = np.concatenate(feats, 0)
    buckets = {"train": [], "val": [], "test": []}
    miss = 0
    for i, nm in enumerate(names):
        split = assignment.get(_norm(nm))
        if split is None:
            miss += 1; continue
        buckets[split].append(i)
    for split, idx in buckets.items():
        idx = np.array(idx, dtype=np.int64)
        np.save(os.path.join(out_dir, f"{split}_data.npy"), feats[idx])
        with open(os.path.join(out_dir, f"{split}_label.pkl"), "wb") as f:
            pickle.dump(([names[i] for i in idx], [labels[i] for i in idx]), f)
        print(f"[dino:{split}] wrote {feats[idx].shape}")
    if os.path.exists(os.path.join(si_dino, "classes.json")):
        import shutil
        shutil.copy(os.path.join(si_dino, "classes.json"), os.path.join(out_dir, "classes.json"))
    print(f"[dino] unmatched clips (not in SD assignment): {miss}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-cache", default="data/bdsl_cache")
    ap.add_argument("--out-pose", default="data/bdsl_sd")
    ap.add_argument("--si-dino", default="data/bdsl_si_dino")
    ap.add_argument("--out-dino", default="data/bdsl_sd_dino")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dino", action="store_true")
    args = ap.parse_args()

    classes, c2i, assignment, per_class = build_assignment(args.pose_cache, args.seed)
    n = {"train": 0, "val": 0, "test": 0}
    for v in assignment.values():
        n[v] += 1
    print(f"random SD split: {len(classes)} classes | train={n['train']} val={n['val']} test={n['test']} "
          f"(total {sum(n.values())})")
    bundle_pose(args.pose_cache, args.out_pose, classes, c2i, assignment)
    if not args.no_dino:
        remap_dino(args.si_dino, args.out_dino, assignment)
    print("done.")


if __name__ == "__main__":
    main()
