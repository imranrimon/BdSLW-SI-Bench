#!/usr/bin/env python
"""Relative-Quantization (RQ) re-encoding of a 27-keypoint BdSL bundle, ported to
our skeleton from the BdSLW60 dataset paper (Rubaiyeat et al., MTAP 2025, Sec. 4.1.8;
authors' code github.com/hasanssl/BdSLW60_Code, PREPROCESSING_RQ/*).

Per clip (C=3: x, y, confidence; T; V=27; M=1), on valid frames only:
  1. Relative translation to the physiological parent (authors' rule; refs taken
     from the untranslated coordinates; a missing point, all channels <= 0, is
     left untouched):
        nose -> shoulder midpoint; elbow -> same-side shoulder;
        wrist -> same-side elbow; every hand point -> same-side pose wrist.
     Shoulders keep "global" coordinates calibrated to the first valid frame's
     shoulder midpoint (the authors' calibration step).
     Channel 2 is MediaPipe confidence here (the authors' third channel was depth),
     so it is not translated -- only quantized.
  2. Per-feature min-max normalisation with statistics fit on the TRAIN split only
     (the authors fit on train+test; fixed here), clipped to [0, 1].
  3. Quantisation int(v * level): hands/elbows/wrists (10, 10, 5) levels,
     nose/shoulders (5, 5, 3) -- the authors' levels.
Zero-padding after the valid length stays exactly 0 so the feeder's
valid-length crop is unchanged.

Usage:
  python preprocessing/make_rq_bundle.py --src data/bdsl_si --dst data/bdsl_si_rq
"""
import argparse, os, shutil, sys
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from feeders.tools import valid_frame_length  # noqa: E402

# 27-node layout: 0 nose, 1 R-shoulder, 2 L-shoulder, 3 R-elbow, 4 L-elbow,
# 5 R-wrist, 6 L-wrist, 7-16 right hand (root 7), 17-26 left hand (root 17).
PARENT = {0: None, 1: None, 2: None, 3: 1, 4: 2, 5: 3, 6: 4}
PARENT.update({v: 5 for v in range(7, 17)})
PARENT.update({v: 6 for v in range(17, 27)})
LEVELS = np.ones((3, 27), dtype=np.float32)
LEVELS[:, :3] = np.array([5, 5, 3], dtype=np.float32)[:, None]
LEVELS[:, 3:] = np.array([10, 10, 5], dtype=np.float32)[:, None]


def relative(clip):
    """clip (3, T, 27) valid frames only -> relative-translated copy."""
    out = clip.copy()
    xy = clip[:2]
    missing = (clip <= 0).all(axis=0)                       # (T, 27)
    mid = 0.5 * (xy[:, :, 1] + xy[:, :, 2])                 # (2, T)
    both = ~missing[:, 1] & ~missing[:, 2]
    ms0 = mid[:, np.argmax(both)] if both.any() else np.zeros(2, np.float32)
    for v in range(27):
        if v in (1, 2):
            ref = np.broadcast_to(ms0[:, None], (2, clip.shape[1]))
        elif v == 0:
            ref = mid
        else:
            ref = xy[:, :, PARENT[v]]
        keep = ~missing[:, v]
        out[:2, keep, v] = xy[:, keep, v] - ref[:, keep]
    return out


def load(src, split):
    return np.load(os.path.join(src, f"{split}_data.npy"), mmap_mode="r")


def encode_split(data, lo, hi, chunk=256):
    N = data.shape[0]
    out = np.zeros(data.shape, dtype=np.float32)
    rng = (hi - lo) + 1e-5
    for s in range(0, N, chunk):
        blk = np.asarray(data[s:s + chunk], dtype=np.float32)
        for i in range(blk.shape[0]):
            L = valid_frame_length(blk[i])
            c = relative(blk[i, :, :L, :, 0])                 # (3, L, 27)
            q = np.floor(np.clip((c - lo[:, None, :]) / rng[:, None, :], 0.0, 1.0) * LEVELS[:, None, :])
            out[s + i, :, :L, :, 0] = q
    return out


def fit_minmax(data, chunk=256):
    lo = np.full((3, 27), np.inf, np.float32); hi = np.full((3, 27), -np.inf, np.float32)
    for s in range(0, data.shape[0], chunk):
        blk = np.asarray(data[s:s + chunk], dtype=np.float32)
        for i in range(blk.shape[0]):
            L = valid_frame_length(blk[i])
            c = relative(blk[i, :, :L, :, 0])
            lo = np.minimum(lo, c.min(axis=1)); hi = np.maximum(hi, c.max(axis=1))
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    lo, hi = fit_minmax(load(a.src, "train"))
    np.savez(os.path.join(a.dst, "rq_minmax_train.npz"), lo=lo, hi=hi, levels=LEVELS)
    for sp in a.splits:
        enc = encode_split(load(a.src, sp), lo, hi)
        tmp = os.path.join(a.dst, f"{sp}_data.tmp.npy")
        np.save(tmp, enc); os.replace(tmp, os.path.join(a.dst, f"{sp}_data.npy"))
        shutil.copyfile(os.path.join(a.src, f"{sp}_label.pkl"), os.path.join(a.dst, f"{sp}_label.pkl"))
        nz = enc[:, :, :, :, 0]
        print(f"[rq] {sp}: {enc.shape}  value range [{nz.min():.0f}, {nz.max():.0f}]")
    if os.path.exists(os.path.join(a.src, "classes.json")):
        shutil.copyfile(os.path.join(a.src, "classes.json"), os.path.join(a.dst, "classes.json"))
    open(os.path.join(a.dst, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
