"""P2 Tier-2 — causal cue ablations on pose (counterfactual transforms of BdSLW60-SI).

Creates transformed copies of the canonical SI bundles so each transform isolates one
cue. Training A_SI on each (and the signer-decodability probe) separates *which* cue
carries recognition signal vs signer information. 27-node skeleton = 7 body + 20 hand
(indices below); it has no face landmarks (a documented limitation).

Cues:
  static   : replicate the middle valid frame across time  (kills motion, keeps pose)
  shuffle  : randomly permute frames within each clip       (kills trajectory, keeps static identity)
  bodyonly : zero the 20 hand joints                        (posture/morphology only)
  handsonly: zero the 7 body joints                         (manual articulation only)
  rootcenter: subtract the root (joint 0) each frame        (removes global position)
  speednorm: divide per-frame displacement by its clip-mean speed (removes signing tempo)

Usage: python preprocessing/make_cue_bundle.py --cue static --out data/bdsl_cue_static
"""
import argparse, os, pickle
import numpy as np

SRC = "data/bdsl_si"
BODY = list(range(7))          # 7 body joints
HANDS = list(range(7, 27))     # 20 hand joints (10 per hand)


def valid_mask(clip):  # clip: (C,T,V,M) -> (T,) nonzero frames
    return np.abs(clip).sum(axis=(0, 2, 3)) > 0


def transform(clip, cue, rng):
    c = clip.copy()
    C, T, V, M = c.shape
    vm = valid_mask(c); idx = np.where(vm)[0]
    if len(idx) == 0:
        return c
    if cue == "static":
        mid = idx[len(idx) // 2]
        for t in idx:
            c[:, t] = clip[:, mid]
    elif cue == "shuffle":
        perm = rng.permutation(idx)
        c[:, idx] = clip[:, perm]
    elif cue == "bodyonly":
        c[:, :, HANDS, :] = 0
    elif cue == "handsonly":
        c[:, :, BODY, :] = 0
    elif cue == "rootcenter":
        root = clip[:2, :, 0:1, :]                 # x,y of joint 0
        c[:2] = clip[:2] - root
    elif cue == "speednorm":
        d = np.diff(clip[:2, idx], axis=1)         # (2, len-1, V, M)
        sp = np.linalg.norm(d, axis=0).mean() + 1e-6
        c[:2, idx] = clip[:2, idx] / sp
    else:
        raise ValueError(cue)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    for sp in ("train", "val", "test"):
        X = np.load(f"{SRC}/{sp}_data.npy", mmap_mode="r")
        nm, lb = pickle.load(open(f"{SRC}/{sp}_label.pkl", "rb"))
        out = np.lib.format.open_memmap(f"{a.out}/{sp}_data.npy", mode="w+",
                                        dtype=np.float32, shape=X.shape)
        for i in range(X.shape[0]):
            out[i] = transform(np.asarray(X[i]), a.cue, rng)
        out.flush(); del out
        pickle.dump((nm, lb), open(f"{a.out}/{sp}_label.pkl", "wb"))
        print(f"  {a.cue} {sp}: {X.shape[0]} clips")
    print(f"DONE cue={a.cue} -> {a.out}")


if __name__ == "__main__":
    main()
