"""Missing-joint-aware versions of two P2 cue perturbations (fix, 2026-10).

preprocessing/make_cue_bundle.py treats MediaPipe's undetected joints, stored as
(x, y) = (0, 0), as real coordinates:
  speednorm : the clip's mean frame-to-frame speed includes 0 <-> coordinate jumps
              whenever a hand drops in or out (median ~69% of the "speed" estimate);
  rootcenter: a missing joint becomes (0,0) - root = -root, a nonzero fake position.

Masked versions (everything else identical to make_cue_bundle.py):
  speednorm_masked : speed is averaged only over (joint, t->t+1) pairs of consecutive
                     valid frames where that joint is detected in BOTH frames; the clip's
                     x,y are then divided by it exactly as before (missing stays 0).
  rootcenter_masked: subtract the root (joint 0) x,y per frame from DETECTED joints only;
                     missing joints stay (0,0).
A joint is missing iff x == y == 0. Writes a new bundle dir; never touches data/bdsl_si.

  python preprocessing/make_cue_bundle_masked.py --cue speednorm_masked --out data/bdsl_cue_speednorm_masked
"""
import argparse, os, pickle
import numpy as np

SRC = "data/bdsl_si"


def valid_mask(clip):  # clip: (C,T,V,M) -> (T,) nonzero frames (as make_cue_bundle.py)
    return np.abs(clip).sum(axis=(0, 2, 3)) > 0


def speed_unmasked(clip, idx):
    d = np.diff(clip[:2, idx], axis=1)
    return np.linalg.norm(d, axis=0).mean()


def speed_masked(clip, idx, return_share=False):
    xy = clip[:2, idx]                                   # (2, L, V, M)
    det = ~((xy[0] == 0) & (xy[1] == 0))                 # (L, V, M)
    both = det[1:] & det[:-1]
    sp = np.linalg.norm(np.diff(xy, axis=1), axis=0)     # (L-1, V, M)
    m = sp[both].mean() if both.any() else 0.0
    if return_share:   # share of the unmasked speed sum contributed by 0 <-> coordinate jumps
        jump = det[1:] ^ det[:-1]
        return m, (sp[jump].sum() / sp.sum() if sp.sum() > 0 else 0.0)
    return m


def transform(clip, cue):
    c = clip.copy()
    idx = np.where(valid_mask(c))[0]
    if len(idx) == 0:
        return c, None
    if cue == "speednorm_masked":
        sp, share = speed_masked(clip, idx, return_share=True)
        sp += 1e-6
        c[:2, idx] = clip[:2, idx] / sp
        return c, ((speed_unmasked(clip, idx) + 1e-6) / sp, share)
    if cue == "rootcenter_masked":
        miss = (clip[0] == 0) & (clip[1] == 0)           # (T, V, M)
        root = clip[:2, :, 0:1, :]
        cen = clip[:2] - root
        cen[:, miss] = 0.0
        c[:2] = cen
        return c, None
    raise ValueError(cue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue", required=True, choices=["speednorm_masked", "rootcenter_masked"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.abspath(a.out) == os.path.abspath(SRC):
        raise SystemExit("refusing to write into the source bundle")
    os.makedirs(a.out, exist_ok=True)
    for sp in ("train", "val", "test"):
        X = np.load(f"{SRC}/{sp}_data.npy", mmap_mode="r")
        nm, lb = pickle.load(open(f"{SRC}/{sp}_label.pkl", "rb"))
        out = np.lib.format.open_memmap(f"{a.out}/{sp}_data.npy", mode="w+", dtype=np.float32, shape=X.shape)
        ratios, shares = [], []
        for i in range(X.shape[0]):
            out[i], info = transform(np.asarray(X[i]), a.cue)
            if info is not None:
                ratios.append(info[0]); shares.append(info[1])
        out.flush(); del out
        pickle.dump((nm, lb), open(f"{a.out}/{sp}_label.pkl", "wb"))
        extra = ""
        if ratios:
            r = np.array(ratios)
            extra = (f"  unmasked/masked speed ratio median {np.median(r):.2f} [IQR {np.percentile(r, 25):.2f}, "
                     f"{np.percentile(r, 75):.2f}]; median share of the old speed sum from 0<->coord jumps "
                     f"{np.median(shares):.0%}")
        print(f"  {a.cue} {sp}: {X.shape[0]} clips{extra}", flush=True)
    print(f"DONE cue={a.cue} -> {a.out}")


if __name__ == "__main__":
    main()
