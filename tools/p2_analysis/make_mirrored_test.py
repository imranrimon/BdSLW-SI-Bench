#!/usr/bin/env python
"""Test-time mirroring intervention for analysis 1 (handedness).

For a LOSO fold, writes a copy of the held-out test bundle in which the clips whose
filename hand tag is `_L` are horizontally mirrored exactly as the training feeder's
`random_mirror` does (feeders/feeder.py: joint permutation `flip_index` + x -> 1 - x
for non-zero coordinates). `--mode all` mirrors every clip (control: mirroring
right-handed clips should HURT them). `--mode tracked` is the metadata-free rule:
mirror a clip when MediaPipe's left-hand landmarks (joints 17-26) are present in more
frames than the right-hand ones (joints 7-16). Labels/names are copied unchanged, so
tools/eval_ckpt_on_npy.py --dump gives per-clip scores directly comparable with the
un-mirrored dumps.

  python tools/p2_analysis/make_mirrored_test.py --fold 01 --mode left \
      --out work_dir/p2_analysis_tmp/mirror
"""
import argparse, glob, os, pickle, re, shutil, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from feeders.feeder import flip_index  # noqa: E402


def mirror(clip):
    """clip: (C, T, V, M) raw normalised coords. Same op as Feeder random_mirror."""
    out = clip[:, :, flip_index, :].copy()
    x = out[0]
    out[0] = np.where(x == 0, 0, 1.0 - x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--mode", choices=["left", "all", "tracked"], default="left",
                    help="left: mirror _L-tagged clips (filename metadata); all: mirror every clip; "
                         "tracked: metadata-free rule, mirror clips in which MediaPipe tracks the hand in "
                         "joints 17-26 (left_hand_landmarks) in more frames than joints 7-16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = glob.glob(os.path.join(ROOT, f"data/bdsl_si_loso/test_U{a.fold}_val_U*"))[0]
    names, labels = pickle.load(open(os.path.join(src, "test_label.pkl"), "rb"))
    X = np.load(os.path.join(src, "test_data.npy"))
    assert X[..., 0].max() <= 2.0, "expected normalised coordinates"
    n_m = 0
    for i, n in enumerate(names):
        tag = re.search(r"_([LR])\.mp4$", n, re.I).group(1).upper()
        if a.mode == "tracked":
            pres = np.abs(X[i, :2]).sum(0)[..., 0] > 0          # (T, V)
            do = pres[:, 17:27].any(1).sum() > pres[:, 7:17].any(1).sum()
        else:
            do = a.mode == "all" or tag == "L"
        if do:
            X[i] = mirror(X[i]); n_m += 1
    od = os.path.join(a.out, f"test_U{a.fold}_{a.mode}")
    os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "test_data.npy"), X)
    shutil.copy(os.path.join(src, "test_label.pkl"), os.path.join(od, "test_label.pkl"))
    print(f"fold U{a.fold} mode={a.mode}: mirrored {n_m}/{len(names)} clips -> {od}")


if __name__ == "__main__":
    main()
