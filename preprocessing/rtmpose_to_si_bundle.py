"""Build signer-independent BdSLW60 NPY bundles from RTMPose-133 poses.

Pose-quality ablation: does a richer/better pose extractor (RTMPose
COCO-WholeBody-133) reduce the signer-mixing gap versus our MediaPipe
27-keypoint skeleton? The controlled comparison keeps *everything* fixed
(model, SI split, feeder, recipe) and changes only the pose source, so
this script maps RTMPose-133 onto our *identical* 27-keypoint topology
and emits bundles byte-compatible with `data/bdsl_si`.

Design (kept deliberately faithful to `preprocessing/preprocess_bdsl.py`):

* **Same 27-keypoint layout / ordering.** 7 body + 10 right-hand + 10
  left-hand, in the exact slot order the models expect (see
  `graph/sign_27.py` and `preprocess_bdsl.BODY_INDICES/HAND_INDICES`).
  Only the *source* of each point changes (RTMPose vs MediaPipe).
* **Same (N, 3, T, 27, 1) tensor, same channels** = [x, y, confidence].
  Coordinates are the RTMPose normalized image coords (as stored in the
  .pkl, matching MediaPipe's [0,1]-ish range); the confidence channel is
  the RTMPose per-keypoint score. The feeder's `normalization: True`
  centers x/y and leaves the confidence channel untouched, exactly as
  for the MediaPipe bundle.
* **Same N / ordering / labels per split.** We iterate the *existing*
  `data/bdsl_si/<split>_label.pkl` `(sample_names, labels)` and load the
  matching RTMPose pkl for each clip, so train/val/test N, per-clip
  ordering, and integer labels are identical to the MediaPipe bundle by
  construction. The canonical SI split therefore carries over unchanged
  (train U01,04,05,06,08,09,11,12 / val U15 / test U02,13).

RTMPose .pkl format (COCO-WholeBody-133):
    dict with
      "keypoints": list of length T, each a (1, 133, 2) float array
      "scores":    list of length T, each a (1, 133) float array
  133-point layout: body[0:17] (COCO-17), feet[17:23], face[23:91],
  left-hand[91:112], right-hand[112:133]. Hand blocks use the standard
  21-point MediaPipe topology (wrist=0, thumb 1-4, index 5-8, middle
  9-12, ring 13-16, pinky 17-20), so our MediaPipe HAND_INDICES apply
  unchanged.

Output (mirrors `generate_signer_split_npy.py`):
    <output-dir>/<split>_data.npy    (N, 3, max_frames, 27, 1) float32
    <output-dir>/<split>_label.pkl   tuple (sample_names, labels)  [copied verbatim]

Typical invocation (from repo root):

    python preprocessing/rtmpose_to_si_bundle.py \
        --pose-root external/Uni-Sign/dataset/BdSLW60/pose_format \
        --si-bundle-dir data/bdsl_si \
        --output-dir data/bdsl_si_rtm \
        --splits train val test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

# Make the repo root importable regardless of how the script is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# --- 27-keypoint layout (identical to preprocess_bdsl.py) -------------------
# Our skeleton: slots 0-6 body, 7-16 right hand, 17-26 left hand.

# Body slots 0-6, in MediaPipe semantics:
#   0 Nose, 1 R-Shoulder, 2 L-Shoulder, 3 R-Elbow, 4 L-Elbow, 5 R-Wrist, 6 L-Wrist
# COCO-WholeBody-133 body block is COCO-17:
#   0 nose, 5 L-sho, 6 R-sho, 7 L-elb, 8 R-elb, 9 L-wri, 10 R-wri
# Mapping our body slot -> COCO-133 index (same physical joints as MediaPipe):
RTM_BODY_INDICES = [0, 6, 5, 8, 7, 10, 9]  # nose, Rsho, Lsho, Relb, Lelb, Rwri, Lwri

# Hand slots use the MediaPipe 21-pt selection [0,4,5,8,9,12,13,16,17,20]
#   = wrist, thumbTIP, indexMCP, indexTIP, midMCP, midTIP, ringMCP, ringTIP, pinkyMCP, pinkyTIP
# COCO-WholeBody hand blocks share the same 21-pt topology, so the same
# offsets apply within each hand block.
HAND_SELECT = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
RTM_LEFT_HAND_BASE = 91    # left-hand block start in the 133 layout
RTM_RIGHT_HAND_BASE = 112  # right-hand block start in the 133 layout
RTM_RIGHT_HAND_INDICES = [RTM_RIGHT_HAND_BASE + j for j in HAND_SELECT]
RTM_LEFT_HAND_INDICES = [RTM_LEFT_HAND_BASE + j for j in HAND_SELECT]

# Full ordered list of the 27 source indices into the 133-point array,
# arranged as [body(7), right hand(10), left hand(10)] to match our slots.
RTM_27_INDICES = RTM_BODY_INDICES + RTM_RIGHT_HAND_INDICES + RTM_LEFT_HAND_INDICES
assert len(RTM_27_INDICES) == 27


def _load_pkl_to_ctv(pkl_path):
    """Return (C=3, T, V=27) float32 = [x, y, score], or None if empty.

    Missing / low-confidence points are left as-is (RTMPose fills every
    keypoint every frame), matching the "store raw, normalize in feeder"
    convention of the MediaPipe bundle.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    kps = obj["keypoints"]  # list of (1, 133, 2)
    scs = obj["scores"]     # list of (1, 133)
    T = len(kps)
    if T == 0:
        return None
    kp = np.stack([np.asarray(x)[0] for x in kps]).astype(np.float32)  # (T,133,2)
    sc = np.stack([np.asarray(x)[0] for x in scs]).astype(np.float32)  # (T,133)
    sel_xy = kp[:, RTM_27_INDICES, :]   # (T,27,2)
    sel_sc = sc[:, RTM_27_INDICES]      # (T,27)
    out = np.empty((3, T, 27), dtype=np.float32)
    out[0] = sel_xy[:, :, 0]            # x  (T,27)
    out[1] = sel_xy[:, :, 1]            # y  (T,27)
    out[2] = sel_sc                     # confidence/score (T,27)
    return out


def _resolve_pkl_path(pose_root, class_name, sample_name):
    """Map a bundle sample name (e.g. 'U11W37F_trial_0_R.mp4') to its pkl.

    The RTMPose pkl for a clip lives at
    <pose_root>/<class_name>/<stem>.pkl, where <stem> is the sample name
    with the .mp4 extension stripped (the _L/_R dominant-hand suffix is
    part of the stem and is preserved).
    """
    stem = sample_name
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    return os.path.join(pose_root, class_name, stem + ".pkl")


def _build_split(split, pose_root, si_bundle_dir, classes, output_dir, max_frames):
    label_path = os.path.join(si_bundle_dir, "{}_label.pkl".format(split))
    with open(label_path, "rb") as f:
        sample_names, labels = pickle.load(f)

    n = len(sample_names)
    large = np.zeros((n, 3, max_frames, 27, 1), dtype=np.float32)

    missing = []
    empty = []
    kept_names = []
    kept_labels = []
    write_i = 0
    for sample_name, label in tqdm(
        list(zip(sample_names, labels)), desc="[{}] map".format(split)
    ):
        class_name = classes[label]
        pkl_path = _resolve_pkl_path(pose_root, class_name, sample_name)
        if not os.path.exists(pkl_path):
            missing.append((class_name, sample_name))
            continue
        ctv = _load_pkl_to_ctv(pkl_path)  # (3, T, 27)
        if ctv is None:
            empty.append((class_name, sample_name))
            continue
        t = ctv.shape[1]
        t_use = min(t, max_frames)
        large[write_i, :, :t_use, :, 0] = ctv[:, :t_use, :]
        kept_names.append(sample_name)
        kept_labels.append(label)
        write_i += 1

    if missing or empty:
        print(
            "[{}] WARNING: {} missing pkl, {} empty; dropping them.".format(
                split, len(missing), len(empty)
            )
        )
        for tag, lst in (("missing", missing), ("empty", empty)):
            for c, s in lst[:5]:
                print("    {}: {}/{}".format(tag, c, s))

    if write_i != n:
        large = large[:write_i]

    data_out = os.path.join(output_dir, "{}_data.npy".format(split))
    label_out = os.path.join(output_dir, "{}_label.pkl".format(split))
    np.save(data_out, large)
    with open(label_out, "wb") as f:
        pickle.dump((kept_names, kept_labels), f)
    print(
        "[{}] wrote {} shape={} (N={}, dropped={})".format(
            split, data_out, large.shape, write_i, n - write_i
        )
    )
    return write_i, n - write_i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-root",
        default="external/Uni-Sign/dataset/BdSLW60/pose_format",
        help="root holding <class>/<stem>.pkl RTMPose-133 files",
    )
    parser.add_argument(
        "--si-bundle-dir",
        default="data/bdsl_si",
        help="existing MediaPipe SI bundle dir (source of split N/labels/classes)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/bdsl_si_rtm",
        help="where to write <split>_data.npy / <split>_label.pkl",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="which splits to emit (default: train val test)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="temporal cap; must match the MediaPipe bundle (default 300)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="debug: only process the first N clips per split (0 = all)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    classes_json = os.path.join(args.si_bundle_dir, "classes.json")
    with open(classes_json) as f:
        cj = json.load(f)
    classes = cj["classes"]  # label index -> class name

    # Copy classes.json through so downstream tooling matches the source bundle.
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump(cj, f, indent=2)

    print("pose-root:   {}".format(args.pose_root))
    print("si-bundle:   {}".format(args.si_bundle_dir))
    print("output-dir:  {}".format(args.output_dir))
    print("max-frames:  {}".format(args.max_frames))
    print("27-index map (into COCO-WholeBody-133): {}".format(RTM_27_INDICES))

    if args.limit:
        # Debug slice: temporarily truncate the label lists in-memory only.
        print("DEBUG: limiting to first {} clips per split".format(args.limit))

    for split in args.splits:
        if args.limit:
            _build_split_limited(
                split, args.pose_root, args.si_bundle_dir, classes,
                args.output_dir, args.max_frames, args.limit,
            )
        else:
            _build_split(
                split, args.pose_root, args.si_bundle_dir, classes,
                args.output_dir, args.max_frames,
            )


def _build_split_limited(
    split, pose_root, si_bundle_dir, classes, output_dir, max_frames, limit
):
    """Same as _build_split but only the first `limit` clips (validation)."""
    label_path = os.path.join(si_bundle_dir, "{}_label.pkl".format(split))
    with open(label_path, "rb") as f:
        sample_names, labels = pickle.load(f)
    sample_names = sample_names[:limit]
    labels = labels[:limit]

    n = len(sample_names)
    large = np.zeros((n, 3, max_frames, 27, 1), dtype=np.float32)
    kept_names, kept_labels = [], []
    write_i = 0
    for sample_name, label in tqdm(
        list(zip(sample_names, labels)), desc="[{}] map(limit)".format(split)
    ):
        class_name = classes[label]
        pkl_path = _resolve_pkl_path(pose_root, class_name, sample_name)
        if not os.path.exists(pkl_path):
            continue
        ctv = _load_pkl_to_ctv(pkl_path)
        if ctv is None:
            continue
        t_use = min(ctv.shape[1], max_frames)
        large[write_i, :, :t_use, :, 0] = ctv[:, :t_use, :]
        kept_names.append(sample_name)
        kept_labels.append(label)
        write_i += 1
    if write_i != n:
        large = large[:write_i]
    data_out = os.path.join(output_dir, "{}_data_SLICE.npy".format(split))
    label_out = os.path.join(output_dir, "{}_label_SLICE.pkl".format(split))
    np.save(data_out, large)
    with open(label_out, "wb") as f:
        pickle.dump((kept_names, kept_labels), f)
    print("[{}] SLICE wrote {} shape={}".format(split, data_out, large.shape))


if __name__ == "__main__":
    main()
