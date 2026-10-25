"""Missing-joint-aware bone modality (P2 fix, 2026-10).

Identical to preprocessing/generate_bone_data.py (bone = child - parent along
graph.sign_27.Graph.inward, root bone = 0) EXCEPT that a bone is set to 0 (all
channels) whenever the child OR the parent joint is missing in that frame.

Why: MediaPipe stores an undetected joint as (x, y, conf) = (0, 0, 0). The
unmasked generator then emits bone = 0 - parent = -parent for a missing child
with a detected parent (the two hand-root bones, body-wrist -> hand-wrist,
whenever a hand is not detected). That vector is not a bone; it is minus the
body-wrist position (median |.| ~1.25 vs ~0.12 for real bones).

A joint is "missing" iff its x and y are both exactly 0 (the cache convention;
conf is also 0 there). Zero-padded frames are all-missing, so their bones stay 0
exactly as before.

Writes <out-dir>/<split>_data_bone.npy and copies <split>_label.pkl so the
directory is self-contained. Never touches the source directory.

  python preprocessing/generate_bone_data_masked.py \
      --data-dir data/bdsl_si --output-dir data/bdsl_si_bonemask
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph.sign_27 import Graph  # noqa: E402


def missing_mask(data):
    """(N, C, T, V, M) -> bool (N, T, V, M): joint undetected (x == y == 0)."""
    return (data[:, 0] == 0) & (data[:, 1] == 0)


def joint_to_bone_masked(data, inward_edges):
    """Masked counterpart of generate_bone_data._joint_to_bone."""
    bone = np.zeros_like(data)
    miss = missing_mask(data)
    for v1, v2 in inward_edges:          # v1 parent (closer to root), v2 child
        b = data[:, :, :, v2, :] - data[:, :, :, v1, :]
        bad = miss[:, :, v2, :] | miss[:, :, v1, :]          # (N, T, M)
        b[np.broadcast_to(bad[:, None], b.shape)] = 0.0
        bone[:, :, :, v2, :] = b
    return bone


def _stats(data, bone_old, bone_new):
    valid = np.abs(data).sum(axis=(1, 3, 4)) > 0              # (N, T)
    miss = missing_mask(data)
    diff = np.abs(bone_old - bone_new).sum(axis=1) > 0         # (N, T, V, M)
    n_changed = int(diff.sum())
    mag_changed = np.linalg.norm(bone_old[:, :2], axis=1)[diff]
    real = (~diff) & valid[:, :, None, None] & (~miss)
    mag_real = np.linalg.norm(bone_old[:, :2], axis=1)[real]
    mag_real = mag_real[mag_real > 0]
    n_missing_valid = int((miss & valid[:, :, None, None]).sum())
    return (f"changed entries {n_changed} ({n_changed / max(n_missing_valid, 1):.1%} of missing-joint "
            f"entries in valid frames); median |old bone| at changed {np.median(mag_changed) if n_changed else 0:.3f} "
            f"vs real bones {np.median(mag_real):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    src, dst = Path(a.data_dir), Path(a.output_dir)
    if src.resolve() == dst.resolve():
        raise SystemExit("--output-dir must differ from --data-dir (never overwrite in place)")
    dst.mkdir(parents=True, exist_ok=True)
    g = Graph(labeling_mode="spatial")
    sys.path.insert(0, str(_ROOT / "preprocessing"))
    from generate_bone_data import _joint_to_bone   # the original, for the change report only
    for sp in a.splits:
        data = np.load(src / f"{sp}_data.npy")
        assert data.ndim == 5 and data.shape[1] == 3, data.shape
        new = joint_to_bone_masked(data, g.inward)
        old = _joint_to_bone(data, g.inward)
        print(f"[{sp}] {data.shape}: {_stats(data, old, new)}", flush=True)
        np.save(dst / f"{sp}_data_bone.npy", new)
        shutil.copy2(src / f"{sp}_label.pkl", dst / f"{sp}_label.pkl")
        print(f"  wrote {dst / f'{sp}_data_bone.npy'}", flush=True)


if __name__ == "__main__":
    main()
