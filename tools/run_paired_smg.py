"""E1 -- Paired-test-instance Signer-Mixing-Gap driver.

For each focus signer F and condition c, this:
  1. ensures the paired bundle data/bdsl_si_pairsmg/U<F>_<c>/ exists
     (generate_paired_smg_bundle.py; CPU-only, cached),
  2. TRAINS one BlockGCN under the shared recipe, selecting on the canonical
     val signer U15 (train_feeder -> condition train; test_feeder -> U15 val),
  3. TESTS the best checkpoint on the FROZEN focus-signer clips
     (main.py --phase test, test_feeder -> condition test), logging one row
     named <stem>_pairsmg_U<F>_<c>_seed<N>_test.

Because every condition for a given F is evaluated on the IDENTICAL frozen test
instances, the accuracy difference isolates signer familiarity:
    paired SMG = acc(seen_matched) - acc(unseen).

The two-phase train->test split guarantees the reported number is the frozen
test accuracy (selection is on U15, never on the test clips), regardless of the
val/test feeder wiring.

Usage:
    python tools/run_paired_smg.py \\
        --config config/bdsl_block_gcn_si.yaml \\
        --focus-signers 2 13 \\
        --conditions unseen seen_matched seen_aug \\
        --seeds 0 1 2 --skip-existing
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAIR_ROOT = ROOT / "data" / "bdsl_si_pairsmg"


def _stem(cfg):
    return os.path.splitext(os.path.basename(cfg))[0]


def _ensure_bundle(cache_dir, focus, conditions):
    cond_dirs = {c: PAIR_ROOT / f"U{focus:02d}_{c}" for c in conditions}
    missing = [c for c, d in cond_dirs.items() if not (d / "test_data.npy").exists()]
    if missing:
        print(f"\n[bundle] generating focus U{focus:02d} ({missing})...")
        rc = subprocess.call([
            sys.executable, "-u",
            str(ROOT / "preprocessing" / "generate_paired_smg_bundle.py"),
            "--cache-dir", str(cache_dir),
            "--output-root", str(PAIR_ROOT),
            "--focus-signer", str(focus),
            "--conditions", *missing,
        ], cwd=str(ROOT))
        if rc != 0:
            raise RuntimeError(f"paired-SMG bundle failed for focus U{focus:02d}")
    return cond_dirs


def _override_cfg(base_cfg, train_dir, eval_dir, eval_split):
    """train_feeder -> train_dir/train; test_feeder -> eval_dir/<eval_split>."""
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("train_feeder_args", {})
    cfg["train_feeder_args"]["data_path"] = str(train_dir / "train_data.npy")
    cfg["train_feeder_args"]["label_path"] = str(train_dir / "train_label.pkl")
    cfg.setdefault("test_feeder_args", {})
    cfg["test_feeder_args"]["data_path"] = str(eval_dir / f"{eval_split}_data.npy")
    cfg["test_feeder_args"]["label_path"] = str(eval_dir / f"{eval_split}_label.pkl")
    os.makedirs(str(ROOT / ".tmp_loso"), exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="pairsmg_", suffix=".yaml", dir=str(ROOT / ".tmp_loso"))
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return path


def _best_ckpt(exp):
    cands = glob.glob(str(ROOT / "work_dir" / exp / "*-best.pt"))
    if not cands:
        cands = sorted(glob.glob(str(ROOT / "work_dir" / exp / "*.pt")))
    return cands[0] if cands else None


def _has_test_marker(exp):
    d = ROOT / "work_dir" / f"{exp}_test" / "eval_results"
    return d.is_dir() and any(p.suffix == ".pkl" for p in d.iterdir())


def _run(cfg, focus, cond, seed, cond_dirs, skip_existing):
    stem = _stem(cfg)
    exp = f"{stem}_pairsmg_U{focus:02d}_{cond}_seed{seed}"
    cond_dir = cond_dirs[cond]

    if skip_existing and _has_test_marker(exp):
        print(f"[skip] {exp}: frozen-test already evaluated")
        return 0

    # phase 1: train, select on U15 val (the condition's own val split == U15).
    train_cfg = _override_cfg(cfg, cond_dir, cond_dir, "val")
    rc = subprocess.call([sys.executable, "-u", "main.py", "--config", train_cfg,
                          "--seed", str(seed), "-Experiment_name", exp], cwd=str(ROOT))
    if rc != 0:
        return rc

    # phase 2: test the best checkpoint on the FROZEN focus-signer clips.
    ckpt = _best_ckpt(exp)
    if not ckpt:
        print(f"[error] no checkpoint for {exp}")
        return 1
    test_cfg = _override_cfg(cfg, cond_dir, cond_dir, "test")
    rc = subprocess.call([sys.executable, "-u", "main.py", "--config", test_cfg,
                          "--phase", "test", "--weights", ckpt,
                          "-Experiment_name", f"{exp}_test"], cwd=str(ROOT))
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/bdsl_block_gcn_si.yaml")
    ap.add_argument("--focus-signers", nargs="+", type=int, default=[2, 13])
    ap.add_argument("--conditions", nargs="+",
                    default=["unseen", "seen_matched", "seen_aug"],
                    choices=["unseen", "seen_matched", "seen_aug"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cache-dir", default="data/bdsl_cache")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    failures = []
    for focus in a.focus_signers:
        cond_dirs = _ensure_bundle(Path(a.cache_dir), focus, a.conditions)
        for cond in a.conditions:
            for seed in a.seeds:
                rc = _run(a.config, focus, cond, seed, cond_dirs, a.skip_existing)
                if rc != 0:
                    failures.append((focus, cond, seed, rc))

    if failures:
        print("\nFAILED:")
        for focus, cond, seed, rc in failures:
            print(f"  U{focus:02d} {cond} seed{seed} exit={rc}")
        sys.exit(1)
    n = len(a.focus_signers) * len(a.conditions) * len(a.seeds)
    print(f"\nAll {n} paired-SMG runs complete.")


if __name__ == "__main__":
    main()
