"""E5 -- signer-diversity SMG(k) driver (mixed-accuracy, multi-subset draws).

For a diversity cell (k signers, draw r) this:
  1. picks a RANDOM size-k subset of the 8 canonical train signers, deterministic
     in (k, r), and ensures its bundle exists (make_diversity_draw_bundle.py),
  2. TRAINS one BlockGCN under the shared recipe, selecting on the canonical
     unseen val signer U15 (data/bdsl_si/val),
  3. TESTS that ONE checkpoint twice:
       - UNSEEN: canonical held-out TEST {2,13}  -> <stem>_divk<k>_d<r>_seed<s>_unseentest
       - SEEN  : the subset's seen-signer holdout -> <stem>_divk<k>_d<r>_seed<s>_seentest
     so SMG(k) = acc(seen) - acc(unseen) for the identical model.

Averaging over draws r gives A_unseen(k), A_seen(k), SMG(k) with which-signers
error bars -- the controlled within-BdSLW60 companion to the ASL-Citizen boundary
observation (more training signers -> smaller mixing gap).

Usage:
    python tools/run_diversity_grid.py --k 4 --draw 0 --seed 0 --skip-existing
"""
from __future__ import annotations

import argparse, glob, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIV_ROOT = ROOT / "data" / "bdsl_si_div"
CANON = ROOT / "data" / "bdsl_si"
POOL = [1, 4, 5, 6, 8, 9, 11, 12]        # canonical BdSLW60 SI train signers
SEED_BASE = 20260824                      # fixed -> draws reproducible everywhere
REGEX = None                              # signer-id regex; None -> bundler default


def _stem(cfg):
    return os.path.splitext(os.path.basename(cfg))[0]


def subset_for(k, draw):
    """Deterministic random size-k subset of POOL for (k, draw), pure in (k, draw).
    Distinct across draws for a fixed k (rejection sampling) so no two cells train
    the identical subset. k>=|POOL| -> full pool."""
    if k >= len(POOL):
        return sorted(POOL)
    rng = np.random.default_rng(SEED_BASE + 1000 * k)   # one stream per k
    seen, draws = set(), []
    while len(draws) <= draw:                            # generate 0..draw distinctly
        s = tuple(sorted(int(x) for x in rng.choice(POOL, size=k, replace=False)))
        if s not in seen:
            seen.add(s); draws.append(list(s))
    return draws[draw]


def _ensure_bundle(k, draw, holdout=0.2):
    out = DIV_ROOT / f"k{k}_d{draw}"
    if (out / "seentest_data.npy").exists() and (out / "train_data.npy").exists():
        return out
    signers = subset_for(k, draw)
    print(f"\n[bundle] k={k} draw={draw} signers={signers} ...")
    cmd = [
        sys.executable, "-u",
        str(ROOT / "preprocessing" / "make_diversity_draw_bundle.py"),
        "--signers", ",".join(str(s) for s in signers),
        "--out", str(out), "--holdout", str(holdout),
        "--seed", str(SEED_BASE + 1000 * k + draw),
        "--src", str(CANON),
    ]
    if REGEX:
        cmd += ["--regex", REGEX]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"bundle build failed for k{k}_d{draw}")
    return out


def _override_cfg(base_cfg, train_dir, test_data, test_label):
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("train_feeder_args", {})
    cfg["train_feeder_args"]["data_path"] = str(train_dir / "train_data.npy")
    cfg["train_feeder_args"]["label_path"] = str(train_dir / "train_label.pkl")
    cfg.setdefault("test_feeder_args", {})
    cfg["test_feeder_args"]["data_path"] = str(test_data)
    cfg["test_feeder_args"]["label_path"] = str(test_label)
    os.makedirs(str(ROOT / ".tmp_loso"), exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="divgrid_", suffix=".yaml", dir=str(ROOT / ".tmp_loso"))
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


def _test_phase(cfg, exp, ckpt, train_dir, test_data, test_label, tag):
    # train_feeder points at the real bundle train (unused in test phase, but keeps
    # main.py's loader construction from failing on a missing path).
    test_cfg = _override_cfg(cfg, train_dir, test_data, test_label)
    return subprocess.call([sys.executable, "-u", "main.py", "--config", test_cfg,
                            "--phase", "test", "--weights", ckpt,
                            "-Experiment_name", f"{exp}_{tag}test"], cwd=str(ROOT))


def run(cfg, k, draw, seed, skip_existing):
    stem = _stem(cfg)
    exp = f"{stem}_divk{k}_d{draw}_seed{seed}"
    if skip_existing and _has_test_marker(f"{exp}_unseen") and _has_test_marker(f"{exp}_seen"):
        print(f"[skip] {exp}: both evals present")
        return 0
    bundle = _ensure_bundle(k, draw)

    # phase 1: train, select on canonical U15 val -- unless a checkpoint already
    # exists (reuse it, e.g. after a test-phase-only bugfix rerun).
    ckpt = _best_ckpt(exp)
    if ckpt:
        print(f"[reuse] existing checkpoint for {exp}: {os.path.basename(ckpt)}")
    else:
        train_cfg = _override_cfg(cfg, bundle, CANON / "val_data.npy", CANON / "val_label.pkl")
        rc = subprocess.call([sys.executable, "-u", "main.py", "--config", train_cfg,
                              "--seed", str(seed), "-Experiment_name", exp], cwd=str(ROOT))
        if rc != 0:
            return rc
        ckpt = _best_ckpt(exp)
        if not ckpt:
            print(f"[error] no checkpoint for {exp}")
            return 1
    # phase 2a: UNSEEN canonical TEST {2,13};  2b: SEEN-signer holdout.
    rc1 = _test_phase(cfg, exp, ckpt, bundle, CANON / "test_data.npy", CANON / "test_label.pkl", "unseen")
    rc2 = _test_phase(cfg, exp, ckpt, bundle, bundle / "seentest_data.npy", bundle / "seentest_label.pkl", "seen")
    return rc1 or rc2


def main():
    global DIV_ROOT, CANON, POOL, REGEX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/bdsl_block_gcn_si.yaml")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    # dataset generalization (defaults = BdSLW60; AUTSL passes all four)
    ap.add_argument("--canon", default=None, help="canonical SI bundle dir (val/test reused)")
    ap.add_argument("--div-root", default=None, help="where per-cell bundles are written")
    ap.add_argument("--pool", default=None,
                    help="comma-list of train-signer ids to subsample (default BdSLW60 8)")
    ap.add_argument("--regex", default=None,
                    help=r"signer-id regex (default BdSLW60 'U(\d+)W'; AUTSL 'signer(\d+)_')")
    a = ap.parse_args()
    if a.canon:
        CANON = Path(a.canon)
    if a.div_root:
        DIV_ROOT = Path(a.div_root)
    if a.pool:
        POOL = [int(s) for s in a.pool.split(",") if s.strip()]
    if a.regex:
        REGEX = a.regex
    rc = run(a.config, a.k, a.draw, a.seed, a.skip_existing)
    if rc != 0:
        sys.exit(rc)
    print(f"\ndiversity cell k={a.k} draw={a.draw} seed={a.seed} complete.")


if __name__ == "__main__":
    main()
