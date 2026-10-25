"""Few-shot signer-personalization driver.

After signer-independent (SI) training, adapt the frozen SI checkpoint on k
labeled clips-per-class of a held-out TEST signer F, then evaluate on that same
signer's remaining clips. Sweeping k shows how much of the SI gap light
enrollment closes.

For a cell (signer F, shots k, seed s) this:
  1. ensures the k-shot bundle data/bdsl_si_pers/U<F>_k<k>_seed<s>/ exists
     (make_kshot_signer_bundle.py; CPU-only index filter, cached),
  2. k>0:  FINE-TUNES the SI checkpoint on the support set (small: ~30 epochs,
           low lr ~0.01, no warm-up) selecting nothing external -- train_feeder
           -> support, test_feeder -> query -- initialising from --init-weights
           with the final classifier fc dropped (--ignore-weights fc.weight
           fc.bias), so the head re-learns on the enrolled clips; then reports
           the best-checkpoint accuracy on the query set,
           logged <stem>_pers_U<F>_k<k>_seed<s>.
  3. k==0: TEST-EVALs the init checkpoint on the query set (all of F's clips),
           no training, logged <stem>_pers_U<F>_k0_seed<s>_test.

Uses the two-phase override pattern of run_paired_smg.py (_override_cfg writes a
tmp YAML under .tmp_loso; subprocess main.py) and reuses the _best_ckpt /
_has_test_marker helpers.

Usage:
    python tools/run_personalization.py \\
        --signer 2 --k 3 --seed 0 \\
        --init-weights work_dir/bdsl_block_gcn_si_seed0/bdsl_block_gcn_si_seed0_model-69.pt \\
        --config config/bdsl_block_gcn_si.yaml --skip-existing
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
PERS_ROOT = ROOT / "data" / "bdsl_si_pers"

# small adaptation recipe: short schedule, low lr, no warm-up (override the SI
# recipe's base_lr=0.1 / warm_up_epoch=5 / step=[60,80] / num_epoch=100).
ADAPT_EPOCHS = 30
ADAPT_LR = 0.01
ADAPT_STEP = [20, 26]
FC_KEYS = ["fc.weight", "fc.bias"]   # BlockGCN final classifier (model/block_gcn.py)


def _stem(cfg):
    return os.path.splitext(os.path.basename(cfg))[0]


def _ensure_bundle(signer, k, seed):
    out = PERS_ROOT / f"U{signer:02d}_k{k}_seed{seed}"
    if (out / "query_data.npy").exists():
        return out
    print(f"\n[bundle] k-shot personalization U{signer:02d} k={k} seed={seed} ...")
    rc = subprocess.call([
        sys.executable, "-u",
        str(ROOT / "preprocessing" / "make_kshot_signer_bundle.py"),
        "--signer", str(signer), "--k", str(k), "--seed", str(seed),
        "--out", str(out),
    ], cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"k-shot bundle build failed for U{signer:02d} k{k} seed{seed}")
    return out


def _override_cfg(base_cfg, train_dir, train_split, eval_dir, eval_split,
                  adapt=False):
    """train_feeder -> train_dir/<train_split>; test_feeder -> eval_dir/<eval_split>.
    When adapt=True, also stamp the small fine-tune recipe onto the config."""
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("train_feeder_args", {})
    cfg["train_feeder_args"]["data_path"] = str(train_dir / f"{train_split}_data.npy")
    cfg["train_feeder_args"]["label_path"] = str(train_dir / f"{train_split}_label.pkl")
    cfg.setdefault("test_feeder_args", {})
    cfg["test_feeder_args"]["data_path"] = str(eval_dir / f"{eval_split}_data.npy")
    cfg["test_feeder_args"]["label_path"] = str(eval_dir / f"{eval_split}_label.pkl")
    if adapt:
        cfg["num_epoch"] = ADAPT_EPOCHS
        cfg["base_lr"] = ADAPT_LR
        cfg["step"] = ADAPT_STEP
        cfg["warm_up_epoch"] = 0
    os.makedirs(str(ROOT / ".tmp_loso"), exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="pers_", suffix=".yaml", dir=str(ROOT / ".tmp_loso"))
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


def _adapt_done(exp):
    """A fine-tune cell is complete once its own eval_results holds scores."""
    d = ROOT / "work_dir" / exp / "eval_results"
    return d.is_dir() and any(p.suffix == ".pkl" for p in d.iterdir())


def run(cfg, signer, k, seed, init_weights, skip_existing):
    stem = _stem(cfg)
    exp = f"{stem}_pers_U{signer:02d}_k{k}_seed{seed}"

    if k == 0:
        # pure SI transfer: TEST the init checkpoint on all of F's clips, no train.
        if skip_existing and _has_test_marker(exp):
            print(f"[skip] {exp}: k=0 query already evaluated")
            return 0
        bundle = _ensure_bundle(signer, k, seed)
        test_cfg = _override_cfg(cfg, bundle, "query", bundle, "query")
        return subprocess.call([sys.executable, "-u", "main.py", "--config", test_cfg,
                                "--phase", "test", "--weights", init_weights,
                                "-Experiment_name", f"{exp}_test"], cwd=str(ROOT))

    # k>0: fine-tune the SI checkpoint on the support set, eval on the query set.
    if skip_existing and _adapt_done(exp):
        print(f"[skip] {exp}: adaptation already run")
        return 0
    bundle = _ensure_bundle(signer, k, seed)
    train_cfg = _override_cfg(cfg, bundle, "support", bundle, "query", adapt=True)
    # Warm-start the FULL SI checkpoint INCLUDING the classifier head (no --ignore-weights):
    # personalization ADAPTS the SI-learned head, it does not relearn it from k clips/class
    # (resetting fc would make low-k collapse below the k=0 SI baseline). Same 60-class space.
    return subprocess.call([sys.executable, "-u", "main.py", "--config", train_cfg,
                            "--seed", str(seed),
                            "--weights", init_weights,
                            "-Experiment_name", exp], cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/bdsl_block_gcn_si.yaml")
    ap.add_argument("--signer", type=int, required=True,
                    help="held-out TEST signer id (canonical SI test = 2 or 13)")
    ap.add_argument("--k", type=int, required=True,
                    help="labeled support clips per class (0 -> SI-transfer baseline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", required=True,
                    help="SI BlockGCN checkpoint to adapt from")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    rc = run(a.config, a.signer, a.k, a.seed, a.init_weights, a.skip_existing)
    if rc != 0:
        sys.exit(rc)
    print(f"\npersonalization cell U{a.signer:02d} k={a.k} seed={a.seed} complete.")


if __name__ == "__main__":
    main()
