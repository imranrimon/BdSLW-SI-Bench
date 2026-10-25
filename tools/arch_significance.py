"""Proper architecture-significance tests, replacing the degenerate 3-seed
paired bootstrap (which yields p=0.0000 mechanically whenever the 3 seed-diffs
share a sign).

Two legitimate resampling units:
  (A) PER-INSTANCE clip-level paired bootstrap on the Protocol-A held-out TEST
      (n=1365 clips, 3-seed ensemble), for the arch-table pairwise claims.
      CAVEAT: clips fall within only 2 TEST signers, so this is a per-instance
      (not signer-generalization) test.
  (B) SIGNER-CLUSTERED LOSO bootstrap over the 11 unseen-signer folds
      (resample signers), for the 4 LOSO architectures -- the honest
      signer-generalization significance.
Holm-Bonferroni correction is applied WITHIN each family and reported.

  python tools/arch_significance.py [--lr0p1] [--two-sided] [--bestck]

--two-sided appends, to every pair line, the two-sided p (equal-tailed percentile-bootstrap
inversion: p2 = min(1, 2 * one-sided p), computed from the SAME bootstrap draws, so the default
output and the RNG stream are unchanged) and its Holm adjustment within the same family.
--bestck uses the best-on-val Protocol-A TEST dumps bdsl_<arch>_si_bestck_seed<S>_test for
SLGTFormer, GNN+LSTM, Pose-LSTM and ST-GCN (docs/p2_rerun_fixes_2026-10/).
"""
import os, sys, csv, pickle, collections
import numpy as np


def _alias_numpy_core():
    """Score pkls pickled under numpy>=2 reference 'numpy._core'; alias to
    'numpy.core' so pickle.load works in this numpy<2 env (bdsl_graph)."""
    if hasattr(np, "_core"):
        return
    import numpy.core as _core
    sys.modules.setdefault("numpy._core", _core)
    for sub in ("multiarray", "numeric", "umath", "_multiarray_umath",
                "_exceptions", "numerictypes", "_dtype", "_methods"):
        m = sys.modules.get(f"numpy.core.{sub}")
        if m is not None:
            sys.modules.setdefault(f"numpy._core.{sub}", m)


_alias_numpy_core()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
NBOOT = 10000
RNG = np.random.default_rng(20260827)

ARCHES = ["adaptive_gnn", "block_gcn", "st_gcn_vanilla", "slgtformer",
          "gnn", "gnn_lstm", "gnn_transformer", "pose_lstm"]
PRETTY = {"adaptive_gnn": "Adaptive GNN", "block_gcn": "BlockGCN",
          "st_gcn_vanilla": "ST-GCN", "slgtformer": "SLGTFormer",
          "gnn": "Attention GNN", "gnn_lstm": "GNN+LSTM",
          "gnn_transformer": "GNN+Transformer", "pose_lstm": "Pose LSTM"}
LOSO_ARCHES = ["adaptive_gnn", "block_gcn", "st_gcn_vanilla", "slgtformer",
               "gnn", "gnn_lstm", "pose_lstm", "gnn_transformer"]
# GNN+Transformer's reported LOSO is the grad-clip-5.0 re-run (two unclipped runs diverge).
LOSO_FILE = {"gnn_transformer": "gnn_transformer_clip5",
             # BlockGCN's reported recipe is base LR 0.01 (docs/blockgcn_lr01_switch/).
             "block_gcn": "block_gcn_lr01"}
# Protocol-A TEST score dumps of the reported runs: BlockGCN at lr 0.01, GNN+Transformer clipped at 5.0.
PKL_STEM = {"block_gcn": "block_gcn_si_lr01", "gnn_transformer": "gnn_transformer_si_clip5"}


def test_labels():
    names, labs = pickle.load(open(os.path.join(ROOT, "data/bdsl_si/test_label.pkl"), "rb"))
    return dict(zip(names, labs))


def _pkl_for(arch, seed):
    # ONLY the honest --phase test eval (test signers U02/U13); NO fallback to
    # the training run's val-eval (which contains U15 and would corrupt the test).
    p = os.path.join(ROOT, f"work_dir/bdsl_{PKL_STEM.get(arch, arch + '_si')}_seed{seed}_test", "eval_results", "best_acc.pkl")
    return p if os.path.isfile(p) else None


def ensemble_correct(arch, y):
    """Ensemble (over available TEST-eval seeds) per-clip correctness vec, sorted-clip order."""
    acc = {}
    used = 0
    for s in SEEDS:
        p = _pkl_for(arch, s)
        if p is None:
            continue
        used += 1
        sc = pickle.load(open(p, "rb"))
        for c in sc:
            acc[c] = acc.get(c, 0.0) + np.asarray(sc[c])
    clips = sorted(acc)
    return np.array([int(np.argmax(acc[c]) == y[c]) for c in clips]), clips, used


TWO_SIDED = False


def holm(pairs, key="p"):
    """pairs: list of dicts with `key`; add `key`+'_holm' (Holm step-down, monotone).
    With --two-sided, also adds 'p2' = min(1, 2p) and 'p2_holm'."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][key])
    m = len(pairs)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pairs[i][key])
        adj = max(adj, prev)          # enforce monotonicity
        prev = adj
        pairs[i][key + "_holm"] = adj
    if key == "p" and TWO_SIDED:
        for pr in pairs:
            pr["p2"] = min(1.0, 2.0 * pr["p"])
        holm(pairs, "p2")
    return pairs


def _p2(pr):
    return f" | two-sided p2={pr['p2']:.4f} p2_holm={pr['p2_holm']:.4f}" if TWO_SIDED else ""


def boot_paired(a, b):
    """paired bootstrap of mean(a)-mean(b) over shared units; obs, ci, one-sided p."""
    n = len(a)
    idx = RNG.integers(0, n, size=(NBOOT, n))
    d = (a[idx].mean(1) - b[idx].mean(1)) * 100
    obs = (a.mean() - b.mean()) * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = (d <= 0).mean() if obs >= 0 else (d >= 0).mean()
    return obs, lo, hi, float(p)


# ---------- (A) per-instance clip-level ----------
def per_instance():
    y = test_labels()
    corr, accs, nseed = {}, {}, {}
    for a in ARCHES:
        v, _, used = ensemble_correct(a, y)
        corr[a] = v
        accs[a] = 100 * v.mean()
        nseed[a] = used
    rank = sorted(ARCHES, key=lambda a: -accs[a])
    print("## (A) PER-INSTANCE clip-level paired bootstrap (Protocol-A TEST, seed ensemble)")
    print(f"n = {len(next(iter(corr.values())))} clips, {NBOOT} resamples. "
          f"CAVEAT: clips within 2 TEST signers -> per-instance, not signer-generalization.\n")
    print("Ranking (ensemble Top-1):")
    for a in rank:
        print(f"  {PRETTY[a]:16s} {accs[a]:.2f}  ({nseed[a]} seeds)")
    pairs = []
    for i in range(len(rank) - 1):          # adjacent pairs
        a, b = rank[i], rank[i + 1]
        obs, lo, hi, p = boot_paired(corr[a], corr[b])
        pairs.append({"a": a, "b": b, "obs": obs, "lo": lo, "hi": hi, "p": p})
    holm(pairs)
    print("\nAdjacent-rank pairs (Holm-corrected across the {} adjacent tests):".format(len(pairs)))
    for pr in pairs:
        star = "***" if pr["p_holm"] < .001 else "**" if pr["p_holm"] < .01 else "*" if pr["p_holm"] < .05 else "n.s."
        print(f"  {PRETTY[pr['a']]:16s} > {PRETTY[pr['b']]:16s} "
              f"D=+{pr['obs']:.2f} CI[{pr['lo']:+.2f},{pr['hi']:+.2f}] "
              f"p={pr['p']:.4f} p_holm={pr['p_holm']:.4f} {star}{_p2(pr)}")
    # the specific claimed pair
    a, b = "adaptive_gnn", "st_gcn_vanilla"
    obs, lo, hi, p = boot_paired(corr[a], corr[b])
    print(f"\nClaimed pair Adaptive GNN > ST-GCN: D=+{obs:.2f} CI[{lo:+.2f},{hi:+.2f}] p={p:.4f}"
          + (f" | two-sided p2={min(1.0, 2 * p):.4f}" if TWO_SIDED else ""))
    return rank, accs


# ---------- (B) signer-clustered LOSO ----------
def loso_perfold():
    """{arch: {fold: mean-over-seeds held-out-signer Top-1}} from docs/loso_heldout/<arch>.csv.

    These files score each fold's best checkpoint on that fold's HELD-OUT test
    signer (scripts/hpc/slurm_loso_heldout_allarch.sbatch). Do NOT read the
    LOSO rows of results_final.csv: those runs evaluate on the fold's validation
    signer, so their Top-1 is validation-signer accuracy, not held-out.
    """
    out = {}
    for a in LOSO_ARCHES:
        per = collections.defaultdict(list)
        with open(os.path.join(ROOT, "docs/loso_heldout", f"{LOSO_FILE.get(a, a)}.csv")) as f:
            for r in csv.DictReader(f):
                if r["test_acc"]:
                    per[r["fold"]].append(float(r["test_acc"]) / 100.0)
        out[a] = {fold: float(np.mean(v)) for fold, v in per.items()}
    return out


def signer_clustered():
    pf = loso_perfold()
    folds = sorted(set.intersection(*[set(pf[a]) for a in LOSO_ARCHES]))
    print("\n\n## (B) SIGNER-CLUSTERED LOSO bootstrap (resample the {} unseen-signer folds)".format(len(folds)))
    means = {a: np.mean([pf[a][f] for f in folds]) for a in LOSO_ARCHES}
    # scale check: LOSO accs may be fractions or percents
    scale = 100.0 if means[LOSO_ARCHES[0]] < 1.5 else 1.0
    rank = sorted(LOSO_ARCHES, key=lambda a: -means[a])
    print(f"folds (signers) = {folds}\nLOSO mean Top-1:")
    for a in rank:
        print(f"  {PRETTY[a]:16s} {means[a]*scale:.2f}")
    M = {a: np.array([pf[a][f] for f in folds]) for a in LOSO_ARCHES}
    pairs = []
    for i in range(len(rank) - 1):
        a, b = rank[i], rank[i + 1]
        d = (M[a] - M[b]) * scale
        obs = d.mean()
        idx = RNG.integers(0, len(folds), size=(NBOOT, len(folds)))
        bs = d[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = (bs <= 0).mean() if obs >= 0 else (bs >= 0).mean()
        pairs.append({"a": a, "b": b, "obs": obs, "lo": lo, "hi": hi, "p": float(p)})
    holm(pairs)
    print(f"\nAdjacent-rank pairs, paired-by-signer (n={len(folds)} clusters, Holm-corrected):")
    for pr in pairs:
        star = "**" if pr["p_holm"] < .01 else "*" if pr["p_holm"] < .05 else "n.s."
        print(f"  {PRETTY[pr['a']]:16s} > {PRETTY[pr['b']]:16s} "
              f"D=+{pr['obs']:.2f} CI[{pr['lo']:+.2f},{pr['hi']:+.2f}] "
              f"p={pr['p']:.4f} p_holm={pr['p_holm']:.4f} {star}{_p2(pr)}")


def use_lr0p1():
    """Adaptive GNN + Attention GNN at the 3-seed val-selected base LR 0.1
    (docs/adaptive_attn_lr0p1_switch/): TEST pkls work_dir/bdsl_<arch>_si_lr0p1_seed<S>_test,
    LOSO docs/loso_heldout/<arch>_lr0p1.csv."""
    for a in ("adaptive_gnn", "gnn"):
        PKL_STEM[a] = f"{a}_si_lr0p1"
        LOSO_FILE[a] = f"{a}_lr0p1"


def use_bestck():
    for a in ("slgtformer", "gnn_lstm", "pose_lstm", "st_gcn_vanilla"):
        PKL_STEM[a] = f"{a}_si_bestck"


if __name__ == "__main__":
    TWO_SIDED = "--two-sided" in sys.argv[1:]
    if "--bestck" in sys.argv[1:]:
        use_bestck()
        print("[variant] best-on-val Protocol-A TEST dumps (bdsl_<arch>_si_bestck) for SLGTFormer, GNN+LSTM, "
              "Pose-LSTM, ST-GCN\n")
    if "--lr0p1" in sys.argv[1:]:
        use_lr0p1()
        print("[variant] Adaptive GNN + Attention GNN at base LR 0.1 (lr0p1 TEST pkls + LOSO csv)\n")
    per_instance()
    signer_clustered()
