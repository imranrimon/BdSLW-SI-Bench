"""Hierarchical (signer-cluster) bootstrap — reviewer point #13/#16.

The clip-wise bootstrap in `paired_bootstrap_perclip.py` treats every clip as an
independent unit. Clips from the same signer are correlated, so that underestimates
uncertainty: the real deployment unit is a *signer*, not a clip. This tool instead
does a two-level (hierarchical) resample:

    1. resample SIGNERS (clusters) with replacement;
    2. within each resampled signer, resample its CLIPS with replacement.

and reports the accuracy (or paired Δ-accuracy) with a signer-aware 95% CI.

Signer coverage matters. The canonical SI test split has only 2 signers (2/13), so
`--source test` gives a degenerate 2-cluster CI (reported, but wide/unstable). The
meaningful estimate pools the **LOSO folds** (`--source loso`): every one of the 11
full signers is held out once, so the resample runs over all 11 signer clusters —
exactly the reviewer's "11 signer folds are more informative than one 2-signer split."

Usage:
  # single run, signer-cluster CI over the LOSO folds
  python tools/hierarchical_bootstrap.py --base bdsl_block_gcn_si --source loso
  # canonical test split (2-signer, degenerate CI — for completeness)
  python tools/hierarchical_bootstrap.py --base bdsl_block_gcn_si --source test
  # paired Δ between two conditions (same source), signer-cluster CI
  python tools/hierarchical_bootstrap.py --paired-a ft_dgxl_p10 --paired-b ft_pool6_p10 --source test
"""
import argparse, collections, csv, glob, pickle, re, sys

# old score/label pkls were written under numpy 2.x; make them loadable under 1.x
try:
    import numpy.core as _npc
    sys.modules.setdefault("numpy._core", _npc)
    for _s in ("multiarray", "numeric", "umath", "_multiarray_umath", "numerictypes"):
        try:
            sys.modules.setdefault("numpy._core." + _s, __import__("numpy.core." + _s, fromlist=["x"]))
        except Exception:
            pass
except Exception:
    pass
import numpy as np

SIG_RE = re.compile(r"U(\d+)W")


def _signer(clip):
    m = SIG_RE.search(clip)
    return int(m.group(1)) if m else -1


def _glob_dirs(base, source):
    """Return the eval_results/best_acc.pkl paths for a run `base` under a source."""
    if source == "test":
        pats = [f"work_dir/{base}_seed*_test/eval_results/best_acc.pkl"]
    elif source == "loso":
        pats = [f"work_dir/{base}_loso_test*_seed*/eval_results/best_acc.pkl"]
    else:  # val / selection
        pats = [f"work_dir/{base}_seed*/eval_results/best_acc.pkl"]
    out = []
    for p in pats:
        out += sorted(glob.glob(p))
    # val glob also matches _test dirs; drop them for source=val
    if source == "val":
        out = [p for p in out if "_test/" not in p]
    return out


def ensemble_pred(base, source):
    """Mean-ensemble per-clip scores over seeds (and LOSO folds); -> {clip: pred_class}."""
    pkls = _glob_dirs(base, source)
    if not pkls:
        return None, 0
    acc = {}
    for p in pkls:
        d = pickle.load(open(p, "rb"))
        for k, v in d.items():
            acc.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
    return {k: int(np.mean(vs, axis=0).argmax()) for k, vs in acc.items()}, len(pkls)


def load_labels(path):
    names, labels = pickle.load(open(path, "rb"))
    return {n: int(l) for n, l in zip(names, labels)}


def load_union_labels(glob_pat):
    """Union the per-fold held-out labels for a LOSO run into one {clip: label} map.

    Each LOSO fold's eval set is a DIFFERENT held-out signer, so a single label
    file (e.g. the 2-signer SI test split) covers only ~2 of the 11 signers and the
    paired/base intersection collapses to those. Reading every fold's held-out
    `test_label.pkl` recovers the full 11-signer clip->label map.
    """
    lab = {}
    files = sorted(glob.glob(glob_pat))
    for f in files:
        for n, l in zip(*pickle.load(open(f, "rb"))):
            lab[n] = int(l)
    return lab, len(files)


def loso_fold_accs(stem, csv_path="results_final.csv"):
    """Per-signer held-out test accuracy from the LOSO CSV rows.

    Rows are named `<stem>_loso_test<XX>_seed<N>` and record the held-out
    test-SIGNER Top-1 (per the run_loso policy). We take best-Top1 per (fold,seed)
    then average seeds -> one accuracy per signer fold. Returns {signer:int -> acc%}.
    No per-fold test-score pkls were saved, so this is the signer-level (one-level)
    cluster estimate; the within-signer clip level is omitted (dominant variance is
    across signers). Matches the summarize_seeds best-per-seed policy.
    """
    rx = re.compile(rf"^{re.escape(stem)}_loso_test(\d+)_seed(\d+)$")
    best = collections.defaultdict(dict)  # fold -> {seed: bestTop1}
    for r in csv.reader(open(csv_path)):
        if len(r) < 4 or r[3] in ("", "Top1_Acc"):
            continue
        m = rx.match(r[1])
        if not m:
            continue
        fold, seed = int(m.group(1)), int(m.group(2))
        try:
            v = float(r[3])
        except ValueError:
            continue
        best[fold][seed] = max(best[fold].get(seed, 0.0), v)
    return {fold: float(np.mean(list(sd.values())) * 100) for fold, sd in best.items()}


def cluster_bootstrap_1level(accs, n_boot, seed=1234):
    """Bootstrap the mean of per-signer accuracies by resampling signers w/ replacement."""
    a = np.asarray(accs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a.mean(), lo, hi, boots


def hbootstrap(per_clip_value, signers, n_boot, seed=1234):
    """Two-level bootstrap of the mean of `per_clip_value`, clustered by `signers`.

    per_clip_value : (C,) array (0/1 correctness, or a paired difference).
    signers        : (C,) array of signer ids aligned with per_clip_value.
    Returns (observed_mean, lo, hi, n_signers).
    """
    per_clip_value = np.asarray(per_clip_value, dtype=np.float64)
    signers = np.asarray(signers)
    uniq = np.unique(signers)
    by_sig = {s: np.where(signers == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)  # resample signer clusters
        vals = []
        for s in chosen:
            idx = by_sig[s]
            ridx = rng.choice(idx, size=len(idx), replace=True)  # resample clips within signer
            vals.append(per_clip_value[ridx])
        boots[b] = np.concatenate(vals).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return per_clip_value.mean(), lo, hi, len(uniq), boots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="single run base (globs its seeds/folds)")
    ap.add_argument("--paired-a", help="condition A base (paired mode)")
    ap.add_argument("--paired-b", help="condition B base (paired mode)")
    ap.add_argument("--source", choices=["test", "loso", "val", "loso-csv"], default="loso-csv")
    ap.add_argument("--label", default="data/bdsl_si/test_label.pkl")
    ap.add_argument("--loso-label-glob", default="data/bdsl_si_loso/test_U*_val_U*/test_label.pkl",
                    help="for --source loso: union these per-fold held-out label files "
                         "(each fold evals a distinct held-out signer) instead of --label")
    ap.add_argument("--csv", default="results_final.csv", help="for --source loso-csv")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()

    if a.source == "loso-csv":
        if not a.base:
            ap.error("--source loso-csv needs --base (the config stem)")
        accs = loso_fold_accs(a.base, a.csv)
        if not accs:
            print(f"no LOSO rows for {a.base} in {a.csv}"); return
        arr = np.array([accs[k] for k in sorted(accs)])
        mean, lo, hi, _ = cluster_bootstrap_1level(arr, a.n)
        print(f"[signer-level cluster bootstrap | source=loso-csv | {len(arr)} signer folds]")
        print(f"  mean acc = {mean:.2f}  |  95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  distribution: SD={arr.std():.2f}  worst={arr.min():.2f}  best={arr.max():.2f}  "
              f"p10={np.percentile(arr,10):.2f}")
        per = "  ".join(f"U{k:02d}={accs[k]:.1f}" for k in sorted(accs))
        print(f"  per-signer: {per}")
        return

    if a.source == "loso":
        lab, nlab = load_union_labels(a.loso_label_glob)
        print(f"[loso] union of {nlab} per-fold held-out label files -> {len(lab)} clips, "
              f"{len(set(_signer(c) for c in lab))} distinct signers")
    else:
        lab = load_labels(a.label)

    if a.paired_a and a.paired_b:
        pa, na = ensemble_pred(a.paired_a, a.source)
        pb, nb = ensemble_pred(a.paired_b, a.source)
        if not pa or not pb:
            print(f"missing scores (a files={na}, b files={nb})"); return
        clips = sorted(set(pa) & set(pb) & set(lab))
        sig = np.array([_signer(c) for c in clips])
        ca = np.array([pa[c] == lab[c] for c in clips], dtype=np.float64)
        cb = np.array([pb[c] == lab[c] for c in clips], dtype=np.float64)
        obs, lo, hi, ns, boots = hbootstrap(ca - cb, sig, a.n)
        p = float((boots <= 0).mean()) if obs >= 0 else float((boots >= 0).mean())
        print(f"[signer-cluster bootstrap | source={a.source} | {ns} signers | {len(clips)} clips]")
        print(f"  {a.paired_a} acc={ca.mean()*100:.2f}  vs  {a.paired_b} acc={cb.mean()*100:.2f}")
        print(f"  Δacc = {obs*100:+.2f} pp  |  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]  |  one-sided p = {p:.4f}")
    elif a.base:
        pa, na = ensemble_pred(a.base, a.source)
        if not pa:
            print(f"missing scores for {a.base} (source={a.source})"); return
        clips = sorted(set(pa) & set(lab))
        sig = np.array([_signer(c) for c in clips])
        corr = np.array([pa[c] == lab[c] for c in clips], dtype=np.float64)
        obs, lo, hi, ns, _ = hbootstrap(corr, sig, a.n)
        # per-signer breakdown (reviewer Q2: mean / SD / min / 10th pct)
        per = np.array([corr[sig == s].mean() for s in np.unique(sig)]) * 100
        print(f"[signer-cluster bootstrap | source={a.source} | {ns} signers | {len(clips)} clips | {na} score files]")
        print(f"  acc = {obs*100:.2f}  |  95% CI [{lo*100:.2f}, {hi*100:.2f}]")
        print(f"  per-signer: mean={per.mean():.2f}  SD={per.std():.2f}  min={per.min():.2f}  "
              f"p10={np.percentile(per,10):.2f}  (n_signers={len(per)})")
    else:
        ap.error("give --base OR (--paired-a and --paired-b)")


if __name__ == "__main__":
    main()
