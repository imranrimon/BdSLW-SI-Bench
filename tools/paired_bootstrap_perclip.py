"""Per-clip paired bootstrap between two conditions, from saved best_acc.pkl score dicts.

For each condition we mean-ensemble the per-clip 60-dim scores over its seeds, argmax to a
prediction, and score correctness against the val labels. We then paired-bootstrap the
accuracy difference over the shared clips (n=655), which is far tighter than seed-level.

Usage: python tools/paired_bootstrap_perclip.py --a <baseA> --b <baseB> [--label data/bdsl_si/val_label.pkl]
"""
import argparse, glob, pickle
import numpy as np


def load_labels(path):
    names, labels = pickle.load(open(path, "rb"))
    return {n: int(l) for n, l in zip(names, labels)}


def ensemble_pred(base):
    """Mean-ensemble scores over all seeds of `base`; return {clip: pred_class}."""
    pkls = sorted(glob.glob(f"work_dir/{base}_seed*/eval_results/best_acc.pkl"))
    if not pkls:
        return None, 0
    acc = {}
    for p in pkls:
        d = pickle.load(open(p, "rb"))
        for k, v in d.items():
            acc.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
    return {k: int(np.mean(vs, axis=0).argmax()) for k, vs in acc.items()}, len(pkls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", default="data/bdsl_si/val_label.pkl")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    lab = load_labels(a.label)
    pa, na = ensemble_pred(a.a)
    pb, nb = ensemble_pred(a.b)
    if not pa or not pb:
        print(f"missing scores (a seeds={na}, b seeds={nb})"); return
    clips = sorted(set(pa) & set(pb) & set(lab))
    ca = np.array([pa[c] == lab[c] for c in clips], dtype=np.float64)
    cb = np.array([pb[c] == lab[c] for c in clips], dtype=np.float64)
    diff = ca - cb
    obs = diff.mean() * 100
    rng = np.random.default_rng(1234)
    boot = rng.choice(diff, size=(a.n, len(diff)), replace=True).mean(axis=1) * 100
    p = float((boot <= 0).mean()) if obs >= 0 else float((boot >= 0).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{a.a} ({na} seeds, acc {ca.mean()*100:.2f}) vs {a.b} ({nb} seeds, acc {cb.mean()*100:.2f})")
    print(f"  Δacc = {obs:+.2f} pp  |  95% CI [{lo:+.2f}, {hi:+.2f}]  |  one-sided p = {p:.4f}  |  n_clips = {len(clips)}")


if __name__ == "__main__":
    main()
