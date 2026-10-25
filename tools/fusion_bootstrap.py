"""E9/C8 fix: per-instance (clip-level) bootstrap of the two-stream fusion gain.

The seed-paired test (n=3) is weak. The scores are per-clip, so we bootstrap the
PAIRED per-clip accuracy difference over the 1365 held-out TEST clips:
    fused = joint_score + alpha * bone_score   (alpha=1.0, as in fuse_scores.py)
Reports fused-minus-joint and fused-minus-bone Delta-accuracy with a 95% CI and a
one-sided bootstrap p (fraction of resamples with Delta<=0), using the 3-seed
ENSEMBLE scores (the fused headline object) and, for transparency, per-seed.

Caveat (in the paper): clips within a signer are correlated and the canonical
TEST split has only 2 signers, so a signer-cluster CI is degenerate; this
clip-level CI is the practical per-instance estimate, reported as such.

  python tools/fusion_bootstrap.py
"""
import os, sys, pickle
import numpy as np

if not hasattr(np, "_core"):  # score pkls pickled under numpy>=2
    import numpy.core as _c
    sys.modules.setdefault("numpy._core", _c)
    for _s in ("multiarray", "numeric", "umath", "_multiarray_umath", "_exceptions", "numerictypes", "_dtype", "_methods"):
        _m = sys.modules.get("numpy.core." + _s)
        if _m is not None:
            sys.modules.setdefault("numpy._core." + _s, _m)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
ALPHA = 1.0
NBOOT = 10000
RNG = np.random.default_rng(20260826)


def load(seed):
    # RECIPE-MATCHED: the bone is the clip5 window recipe, so the joint MUST be the
    # clip5 joint (68.47 mean), NOT the non-clip5 arch-table joint (65.23) -- else
    # the fusion gain mixes recipes and contradicts the E9 per-seed table.
    j = pickle.load(open(os.path.join(ROOT, f"work_dir/bdsl_block_gcn_si_clip5_seed{seed}_test/eval_results/best_acc.pkl"), "rb"))
    b = pickle.load(open(os.path.join(ROOT, f"work_dir/bdsl_block_gcn_si_bone_clip5_seed{seed}_test/eval_results/best_acc.pkl"), "rb"))
    return j, b


def labels():
    names, labs = pickle.load(open(os.path.join(ROOT, "data/bdsl_si/test_label.pkl"), "rb"))
    return dict(zip(names, labs))


def correct_vec(scores, y):
    """1 if argmax==label else 0, aligned to sorted clip order."""
    clips = sorted(scores)
    return np.array([int(np.argmax(np.asarray(scores[c])) == y[c]) for c in clips]), clips


def ensemble(y):
    """Seed-averaged (ensemble) per-clip correctness for joint, bone, fused."""
    js, bs = {}, {}
    for s in SEEDS:
        j, b = load(s)
        for c in j:
            js[c] = js.get(c, 0.0) + np.asarray(j[c])
            bs[c] = bs.get(c, 0.0) + np.asarray(b[c])
    clips = sorted(js)
    jc = np.array([int(np.argmax(js[c]) == y[c]) for c in clips])
    bc = np.array([int(np.argmax(bs[c]) == y[c]) for c in clips])
    fc = np.array([int(np.argmax(js[c] + ALPHA * bs[c]) == y[c]) for c in clips])
    return jc, bc, fc


def boot_delta(a, b):
    """95% CI + one-sided p for mean(a)-mean(b), paired clip bootstrap."""
    n = len(a)
    idx = RNG.integers(0, n, size=(NBOOT, n))
    da = (a[idx].mean(1) - b[idx].mean(1)) * 100
    obs = (a.mean() - b.mean()) * 100
    lo, hi = np.percentile(da, [2.5, 97.5])
    p = (da <= 0).mean()
    return obs, lo, hi, p


def main():
    y = labels()
    jc, bc, fc = ensemble(y)
    print("## E9 two-stream fusion — per-instance (clip) bootstrap, 3-seed ensemble\n")
    print(f"n = {len(fc)} held-out TEST clips, {NBOOT} paired bootstrap resamples.\n")
    print(f"- joint  ensemble acc = {100*jc.mean():.2f}%")
    print(f"- bone   ensemble acc = {100*bc.mean():.2f}%")
    print(f"- FUSED  ensemble acc = {100*fc.mean():.2f}%\n")
    for name, other in [("joint", jc), ("bone", bc)]:
        obs, lo, hi, p = boot_delta(fc, other)
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"- fused - {name}: Delta = +{obs:.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"p(one-sided) = {p:.4f}  {star}")
    # per-seed transparency
    print("\nPer-seed fused acc (for reference):")
    for s in SEEDS:
        j, b = load(s)
        clips = sorted(j)
        fac = np.mean([int(np.argmax(np.asarray(j[c]) + ALPHA * np.asarray(b[c])) == y[c]) for c in clips])
        jac = np.mean([int(np.argmax(np.asarray(j[c])) == y[c]) for c in clips])
        print(f"  seed{s}: joint {100*jac:.2f}  fused {100*fac:.2f}  (+{100*(fac-jac):.2f})")


if __name__ == "__main__":
    main()
