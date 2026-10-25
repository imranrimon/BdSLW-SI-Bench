#!/usr/bin/env python
"""Stage-1/2 tables for the signer-normalized pose recipe (docs/signernorm_2026-10/).

Per stem: TEST Top-1 / Macro-F1 per seed (mean ± population SD, as docs/blockgcn_lr01_switch),
seed-ensemble Top-1, val-best Top-1 (U15, from the run's best checkpoint), per-signer ensemble accuracy,
Δ vs baseline with (i) per-seed paired differences and (ii) the per-instance clip-level paired
bootstrap of ensemble correctness (tools/arch_significance.boot_paired, 10k resamples).

  python tools/signernorm_report.py --base bdsl_block_gcn_si_lr01 --base-train bdsl_block_gcn_si_lr01_scratch \
      --stems bdsl_block_gcn_si_lr01_sn_a ...
"""
import argparse, os, pickle, re, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from arch_significance import boot_paired  # noqa: E402  (also aliases numpy._core)
from p2_metric_suite import _metrics_one, to_prob, load_labels  # noqa: E402


def seed_pkls(stem, seeds):
    out = {}
    for s in seeds:
        p = os.path.join(ROOT, f"work_dir/{stem}_seed{s}_test/eval_results/best_acc.pkl")
        if os.path.isfile(p):
            out[s] = {k: to_prob(v) for k, v in pickle.load(open(p, "rb")).items()}
    return out


def val_best(train_stem, s):
    p = os.path.join(ROOT, f"work_dir/{train_stem}_seed{s}/{train_stem}_seed{s}_model-best.pt")
    try:
        ck = torch.load(p, map_location="cpu")
        return 100 * float(ck["acc"]), int(ck["epoch"])
    except FileNotFoundError:
        return float("nan"), -1


def summarize(stem, train_stem, lab, seeds):
    per = seed_pkls(stem, seeds)
    if not per:
        return None
    clips = sorted(set(lab) & set.intersection(*[set(d) for d in per.values()]))
    y = np.array([lab[c] for c in clips])
    m = {s: _metrics_one(np.stack([d[c] for c in clips]), y) for s, d in per.items()}
    Pe = np.mean([np.stack([d[c] for c in clips]) for d in per.values()], 0)
    ens = _metrics_one(Pe, y)
    sig = {}
    for i, c in enumerate(clips):
        sig.setdefault(re.match(r"U(\d+)", c).group(1), []).append(ens["correct"][i])
    return dict(seeds=sorted(per), clips=clips, top1={s: m[s]["top1"] for s in m},
                f1={s: m[s]["macrof1"] for s in m}, ens_top1=ens["top1"], ens_f1=ens["macrof1"],
                ens_correct=ens["correct"], C=np.stack([m[s]["correct"] for s in sorted(per)]), per_signer={k: 100 * np.mean(v) for k, v in sig.items()},
                val={s: val_best(train_stem, s) for s in per})


def hier_ci(CA, CB, nb=10000, seed=20261023):
    """Hierarchical bootstrap of Δ = mean_seeds(acc A) - mean_seeds(acc B), in pp.

    CA, CB: (n_seeds, n_clips) 0/1 correctness on the SAME clip order. Per draw, each arm's seeds are
    resampled with replacement independently, and the clips are resampled once, paired across arms.
    Returns (lo, hi, p_two_sided) with percentile limits."""
    rng = np.random.default_rng(seed)
    n = CA.shape[1]
    out = np.empty(nb)
    for b in range(nb):
        ii = rng.integers(0, n, n)
        sa = rng.integers(0, CA.shape[0], CA.shape[0]); sb = rng.integers(0, CB.shape[0], CB.shape[0])
        out[b] = (CA[sa][:, ii].mean() - CB[sb][:, ii].mean()) * 100
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))
    return float(lo), float(hi), float(p)


def fmt(d):
    v = np.array(list(d.values()))
    return f"{v.mean():.2f} ± {v.std():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bdsl_block_gcn_si_lr01")
    ap.add_argument("--base-train", default="bdsl_block_gcn_si_lr01_scratch")
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", help="pretty names for --stems")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    lab = load_labels()
    B = summarize(a.base, a.base_train, lab, a.seeds)
    rows = [("baseline", a.base, a.base_train, B)]
    for i, st in enumerate(a.stems):
        rows.append(((a.labels or a.stems)[i] if a.labels else st, st, st, summarize(st, st, lab, a.seeds)))
    print("| recipe | seeds | TEST Top-1 per seed | TEST Top-1 mean ± SD | Macro-F1 mean ± SD | ensemble Top-1 | "
          "U02 / U13 (ens) | val U15 best (mean) | Δ Top-1 vs base (per-seed paired) | Δ ens, clip bootstrap 95% CI, p |")
    print("|---|--:|---|--:|--:|--:|--:|--:|--:|--:|")
    for name, stem, tstem, R in rows:
        if R is None:
            print(f"| {name} | 0 | (pending) |||||||| "); continue
        ps = " / ".join(f"{R['top1'][s]:.2f}" for s in R["seeds"])
        vb = np.nanmean([R["val"][s][0] for s in R["seeds"]])
        sg = f"{R['per_signer'].get('02', R['per_signer'].get('2', float('nan'))):.1f} / {R['per_signer'].get('13', float('nan')):.1f}"
        if R is B:
            d = "—"; bt = "—"
        else:
            common = [s for s in R["seeds"] if s in B["seeds"]]
            dd = np.array([R["top1"][s] - B["top1"][s] for s in common])
            d = f"{dd.mean():+.2f} ({', '.join(f'{x:+.2f}' for x in dd)})"
            assert R["clips"] == B["clips"]
            obs, lo, hi, p = boot_paired(R["ens_correct"], B["ens_correct"])
            bt = f"{obs:+.2f} [{lo:+.2f}, {hi:+.2f}], p={p:.4f}"
        print(f"| {name} | {len(R['seeds'])} | {ps} | {fmt(R['top1'])} | {fmt(R['f1'])} | {R['ens_top1']:.2f} | {sg} | "
              f"{vb:.2f} | {d} | {bt} |")


if __name__ == "__main__":
    main()
