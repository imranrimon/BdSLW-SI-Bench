#!/usr/bin/env python
"""Analysis 1 (stage 3): test-time mirroring intervention for left-handed productions.

Compares per-clip held-out predictions of the SAME checkpoints (Adaptive GNN, seed 0,
each LOSO fold's model-best.pt) on
  base     : the fold's held-out test bundle            (loso_dumps/<arch>_test<XX>_seed0.pkl)
  left     : _L-tagged clips mirrored (filename metadata) (loso_dumps_mirror/..._mirrorleft.pkl)
  all      : every clip mirrored (control)               (..._mirrorall.pkl)
  tracked  : metadata-free rule - mirror clips whose MediaPipe left-hand landmarks are
             present in more frames than the right-hand ones (..._mirrortracked.pkl)
Mirroring = feeders/feeder.py random_mirror (flip_index + x -> 1-x), see
tools/p2_analysis/make_mirrored_test.py.

Reports per fold: Top-1 base vs condition, split by the filename hand tag, and for
`tracked` (all 11 folds) the LOSO mean / SD across signers / worst signer before vs after,
with a signer-clustered bootstrap CI of the mean change and an exact sign test over folds.
Output: docs/p2_analysis_2026-10/handedness_mirror.json
"""
import glob, json, os, pickle, re

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(os.environ.get("P2_ANALYSIS_TMP", "work_dir/p2_analysis_tmp"), "loso_dumps")
MIR = os.path.join(os.environ.get("P2_ANALYSIS_TMP", "work_dir/p2_analysis_tmp"), "loso_dumps_mirror")
OUT = os.path.join(ROOT, "docs/p2_analysis_2026-10/handedness_mirror.json")
ARCH, SEED = "adaptive_gnn", 0


def labels(fold):
    d = glob.glob(os.path.join(ROOT, f"data/bdsl_si_loso/test_U{fold}_val_U*"))[0]
    n, l = pickle.load(open(os.path.join(d, "test_label.pkl"), "rb"))
    return dict(zip(n, [int(x) for x in l]))


def correct(path, lab):
    sc = pickle.load(open(path, "rb"))
    assert set(sc) == set(lab), path
    return {k: int(np.argmax(v) == lab[k]) for k, v in sc.items()}


def summarize(c, names):
    tag = {n: re.search(r"_([LR])\.mp4$", n).group(1) for n in names}
    out = dict(acc=100 * np.mean([c[n] for n in names]), n=len(names))
    for t in "LR":
        sub = [c[n] for n in names if tag[n] == t]
        out[f"acc_{t}"] = 100 * float(np.mean(sub)) if sub else None
        out[f"n_{t}"] = len(sub)
    return out


PA = {  # Protocol-A TEST per-clip dumps, mirror augmentation off vs on (paper tab:mirroraug)
    "BlockGCN (lr0.01)": ("work_dir/bdsl_block_gcn_si_lr01_seed{s}_test/eval_results/best_acc.pkl",
                          "work_dir/bdsl_block_gcn_si_mirror_lr01_test_seed{s}/eval_results/best_acc.pkl"),
    "SLGTFormer": ("work_dir/bdsl_slgtformer_si_seed{s}_test/eval_results/best_acc.pkl",
                   "work_dir/bdsl_slgtformer_si_mirror_test_seed{s}/eval_results/best_acc.pkl"),
}


def protocol_a_mirror():
    """Where does the train-time mirror-augmentation gain on Protocol-A TEST (U02+U13) come from?
    Accuracy split by filename hand tag, mirror off vs on, 3 seeds."""
    n, l = pickle.load(open(os.path.join(ROOT, "data/bdsl_si/test_label.pkl"), "rb"))
    lab = dict(zip(n, [int(x) for x in l]))
    names = sorted(lab)
    out = {}
    for arch, (off, on) in PA.items():
        rows = []
        for s in (0, 1, 2):
            r = {"seed": s}
            for tag, path in (("off", off), ("on", on)):
                pth = os.path.join(ROOT, path.format(s=s))
                c = correct(pth, lab)
                sm = summarize(c, names)
                r[tag] = dict(acc=sm["acc"], acc_L=sm["acc_L"], acc_R=sm["acc_R"], file=path.format(s=s))
                r["n_L"], r["n_R"] = sm["n_L"], sm["n_R"]
            rows.append(r)
        agg = {}
        for k in ("acc", "acc_L", "acc_R"):
            agg[k] = dict(off=float(np.mean([r["off"][k] for r in rows])), on=float(np.mean([r["on"][k] for r in rows])))
            agg[k]["delta"] = agg[k]["on"] - agg[k]["off"]
        nL, nR = rows[0]["n_L"], rows[0]["n_R"]
        # share of the overall gain contributed by L-tagged clips
        agg["share_of_gain_from_L"] = float(agg["acc_L"]["delta"] * nL / (agg["acc"]["delta"] * (nL + nR))) if agg["acc"]["delta"] else None
        out[arch] = dict(per_seed=rows, mean=agg, n_L=nL, n_R=nR)
    return out


def main():
    res = {"arch": ARCH, "seed": SEED, "folds": {}}
    tracked_rows = []
    for bp in sorted(glob.glob(os.path.join(BASE, f"{ARCH}_test*_seed{SEED}.pkl"))):
        fold = re.search(r"_test(\d+)_", bp).group(1)
        lab = labels(fold)
        names = sorted(lab)
        cb = correct(bp, lab)
        fr = {"base": summarize(cb, names), "base_file": bp}
        for cond in ("left", "all", "tracked"):
            mp = os.path.join(MIR, f"{ARCH}_test{fold}_seed{SEED}_mirror{cond}.pkl")
            if not os.path.exists(mp):
                continue
            cm = correct(mp, lab)
            s = summarize(cm, names)
            s["file"] = mp
            # paired clip-level change: flips
            s["fixed"] = int(sum(1 for n in names if cm[n] and not cb[n]))
            s["broken"] = int(sum(1 for n in names if cb[n] and not cm[n]))
            s["mcnemar_p"] = float(stats.binomtest(s["fixed"], s["fixed"] + s["broken"]).pvalue) if s["fixed"] + s["broken"] else 1.0
            fr[cond] = s
            if cond == "tracked":
                tracked_rows.append((f"U{fold}", fr["base"]["acc"], s["acc"]))
        res["folds"][f"U{fold}"] = fr
    if len(tracked_rows) == 11:
        sig, b, a = zip(*tracked_rows)
        b, a = np.array(b), np.array(a)
        d = a - b
        rng = np.random.default_rng(0)
        bs = np.array([d[rng.integers(0, 11, 11)].mean() for _ in range(10000)])
        npos = int((d > 0).sum()); nneg = int((d < 0).sum())
        res["tracked_summary"] = dict(
            signers=list(sig), base=b.round(2).tolist(), tracked=a.round(2).tolist(),
            mean_base=float(b.mean()), mean_tracked=float(a.mean()), mean_change=float(d.mean()),
            mean_change_ci95_signer_bootstrap=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            sd_across_signers_base=float(b.std(ddof=1)), sd_across_signers_tracked=float(a.std(ddof=1)),
            worst_base=[sig[int(b.argmin())], float(b.min())], worst_tracked=[sig[int(a.argmin())], float(a.min())],
            range_base=float(b.max() - b.min()), range_tracked=float(a.max() - a.min()),
            folds_improved=npos, folds_worse=nneg,
            sign_test_p=float(stats.binomtest(npos, npos + nneg).pvalue) if npos + nneg else 1.0)
    res["train_time_mirror_protocolA"] = protocol_a_mirror()
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    for f, fr in res["folds"].items():
        line = f"{f} base {fr['base']['acc']:.2f} (L {fr['base']['acc_L']} n={fr['base']['n_L']}, R {fr['base']['acc_R']:.2f})"
        for cond in ("left", "all", "tracked"):
            if cond in fr:
                s = fr[cond]
                line += f" | {cond} {s['acc']:.2f} (L {s['acc_L'] if s['acc_L'] is None else round(s['acc_L'],2)}, R {s['acc_R']:.2f}; +{s['fixed']}/-{s['broken']})"
        print(line)
    print(json.dumps({a: v["mean"] for a, v in res["train_time_mirror_protocolA"].items()}, indent=1))
    if "tracked_summary" in res:
        print(json.dumps(res["tracked_summary"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
