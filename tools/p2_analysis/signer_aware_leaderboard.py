#!/usr/bin/env python
"""P2 analysis 4: signer-aware leaderboard statistics on held-out LOSO.

Data: docs/loso_heldout/<file>.csv (held-out-signer Top-1, 11 signers x 3 seeds),
with the paper's file choice (tools/arch_significance.py LOSO_FILE):
GNN+Transformer = gnn_transformer_clip5 (grad-clip 5.0; the unclipped runs have
collapsed seeds), BlockGCN = block_gcn_lr01. The unclipped GNN+Transformer is
reported as a sensitivity row only.

Statistics per architecture (A = 11 x 3 matrix, signers x seeds, Top-1 %):
  mean, worst-signer (min over signers of the seed-mean), P10 over the 33 runs,
  P10 over the 11 signer-means (linear interpolation), IQM over the 33 runs
  (rliable: mean of the middle 50% of pooled run scores), and IQM over signer-means.
CIs (95% percentile, 10k resamples):
  (i)  rliable-style stratified bootstrap: signers (tasks) fixed, seeds resampled
       with replacement within each signer;
  (ii) signer-clustered bootstrap: resample the 11 signers with replacement, then
       seeds within each drawn signer (the generalisation-to-new-signers CI).
Probability of improvement P(X > Y) (rliable, Agarwal et al. 2021): for each
signer, the Mann-Whitney statistic over X's and Y's 3 seeds (ties = 0.5), averaged
over signers; CIs by both bootstraps above (same resample indices for X and Y).
Mixed models (statsmodels MixedLM, REML), on the 264 run-level scores:
  M1 (requested):  acc ~ C(arch) + (1|signer) + (1|seed)   [crossed, via vc_formula]
  M2 (structure-correct): acc ~ C(arch) + (1|signer) + (1|signer:arch)
     (seed index is NOT shared across architectures - seed 0 of arch A and of arch B
      are unrelated runs - so (1|seed) is not a meaningful crossed factor; the 3 seeds
      within an arch x signer cell are the replicate level.)
  Arch contrasts for the paper's adjacent-rank pairs, Wald z, Holm-corrected.
Output: docs/p2_analysis_2026-10/signer_aware_leaderboard.json
"""
import csv, itertools, json, os, warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs/p2_analysis_2026-10/signer_aware_leaderboard.json")
FILES = [
    ("Adaptive GNN", "adaptive_gnn"), ("ST-GCN", "st_gcn_vanilla"), ("SLGTFormer", "slgtformer"),
    ("GNN+Transformer (clip5)", "gnn_transformer_clip5"), ("Attention GNN", "gnn"),
    ("BlockGCN (lr0.01)", "block_gcn_lr01"), ("GNN+LSTM", "gnn_lstm"), ("Pose LSTM", "pose_lstm")]
SENS = [("GNN+Transformer (unclipped)", "gnn_transformer"), ("BlockGCN (lr0.1, old)", "block_gcn")]
NBOOT = 10000
RNG = np.random.default_rng(20261022)


def load(fname):
    per = {}
    with open(os.path.join(ROOT, "docs/loso_heldout", fname + ".csv")) as f:
        for r in csv.DictReader(f):
            if r["test_acc"]:
                per.setdefault(r["heldout_signer"], {})[int(r["seed"])] = float(r["test_acc"])
    signers = sorted(per)
    A = np.array([[per[s][k] for k in (0, 1, 2)] for s in signers])
    return signers, A


def iqm(x):
    x = np.sort(np.asarray(x).ravel())
    n = len(x)
    # rliable uses scipy.stats.trim_mean(x, 0.25)
    from scipy.stats import trim_mean
    return float(trim_mean(x, 0.25))


STATS = {
    "mean": lambda A: float(A.mean()),
    "median_signer": lambda A: float(np.median(A.mean(1))),
    "iqm_runs": lambda A: iqm(A),
    "iqm_signer_means": lambda A: iqm(A.mean(1)),
    "worst_signer": lambda A: float(A.mean(1).min()),
    "p10_runs": lambda A: float(np.percentile(A.ravel(), 10)),
    "p10_signer_means": lambda A: float(np.percentile(A.mean(1), 10)),
    "best_signer": lambda A: float(A.mean(1).max()),
}


def boot_indices(nsig, nseed, kind):
    if kind == "stratified":
        sig = np.tile(np.arange(nsig), (NBOOT, 1))
    else:
        sig = RNG.integers(0, nsig, (NBOOT, nsig))
    seed = RNG.integers(0, nseed, (NBOOT, nsig, nseed))
    return sig, seed


def apply(A, sig, seed, b):
    return A[sig[b]][np.arange(sig.shape[1])[:, None], seed[b]]


def poi(X, Y):
    """rliable probability of improvement: mean over tasks of P(x > y) + 0.5 P(x = y)."""
    vals = []
    for x, y in zip(X, Y):
        gt = (x[:, None] > y[None, :]).mean()
        eq = (x[:, None] == y[None, :]).mean()
        vals.append(gt + 0.5 * eq)
    return float(np.mean(vals))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def mixed_models(data):
    import statsmodels.formula.api as smf
    df = pd.DataFrame(data, columns=["arch", "signer", "seed", "acc"])
    ref = "Adaptive GNN"
    df["arch"] = pd.Categorical(df["arch"], categories=[n for n, _ in FILES])
    df["signer_arch"] = df["signer"] + "_" + df["arch"].astype(str)
    df["one"] = 1
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = smf.mixedlm("acc ~ C(arch)", df, groups="one",
                         vc_formula={"signer": "0 + C(signer)", "seed": "0 + C(seed)"}).fit(reml=True)
        m2 = smf.mixedlm("acc ~ C(arch)", df, groups="signer",
                         vc_formula={"signer_arch": "0 + C(signer_arch)"}, re_formula="1").fit(reml=True)
    names = [n for n, _ in FILES]
    for tag, m in (("M1_signer_seed", m1), ("M2_signer_signerXarch", m2)):
        fe = m.fe_params
        cov = m.cov_params().loc[fe.index, fe.index]
        # arch effect vector relative to intercept (= reference arch)
        def vec(a):
            v = pd.Series(0.0, index=fe.index)
            v["Intercept"] = 1.0
            if a != ref:
                v[f"C(arch)[T.{a}]"] = 1.0
            return v
        marg = {}
        for a in names:
            v = vec(a)
            est = float(v @ fe)
            se = float(np.sqrt(v @ cov @ v))
            marg[a] = dict(est=est, se=se, ci95=[est - 1.96 * se, est + 1.96 * se])
        rank = sorted(names, key=lambda a: -marg[a]["est"])
        pairs = []
        from scipy.stats import norm
        for a, b in zip(rank[:-1], rank[1:]):
            v = vec(a) - vec(b)
            d = float(v @ fe)
            se = float(np.sqrt(v @ cov @ v))
            p = float(1 - norm.cdf(d / se))  # one-sided, a > b
            pairs.append(dict(a=a, b=b, diff=d, se=se, ci95=[d - 1.96 * se, d + 1.96 * se], p_one_sided=p))
        adj = holm(np.array([p["p_one_sided"] for p in pairs]))
        for p, q in zip(pairs, adj):
            p["p_holm"] = float(q)
        vcs = {k: float(v) for k, v in zip(m.model.exog_vc.names, m.vcomp)}
        res[tag] = dict(formula=("acc ~ C(arch) + (1|signer) + (1|seed)" if tag.startswith("M1")
                                 else "acc ~ C(arch) + (1|signer) + (1|signer:arch)"),
                        converged=bool(m.converged), group_var=float(m.cov_re.values.ravel()[0]) if m.cov_re.size else None,
                        vc_var=vcs, resid_var=float(m.scale), arch_marginal=marg, ranking=rank, adjacent=pairs)
    return res


def main():
    mats = {}
    for name, f in FILES + SENS:
        signers, A = load(f)
        mats[name] = (signers, A)
    signers = mats[FILES[0][0]][0]
    for name, (s, _A) in mats.items():
        assert s == signers, (name, s)
    nsig, nseed = mats[FILES[0][0]][1].shape

    idx = {k: boot_indices(nsig, nseed, k) for k in ("stratified", "signer_clustered")}
    table = {}
    for name, _f in FILES + SENS:
        A = mats[name][1]
        row = {"per_signer_seedmean": dict(zip(signers, A.mean(1).round(4).tolist())),
               "worst_signer_id": signers[int(A.mean(1).argmin())],
               "collapsed_runs_lt10pct": int((A < 10).sum()),
               "source": f"docs/loso_heldout/{_f}.csv"}
        for st, fn in STATS.items():
            row[st] = fn(A)
            for kind, (sig, seed) in idx.items():
                bs = np.array([fn(apply(A, sig, seed, b)) for b in range(NBOOT)])
                row[f"{st}_ci_{kind}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        table[name] = row

    main_names = [n for n, _ in FILES]
    # probability of improvement: all ordered pairs among main archs (by mean)
    rank = sorted(main_names, key=lambda n: -table[n]["mean"])
    poi_res = {}
    for a, b in itertools.combinations(rank, 2):
        X, Y = mats[a][1], mats[b][1]
        r = dict(poi=poi(X, Y))
        for kind, (sig, seed) in idx.items():
            bs = np.array([poi(apply(X, sig, seed, k), apply(Y, sig, seed, k)) for k in range(2000)])
            r[f"ci_{kind}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        poi_res[f"{a} > {b}"] = r

    # rankings by each statistic
    rankings = {st: sorted(main_names, key=lambda n: -table[n][st]) for st in STATS}

    data = []
    for name in main_names:
        A = mats[name][1]
        for i, s in enumerate(signers):
            for k in range(nseed):
                data.append((name, s, f"seed{k}", A[i, k]))
    mm = mixed_models(data)

    out = dict(signers=signers, n_boot=NBOOT, files={n: f"docs/loso_heldout/{f}.csv" for n, f in FILES + SENS},
               table=table, rankings=rankings, prob_improvement=poi_res, mixed_models=mm,
               notes=["GNN+Transformer main row = grad-clip-5.0 LOSO re-run (paper convention, "
                      "tools/arch_significance.py LOSO_FILE); unclipped run is a sensitivity row.",
                      "PoI CIs use 2000 resamples (each PoI resample is 11 x 9 comparisons)."])
    json.dump(out, open(OUT, "w"), indent=1)

    # console
    print(f"{'arch':28s} {'mean':>6s} {'IQM':>18s} {'worst':>20s} {'P10runs':>18s} {'P10sig':>8s}")
    for n in sorted(table, key=lambda n: -table[n]["mean"]):
        t = table[n]
        print(f"{n:28s} {t['mean']:6.2f} {t['iqm_runs']:6.2f}[{t['iqm_runs_ci_signer_clustered'][0]:.1f},{t['iqm_runs_ci_signer_clustered'][1]:.1f}] "
              f"{t['worst_signer']:6.2f}({t['worst_signer_id']})[{t['worst_signer_ci_stratified'][0]:.1f},{t['worst_signer_ci_stratified'][1]:.1f}] "
              f"{t['p10_runs']:6.2f}[{t['p10_runs_ci_signer_clustered'][0]:.1f},{t['p10_runs_ci_signer_clustered'][1]:.1f}] {t['p10_signer_means']:6.2f} collapsed={t['collapsed_runs_lt10pct']}")
    for st, r in rankings.items():
        print(st, r)
    for a, b in zip(rank[:-1], rank[1:]):
        r = poi_res[f"{a} > {b}"]
        print(f"PoI {a} > {b}: {r['poi']:.3f} strat{r['ci_stratified']} clus{r['ci_signer_clustered']}")
    r = poi_res.get(f"{rank[0]} > {rank[-1]}")
    for tag, m in mm.items():
        print(tag, "vc", m["vc_var"], "group", m["group_var"], "resid", m["resid_var"])
        for p in m["adjacent"]:
            print(f"  {p['a']} > {p['b']}: {p['diff']:+.2f} CI[{p['ci95'][0]:+.2f},{p['ci95'][1]:+.2f}] p_holm={p['p_holm']:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
