#!/usr/bin/env python
"""P2 analysis 3: is the SMG ordering a ceiling artifact?

Mixed (signer-dependent) accuracy saturates near 98-99%, so
SMG = A_mixed - A_SI is mechanically ~ (98 - A_SI) and largest for weak models.
For each of the 8 pose architectures we compare three gap scales:
  SMG        = mean(A_mixed) - mean(A_SI)                      (pp, the paper's)
  error ratio = (100 - mean A_SI) / (100 - mean A_mixed)       (x-fold error inflation)
  log-odds gap = logit(mean A_mixed) - logit(mean A_SI)        (ceiling-robust)

Inputs are EXACTLY the per-seed sources of the paper's tab:smg8 rows: they are
resolved through tools/p2_make_tables.py (POSE spec, `mixed` and `test` sources),
so no accuracy is recomputed differently from the paper.

Uncertainty: seeds are separate runs per split (no natural pairing), so CIs come
from an exact seed bootstrap — all 3^3 x 3^3 = 729 resamples of the 3 mixed and
3 SI seeds (independently) per arch; percentile 95% CI. With n=3 per split this
under-covers; read the CIs as optimistic. Rank stability: 20k Monte Carlo draws
of one bootstrap resample per arch -> ranking per metric; we report P(rank order of
the top / bottom arch) and Kendall tau between metric rankings.

Output: docs/p2_analysis_2026-10/smg_ceiling.json (+ .md written by hand from it)
"""
import itertools, json, os, sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import p2_make_tables as T  # noqa: E402  (chdir's to ROOT)

OUT = os.path.join(ROOT, "docs/p2_analysis_2026-10/smg_ceiling.json")


def logit(p):
    p = np.clip(np.asarray(p, float) / 100.0, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metrics(mixed_mean, si_mean):
    return dict(smg=mixed_mean - si_mean,
                err_ratio=(100 - si_mean) / (100 - mixed_mean),
                logodds_gap=float(logit(mixed_mean) - logit(si_mean)))


def main():
    ctx = T.Ctx()
    rows = {}
    for arch, spec in T.POSE.items():
        src = ctx.pose_src(arch)
        test, mixed = T.resolve(src["test"]), T.resolve(src["mixed"])
        mv, sv = np.array(mixed.values), np.array(test.values)
        point = metrics(mv.mean(), sv.mean())
        boots = {k: [] for k in point}
        for mi in itertools.product(range(len(mv)), repeat=len(mv)):
            for si in itertools.product(range(len(sv)), repeat=len(sv)):
                m = metrics(mv[list(mi)].mean(), sv[list(si)].mean())
                for k in m:
                    boots[k].append(m[k])
        rows[spec["names"]["smg8"]] = dict(
            arch_key=arch, mixed_seeds=mv.tolist(), si_seeds=sv.tolist(),
            mixed_mean=float(mv.mean()), mixed_sd_pop=float(mv.std()), si_mean=float(sv.mean()),
            si_sd_pop=float(sv.std()), err_mixed=float(100 - mv.mean()), err_si=float(100 - sv.mean()),
            mixed_src=mixed.src, si_src=test.src,
            **{k: float(v) for k, v in point.items()},
            **{f"{k}_ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] for k, b in boots.items()},
            _boot={k: np.array(b) for k, b in boots.items()})

    names = list(rows)
    keys = ["smg", "err_ratio", "logodds_gap"]
    ranks = {}
    for k in keys:
        order = sorted(names, key=lambda n: -rows[n][k])
        ranks[k] = order
    si_order = sorted(names, key=lambda n: rows[n]["si_mean"])  # weakest first
    # correlations of each gap metric with SI accuracy (mechanical coupling) and with mixed error
    corr = {}
    for k in keys:
        x = [rows[n][k] for n in names]
        corr[k] = dict(
            spearman_vs_si=stats.spearmanr(x, [rows[n]["si_mean"] for n in names])[0],
            spearman_vs_mixed=stats.spearmanr(x, [rows[n]["mixed_mean"] for n in names])[0])
    tau = {f"{a}_vs_{b}": float(stats.kendalltau([ranks[a].index(n) for n in names],
                                                 [ranks[b].index(n) for n in names])[0])
           for a, b in itertools.combinations(keys, 2)}

    # Monte Carlo rank stability
    rng = np.random.default_rng(0)
    B = 20000
    stab = {}
    for k in keys:
        draws = np.stack([rows[n]["_boot"][k][rng.integers(0, 729, B)] for n in names], 1)  # (B, A)
        order_idx = np.argsort(-draws, 1)
        top = [names[i] for i in order_idx[:, 0]]
        bottom = [names[i] for i in order_idx[:, -1]]
        rank_of = np.argsort(order_idx, 1)  # rank position per arch
        stab[k] = dict(
            p_top={n: float(np.mean([t == n for t in top])) for n in names},
            p_bottom={n: float(np.mean([b == n for b in bottom])) for n in names},
            median_rank={n: float(np.median(rank_of[:, i]) + 1) for i, n in enumerate(names)},
            rank_ci95={n: [int(np.percentile(rank_of[:, i], 2.5) + 1), int(np.percentile(rank_of[:, i], 97.5) + 1)]
                       for i, n in enumerate(names)})
        # Kendall tau of each MC draw's ranking vs the SI-accuracy ranking (weakest first)
    # does the 'largest gap for the weakest models' claim survive? tau between gap and -SI ranking per draw
    claim = {}
    for k in keys:
        draws = np.stack([rows[n]["_boot"][k][rng.integers(0, 729, B)] for n in names], 1)
        si = np.array([rows[n]["si_mean"] for n in names])
        taus = np.array([stats.kendalltau(d, -si)[0] for d in draws[:4000]])
        claim[k] = dict(kendall_tau_gap_vs_weakness_mean=float(taus.mean()),
                        ci95=[float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))],
                        p_tau_positive=float(np.mean(taus > 0)))

    for n in names:
        rows[n].pop("_boot")
    out = dict(
        source="tools/p2_make_tables.py POSE spec (tab:smg8 mixed/test sources), resolved live",
        ci_method="exact seed bootstrap (729 resamples/arch, mixed and SI seeds resampled independently), percentile 95%",
        rows=rows, rankings=ranks, si_order_weakest_first=si_order, spearman=corr, kendall_tau_between_metrics=tau,
        rank_stability_mc=stab, weak_model_claim=claim)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    # console table
    print(f"{'arch':16s} {'mixed':>12s} {'SI':>12s} {'SMG':>16s} {'errRatio':>18s} {'logodds':>18s}")
    for n in ranks["smg"]:
        r = rows[n]
        print(f"{n:16s} {r['mixed_mean']:6.2f}±{r['mixed_sd_pop']:.2f} {r['si_mean']:6.2f}±{r['si_sd_pop']:.2f} "
              f"{r['smg']:6.2f}[{r['smg_ci95'][0]:.1f},{r['smg_ci95'][1]:.1f}] "
              f"{r['err_ratio']:6.2f}[{r['err_ratio_ci95'][0]:.1f},{r['err_ratio_ci95'][1]:.1f}] "
              f"{r['logodds_gap']:5.2f}[{r['logodds_gap_ci95'][0]:.2f},{r['logodds_gap_ci95'][1]:.2f}]")
    for k in keys:
        print(k, "ranking (largest first):", ranks[k])
    print("spearman", json.dumps(corr, indent=0))
    print("tau", tau)
    print("claim", json.dumps(claim, indent=0))
    for k in keys:
        print(k, "median rank", stab[k]["median_rank"], "rankCI", stab[k]["rank_ci95"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
