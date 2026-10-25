"""Render the P2 (NeurIPS D&B) publication figures as vector PDFs.

Top-tier venue standard (see tools/_figstyle.py): clean panels, Okabe-Ito
colorblind-safe palette, despined serif axes, editable-text vector PDF, direct
labelling over legends where possible, generous margins so nothing overlaps.
Panels carry axes + data + minimal labels only; every descriptive sentence lives
in the LaTeX caption ("figure footnote"), not in the panel.

Numbers are the FINALIZED values (docs/P2_FINAL_NUMBERS.md).

Outputs:
  results/figures/fig1_smg.pdf        signer-mixing gap (matched-split, 2 models)
  results/figures/fig2_loso.pdf       unseen-signer distribution (11 signers)
  results/figures/fig3_pareto.pdf     accuracy vs signer-leakage (GRL, negative)
  results/figures/fig4_diversity.pdf  signer diversity: grid + SMG(k) decay law
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from tools._figstyle import (apply_style, despine, light_grid,
                             BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, GREY, INK)

apply_style()


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_smg(out):
    """Matched-split SMG (the headline): signer-mixed vs signer-disjoint for the
    two probed models, with the range over the 5 random splits on the mixed condition. Dumbbell so
    the gap itself is the visual object. Every mixed checkpoint is selected on its split's own VAL.
    BlockGCN lr 0.01: docs/blockgcn_lr01_switch/03_matched_splits.md (val-selected 98.12 [98.02, 98.24]).
    Adaptive GNN lr 0.1 (Table 7 recipe): docs/p2_rerun_fixes_2026-10/raw/adaptive_gnn_rand_lr0p1_valsel_test.txt
    (98.83/99.12/98.53/99.12/98.75) vs Protocol-A TEST 77.05 (bdsl_adaptive_gnn_si_lr0p1_seed*_test)."""
    models = ["Adaptive GNN", "BlockGCN"]
    si    = [77.05, 68.82]
    mixed = [98.87, 98.12]
    cilo  = [98.53, 98.02]   # min over the 5 splits (max in cihi)
    cihi  = [99.12, 98.24]
    gap   = [21.82, 29.30]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    y = np.arange(len(models))
    for i in range(len(models)):
        ax.plot([si[i], mixed[i]], [y[i], y[i]], color="#C4CDD6", lw=3.0,
                solid_capstyle="round", zorder=1)
        ax.errorbar(mixed[i], y[i],
                    xerr=[[mixed[i] - cilo[i]], [cihi[i] - mixed[i]]],
                    fmt="none", ecolor=VERM, elinewidth=1.3, capsize=2.5, zorder=2)
        ax.annotate(fr"$+{gap[i]:.1f}$ pp", ((si[i] + mixed[i]) / 2, y[i]),
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.scatter(si, y, s=95, color=BLUE, zorder=3, edgecolor="white", lw=1.1)
    ax.scatter(mixed, y, s=95, color=VERM, zorder=3, edgecolor="white", lw=1.1)
    # direct labels (no legend box), placed in the empty mid-band on the first row
    ax.annotate("signer-disjoint\n(held-out TEST)", (si[0], y[0]), xytext=(-8, -20),
                textcoords="offset points", ha="right", va="top", fontsize=8.3,
                color=BLUE)
    ax.annotate("signer-mixed\n(5 random splits, range)", (mixed[0], y[0]),
                xytext=(4, -20), textcoords="offset points", ha="right", va="top",
                fontsize=8.3, color=VERM)
    ax.set_yticks(y); ax.set_yticklabels(models)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_xlim(60, 103); ax.set_ylim(-0.7, len(models) - 0.15)
    despine(ax); light_grid(ax, "x")
    _save(fig, out)


def fig_loso(out):
    """Forest/dot plot of the 11 unseen-signer folds. Value column on the right
    (never overlaps the CI band); CI band + mean line in the background."""
    # TRUE held-out-test-signer LOSO (BlockGCN, base LR 0.01), read from docs/loso_heldout/block_gcn_lr01.csv:
    # fold value = mean over seeds; 95% CI = signer-clustered percentile bootstrap
    # over the 11 folds (10k resamples, seed 1234), as in the paper's LOSO table.
    import csv, collections
    per = collections.defaultdict(list)
    with open("docs/loso_heldout/block_gcn_lr01.csv") as f:
        for r in csv.DictReader(f):
            if r["test_acc"]:
                per[r["heldout_signer"]].append(float(r["test_acc"]))
    folds = [float(np.mean(v)) for v in per.values()]          # file (fold) order
    rng = np.random.default_rng(1234)
    boots = [rng.choice(folds, len(folds)).mean() for _ in range(10000)]
    mean = float(np.mean(folds)); lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    sig = sorted(per, key=lambda k: np.mean(per[k]))
    acc = [float(np.mean(per[k])) for k in sig]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    y = np.arange(len(sig))
    xr = 99.0
    ax.axvspan(lo, hi, color=BLUE, alpha=0.09, lw=0, zorder=0)
    ax.axvline(mean, color=BLUE, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.hlines(y, 0, acc, color="#CBD5DE", lw=1.5, zorder=2)
    ax.scatter(acc, y, s=52, color=BLUE, zorder=4, edgecolor="white", lw=0.8)
    for yi, a in zip(y, acc):                    # tidy right-aligned value column
        ax.annotate(f"{a:.1f}", (xr, yi), ha="right", va="center",
                    fontsize=8.2, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(sig, fontsize=8.6)
    ax.set_ylabel("held-out signer")
    ax.set_xlabel("held-out-signer Top-1 (%)")
    ax.set_xlim(0, xr + 2); ax.set_ylim(-0.7, len(sig) - 0.3)
    ax.set_xticks([0, 20, 40, 60, 80])
    # one compact in-axes note for the summary stats (no legend box)
    ax.annotate(f"mean {mean:.1f}\n95% CI [{lo:.1f}, {hi:.1f}]",
                (mean, len(sig) - 0.6), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="top", fontsize=8.2, color=BLUE, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0))
    despine(ax); light_grid(ax, "x")
    _save(fig, out)


def fig_pareto(out):
    """GRL negative result, 3 seeds per arm (docs/tta_grl_2026-10/grl_3seed.md): leakage barely falls
    (only at lambda=0.1) while accuracy drops 22-24 pp at every lambda -> no usable frontier.
    lr-0.01 recipe, no clipping. Points = mean over seeds 0/1/2; error bars = population SD over seeds
    on both axes. L_norm = class-conditioned probe of tab:grl (ungrouped, training-signer clips)."""
    lam = [0.0, 0.1, 0.5, 1.0]
    acc_s = [[68.06, 69.01, 69.38], [50.70, 46.23, 43.37], [43.81, 40.59, 48.64], [45.27, 44.10, 48.64]]
    lk_s = [[0.824, 0.871, 0.863], [0.804, 0.786, 0.784], [0.832, 0.828, 0.807], [0.832, 0.849, 0.833]]
    acc = [float(np.mean(v)) for v in acc_s]; acc_sd = [float(np.std(v)) for v in acc_s]
    leak = [float(np.mean(v)) for v in lk_s]; leak_sd = [float(np.std(v)) for v in lk_s]
    off = [(0, 10), (10, 14), (12, -12), (-12, 6)]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(leak, acc, "-", color="#C4CDD6", lw=1.6, zorder=1)
    for i, l in enumerate(lam):
        c = BLUE if i == 0 else VERM
        ax.errorbar(leak[i], acc[i], xerr=leak_sd[i], yerr=acc_sd[i], fmt="none", ecolor=c,
                    elinewidth=1.1, capsize=2.5, zorder=2)
        ax.scatter(leak[i], acc[i], s=100, color=c, zorder=3, edgecolor="white", linewidth=1.1)
        ha = "center" if off[i][0] == 0 else ("left" if off[i][0] > 0 else "right")
        ax.annotate(fr"$\lambda={l:g}$", (leak[i], acc[i]), xytext=off[i],
                    textcoords="offset points", fontsize=9.5, color=c, ha=ha)
    ax.set_xlabel(r"signer decodability $L_{\mathrm{norm}}$  (less leakage $\rightarrow$)")
    ax.set_ylabel("SI accuracy (%)")
    ax.invert_xaxis()
    ax.set_xlim(0.885, 0.770); ax.set_ylim(37, 75)
    despine(ax); light_grid(ax, "both")
    _save(fig, out)


def fig_diversity(out):
    """Two panels: (a) SI accuracy rises with #signers (weakly with #clips);
    (b) the signer-mixing gap SMG(k) decays as training signers grow, within
    BdSLW60 and (further) within AUTSL -- the controlled diversity law."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 3.2))

    # (a) grid as lines with honest error bars
    s = np.array([2, 4, 6, 8])
    # BlockGCN base LR 0.01 (docs/blockgcn_lr01_switch/09_diversity_grid.md)
    m8, e8 = [26.9, 40.9, 37.8, 54.6], [1.6, 2.5, 0.5, 1.6]
    m15, e15 = [30.2, 38.5, 50.7, 57.6], [0.7, 3.6, 0.4, 3.4]
    axL.errorbar(s - 0.06, m8, yerr=e8, marker="o", ms=5, color=BLUE, lw=1.8,
                 capsize=2.5, label="800 clips")
    axL.errorbar(s + 0.06, m15, yerr=e15, marker="s", ms=5, color=ORANGE, lw=1.8,
                 capsize=2.5, label="1,500 clips")
    axL.set_xticks(s)
    axL.set_xlabel("training signers")
    axL.set_ylabel("SI Top-1 (%)")
    axL.set_ylim(24, 72)
    axL.legend(loc="upper left", fontsize=8.4, handlelength=1.4)
    axL.set_title("(a)", fontsize=10, loc="left", fontweight="bold")
    despine(axL); light_grid(axL, "y")

    # (b) SMG(k) decay law across FIVE within-dataset curves (3 languages).
    # Each curve: 5 random size-k signer draws per cell (error bar = SD over draws);
    # the largest-k point per dataset is the single full-pool draw (no error bar).
    # k on a log2 axis so the doublings are evenly spaced over the 2..74 range.
    # ASL-Citizen omits its degenerate k=4 floor (4 signers cannot cover 2731
    # classes -> ~1% acc); the thin 60-class variant is the appendix density check.
    curves = [
        # label,                         k,                 SMG,                          SD,                       color,  marker
        # BdSLW60: BlockGCN base LR 0.01 (docs/blockgcn_lr01_switch/10_smg_k.md).
        # BdSLW401: 15-signer pool, selected on held-out signer S15
        # (docs/improvement_runs_2026-10/item2_w401_smgk_sdval.md).
        ("BdSLW60 (Bangla, 8)",   [2, 4, 6, 8],       [61.6, 44.3, 38.7, 30.4],   [3.8, 3.1, 2.3, 0.0],   BLUE,   "o"),
        ("BdSLW401 (Bangla, 15)", [2, 4, 8, 12, 15],  [77.0, 61.8, 28.0, 16.9, 14.3], [8.7, 6.5, 5.3, 1.3, 0.0], SKY, "D"),
        ("AUTSL (Turkish, 31)",   [4, 8, 16, 24, 31], [45.7, 18.2, 8.0, 5.4, 4.7], [1.1, 2.5, 0.5, 0.3, 0.0], ORANGE, "s"),
        ("LSFB (Bel.-French, 74)",[2, 4, 8, 16, 32, 74], [11.7, 10.8, 10.9, 8.6, 8.1, 6.6], [2.9, 2.9, 1.4, 3.7, 0.7, 0.0], GREEN, "^"),
        ("ASL-Citizen (ASL, 35)", [8, 16, 32, 35],    [4.9, 7.4, 5.0, 3.6],        [6.3, 2.8, 1.8, 0.0],   PURPLE, "v"),
    ]
    for label, ks, smg, sd, color, mk in curves:
        axR.errorbar(ks, smg, yerr=sd, marker=mk, ms=4.4, color=color, lw=1.6,
                     capsize=2.0, elinewidth=0.8, label=label)
    axR.set_xscale("log", base=2)
    axR.set_xticks([2, 4, 8, 16, 32, 74])
    axR.set_xticklabels(["2", "4", "8", "16", "32", "74"])
    axR.set_xlabel(r"training signers $k$ (log scale)")
    axR.set_ylabel(r"signer-mixing gap SMG$(k)$ (pp)")
    axR.set_ylim(-2, 90)
    axR.axhline(0, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=0)
    axR.legend(loc="upper right", fontsize=6.7, handlelength=1.3,
               labelspacing=0.3, borderpad=0.4)
    axR.set_title("(b)", fontsize=10, loc="left", fontweight="bold")
    despine(axR); light_grid(axR, "y")

    fig.subplots_adjust(wspace=0.28)
    _save(fig, out)


def fig_aslc_density(out):
    """Appendix density cross-check: the SAME SMG(k) probe on ASL-Citizen at two
    vocabulary scales. Full-vocab (2731 classes, dense: median 425 clips/signer)
    yields a small-but-positive gap; thinning to the 60 most-frequent classes
    (median 19 clips/signer) starves the seen-holdout estimate so SMG collapses
    into sampling noise around zero -- the metric needs enough clips/signer to be
    measurable. Single panel; every sentence lives in the LaTeX caption."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    # full-vocab (dense) -- includes the degenerate k=4 floor here, to show it
    ax.errorbar([4, 8, 16, 32, 35], [0.4, 4.9, 7.4, 5.0, 3.6],
                yerr=[0.7, 6.3, 2.8, 1.8, 0.0], marker="v", ms=5, color=PURPLE,
                lw=1.8, capsize=2.5, elinewidth=0.9,
                label="ASL-Citizen full (2731 cls, dense)")
    # thin 60-class variant -- SMG hovers around/below zero
    ax.errorbar([2, 4, 8, 16, 32], [-3.9, -3.4, -2.2, -3.9, 1.1],
                yerr=[1.7, 3.1, 7.8, 5.2, 0.0], marker="x", ms=6, color=GREY,
                lw=1.8, ls=(0, (5, 2)), capsize=2.5, elinewidth=0.9,
                label="ASL-Citizen-60 (thin)")
    ax.axhline(0, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_xlabel(r"training signers $k$ (log scale)")
    ax.set_ylabel(r"signer-mixing gap SMG$(k)$ (pp)")
    ax.set_ylim(-14, 16)
    ax.legend(loc="upper left", fontsize=7.6, handlelength=1.5)
    despine(ax); light_grid(ax, "y")
    _save(fig, out)


if __name__ == "__main__":
    print("rendering P2 figures (vector PDF)...")
    if sys.argv[1:]:   # e.g. `python tools/make_paper_figures.py smg pareto` renders only those
        for name in sys.argv[1:]:
            fn = {"smg": fig_smg, "loso": fig_loso, "pareto": fig_pareto, "diversity": fig_diversity,
                  "aslc": fig_aslc_density}[name]
            stem = {"smg": "fig1_smg", "loso": "fig2_loso", "pareto": "fig3_pareto",
                    "diversity": "fig4_diversity", "aslc": "figA_aslc_density"}[name]
            fn(f"results/figures/{stem}.pdf")
        sys.exit(0)
    fig_smg("results/figures/fig1_smg.pdf")
    fig_loso("results/figures/fig2_loso.pdf")
    fig_pareto("results/figures/fig3_pareto.pdf")
    fig_diversity("results/figures/fig4_diversity.pdf")
    fig_aslc_density("results/figures/figA_aslc_density.pdf")
    print("done.")
