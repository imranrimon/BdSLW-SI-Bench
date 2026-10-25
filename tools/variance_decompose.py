"""E2 — Variance decomposition of signer-independent (LOSO) accuracy.

The paper's central claim is that *who you hold out* (signer identity) moves
signer-independent accuracy far more than *which model you pick* or *which
random seed you use*. This tool quantifies that from the LOSO sweep already in
`results_final.csv`.

We read every LOSO cell -- one held-out-signer test accuracy per
(architecture, held-out signer, seed) -- from rows named
`<config_stem>_loso_test<SS>_seed<R>`, and decompose the spread of accuracy
into interpretable variance components of a crossed arch x signer design with
seed as replication:

  sigma_signer   : SD of the held-out-signer main effect  (the headline noise)
  sigma_model    : SD of the architecture main effect
  sigma_seed      : within-cell SD across seeds            (init noise only)
  sigma_interact : SD of the arch x signer interaction

Aggregation matches the paper: per (arch, signer) we mean over seeds first, and
the LOSO mean weights each signer equally (mean of per-signer means), NOT each
cell -- so unbalanced seed counts do not bias the signer effect.

Usage:
  python tools/variance_decompose.py --csv results_final.csv --markdown
  python tools/variance_decompose.py --archs bdsl_block_gcn_si bdsl_adaptive_gnn_si
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict


def load_loso(csv_path):
    """best-Top1 per LOSO cell -> acc[arch][signer][seed] (fraction 0..1)."""
    best = defaultdict(float)
    pat = re.compile(r"(.+?)_loso_test(\d+)_seed(\d+)$")
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            m = pat.match(row[1])
            if not m:
                continue
            try:
                t1 = float(row[3])
            except ValueError:
                continue
            best[(m.group(1), int(m.group(2)), int(m.group(3)))] = max(
                best[(m.group(1), int(m.group(2)), int(m.group(3)))], t1)
    acc = defaultdict(lambda: defaultdict(dict))
    for (arch, sig, seed), v in best.items():
        acc[arch][sig][seed] = v
    return acc


def cell_means(acc, arch):
    """(arch,signer) mean over seeds -> {signer: mean_acc}."""
    return {s: statistics.mean(seeds.values()) for s, seeds in acc[arch].items()}


def within_seed_var(acc, archs, signers):
    """Mean within-(arch,signer) variance across seeds = pure init noise."""
    vs = []
    for a in archs:
        for s in signers:
            seeds = acc[a].get(s, {})
            if len(seeds) >= 2:
                vs.append(statistics.pvariance(list(seeds.values())))
    return statistics.mean(vs) if vs else 0.0


def decompose(acc, archs, signers):
    """Crossed arch x signer decomposition on the balanced (arch,signer) grid.

    Returns a dict of variance components (in accuracy fraction^2) plus the
    per-cell mean table m[a][s].
    """
    m = {a: cell_means(acc, a) for a in archs}
    # keep only signers present for ALL archs (balanced grid)
    signers = [s for s in signers if all(s in m[a] for a in archs)]
    grand = statistics.mean(m[a][s] for a in archs for s in signers)

    arch_eff = {a: statistics.mean(m[a][s] for s in signers) - grand for a in archs}
    sig_eff = {s: statistics.mean(m[a][s] for a in archs) - grand for s in signers}
    inter = {(a, s): m[a][s] - grand - arch_eff[a] - sig_eff[s]
             for a in archs for s in signers}

    def var(vals):
        vals = list(vals)
        return statistics.pvariance(vals) if len(vals) > 1 else 0.0

    return {
        "grand": grand,
        "signers": signers,
        "archs": archs,
        "v_signer": var(sig_eff.values()),
        "v_arch": var(arch_eff.values()),
        "v_interact": var(inter.values()),
        "v_seed": within_seed_var(acc, archs, signers),
        "arch_eff": arch_eff,
        "sig_eff": sig_eff,
        "cell": m,
    }


def _pp(v):
    """variance(fraction^2) -> SD in accuracy percentage points."""
    return (v ** 0.5) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_final.csv")
    ap.add_argument("--archs", nargs="*", default=None,
                    help="restrict to these config stems (default: all with full "
                         "signer coverage)")
    ap.add_argument("--markdown", action="store_true",
                    help="also emit a markdown block for the appendix")
    a = ap.parse_args()

    acc = load_loso(a.csv)
    all_signers = sorted({s for arch in acc for s in acc[arch]})

    coverage = {arch: sorted(acc[arch]) for arch in acc}
    print("LOSO coverage (arch: #signers x #cells):")
    for arch in sorted(coverage):
        ncells = sum(len(acc[arch][s]) for s in acc[arch])
        print(f"  {arch:42s} {len(coverage[arch]):2d} signers, {ncells:2d} cells")

    full = [arch for arch in acc if set(coverage[arch]) >= set(all_signers)]
    archs = a.archs if a.archs else (full if len(full) >= 2 else sorted(acc))
    archs = [x for x in archs if x in acc]
    print(f"\nDecomposing on archs (full-coverage default): {archs}")

    r = decompose(acc, archs, all_signers)
    signers = r["signers"]
    print(f"Balanced grid: {len(archs)} archs x {len(signers)} signers "
          f"(signers={signers})")
    print(f"Grand LOSO mean: {r['grand']*100:.2f}%")

    print("\n--- variance components (SD in accuracy pp) ---")
    comps = [("sigma_signer  (held-out signer)", r["v_signer"]),
             ("sigma_model   (architecture)   ", r["v_arch"]),
             ("sigma_interact(arch x signer)  ", r["v_interact"]),
             ("sigma_seed    (init noise)      ", r["v_seed"])]
    for name, v in comps:
        print(f"  {name} = {_pp(v):6.2f} pp   (var={v:.5f})")

    ss = _pp(r["v_signer"]); se = _pp(r["v_seed"]); sm = _pp(r["v_arch"])
    print(f"\nsigma_signer / sigma_seed  = {ss/se:5.1f}x   "
          "(held-out signer vs random seed)")
    if sm > 0:
        print(f"sigma_signer / sigma_model = {ss/sm:5.1f}x   "
              "(held-out signer vs model choice)")

    print("\n--- arch-controlled view (signer vs seed WITHIN each model) ---")
    print(f"  {'arch':42s} {'n_sig':>5s} {'LOSO mean':>9s} {'SD_signer':>9s} {'SD_seed':>7s}")
    for arch in sorted(acc):
        cm = cell_means(acc, arch)             # signer -> mean over seeds
        sig_sd = statistics.pstdev(cm.values()) * 100 if len(cm) > 1 else 0.0
        seed_sd = within_seed_var(acc, [arch], list(acc[arch])) ** 0.5 * 100
        loso = statistics.mean(cm.values()) * 100   # equal-weight per signer
        print(f"  {arch:42s} {len(cm):5d} {loso:9.2f} {sig_sd:9.2f} {seed_sd:7.2f}")
    print("  (SD_signer = held-out-signer spread within one model = the LOSO"
          " headline SD; SD_seed = init noise)")

    print("\nPer-signer LOSO accuracy (mean over archs & seeds):")
    for s in sorted(signers, key=lambda s: statistics.mean(r["cell"][a][s] for a in archs)):
        mu = statistics.mean(r["cell"][a][s] for a in archs) * 100
        print(f"  U{s:02d}: {mu:5.1f}%")

    if a.markdown:
        print("\n" + "=" * 60 + "\nMARKDOWN (appendix)\n" + "=" * 60)
        print("| Variance source | SD (pp) |")
        print("|---|---|")
        print(f"| Held-out signer | {ss:.2f} |")
        print(f"| Architecture | {sm:.2f} |")
        print(f"| Arch x signer | {_pp(r['v_interact']):.2f} |")
        print(f"| Random seed (init) | {se:.2f} |")
        print(f"\n*Signer identity moves LOSO accuracy {ss/se:.1f}x more than the "
              f"random seed and {ss/sm:.1f}x more than the architecture "
              f"({len(archs)} archs x {len(signers)} signers).*")


if __name__ == "__main__":
    main()
