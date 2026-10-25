"""Signer-dependent ceiling + per-class signer-fragility error analysis (P2 item 7).

Uses the E1 paired-instance runs: the SAME frozen TEST clips (U02, U13) scored by
BlockGCN trained (a) WITHOUT the test signer [unseen / SI] and (b) WITH a matched
number of that signer's clips added [seen_matched / signer-dependent ceiling].
Because the clips are identical across the two conditions, the per-class gap is a
clean paired estimate of how much each sign relies on having seen the signer.

Scores are averaged over the 3 training seeds (per-clip mean logits -> argmax).

Outputs a markdown block: (1) the ceiling headline, (2) the most signer-FRAGILE
signs (largest seen-minus-unseen drop) and the most signer-ROBUST signs.

  python tools/signer_fragility.py
"""
import os, json, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
SIGNERS = ["U02", "U13"]
COND = {"unseen": "unseen", "seen_matched": "seen_matched"}


def load_labels():
    names, labels = pickle.load(open(os.path.join(ROOT, "data/bdsl_si/test_label.pkl"), "rb"))
    return dict(zip(names, labels))


def load_classes():
    c = json.load(open(os.path.join(ROOT, "data/bdsl_si/classes.json")))
    return c["classes"]  # idx -> name


def per_seed_pkls(signer, cond):
    out = []
    for s in SEEDS:
        p = os.path.join(ROOT, f"work_dir/bdsl_block_gcn_si_pairsmg_{signer}_{cond}_seed{s}_test/eval_results/best_acc.pkl")
        if os.path.exists(p):
            out.append(pickle.load(open(p, "rb")))
    return out


def per_class_perseed(pkls, name2lab, C=60):
    """Per-seed per-class hit/total, averaged over seeds (paper convention:
    average per-seed accuracy, NOT logit-ensemble which inflates the number)."""
    cor = np.zeros(C); tot = np.zeros(C)
    for d in pkls:
        c = np.zeros(C); t = np.zeros(C)
        for clip, sc in d.items():
            y = name2lab[clip]; t[y] += 1
            if int(np.argmax(np.asarray(sc))) == y:
                c[y] += 1
        cor += c; tot += t
    # tot counts each clip once per seed -> per-class accuracy = cor/tot is the
    # seed-averaged per-class accuracy; clip count reported as tot/len(pkls).
    nseed = max(len(pkls), 1)
    return cor, tot, nseed


def main():
    name2lab = load_labels()
    classes = load_classes()

    # pool U02 + U13 identical-clip sets for each condition (seed-averaged)
    tot_correct = {c: np.zeros(60) for c in COND}
    tot_total = {c: np.zeros(60) for c in COND}   # clip-count * nseed
    nseed_ref = 3
    for cond in COND:
        for sg in SIGNERS:
            pkls = per_seed_pkls(sg, cond)
            cor, tot, nseed = per_class_perseed(pkls, name2lab)
            nseed_ref = nseed
            tot_correct[cond] += cor
            tot_total[cond] += tot

    seen_acc = 100 * tot_correct["seen_matched"].sum() / tot_total["seen_matched"].sum()
    uns_acc = 100 * tot_correct["unseen"].sum() / tot_total["unseen"].sum()
    nclips = int(tot_total["unseen"].sum() / nseed_ref)

    print("## Signer-dependent ceiling + per-class fragility (E1 paired, identical clips)\n")
    print(f"Clips pooled: U02+U13 held-out TEST ({nclips} clips), "
          f"{nseed_ref} seeds (per-seed accuracy averaged, paper convention).\n")
    print(f"- **SI (signer unseen):** {uns_acc:.2f}%")
    print(f"- **Signer-dependent ceiling (same signer's clips in train, matched size):** "
          f"{seen_acc:.2f}%")
    print(f"- **Ceiling gap (paired, identical clips):** +{seen_acc - uns_acc:.2f} pp  "
          f"— the headroom a perfect signer-adaptation method could recover.\n")

    # per-class paired delta (only classes present in test)
    present = tot_total["unseen"] > 0
    seen_c = np.where(tot_total["seen_matched"] > 0,
                      100 * tot_correct["seen_matched"] / np.maximum(tot_total["seen_matched"], 1), np.nan)
    uns_c = np.where(present,
                     100 * tot_correct["unseen"] / np.maximum(tot_total["unseen"], 1), np.nan)
    delta = seen_c - uns_c

    idx = [i for i in range(60) if present[i]]
    idx_sorted = sorted(idx, key=lambda i: -delta[i])

    def row(i):
        return (f"| {classes[i]} | {uns_c[i]:.1f} | {seen_c[i]:.1f} "
                f"| {'+' if delta[i]>=0 else ''}{delta[i]:.1f} | {int(tot_total['unseen'][i]/nseed_ref)} |")

    print("### Most signer-FRAGILE signs (largest seen−unseen drop)\n")
    print("| sign | SI acc | ceiling acc | Δ (fragility) | n clips |")
    print("|---|---|---|---|---|")
    for i in idx_sorted[:12]:
        print(row(i))

    print("\n### Most signer-ROBUST signs (generalize across signers)\n")
    print("| sign | SI acc | ceiling acc | Δ | n clips |")
    print("|---|---|---|---|---|")
    for i in idx_sorted[::-1][:8]:
        print(row(i))

    # summary stats
    d = delta[idx]
    print(f"\nFragility spread: median Δ {np.nanmedian(d):+.1f} pp, "
          f"IQR [{np.nanpercentile(d,25):+.1f}, {np.nanpercentile(d,75):+.1f}]; "
          f"{int((d<=1).sum())}/{len(idx)} signs already at/near ceiling under SI "
          f"(signer-robust), {int((d>=40).sum())} signs lose ≥40 pp.")


if __name__ == "__main__":
    main()
