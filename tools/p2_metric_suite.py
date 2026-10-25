#!/usr/bin/env python3
"""P2 full metric suite + error analysis across ALL architectures, from saved
softmax score files (NO retraining). Produces:
  (C) per-arch held-out-TEST leaderboard: Top-1/5, macro-F1, MRR, R@10,
      ECE (equal-mass), NLL, Brier  (3-seed ensemble).
  (F) error analysis: per-signer accuracy (TEST signers) + hardest classes.
Reads work_dir/<stem>{_seed<N>_test,_test_seed<N>}/eval_results/best_acc.pkl
against data/bdsl_si/test_label.pkl. Writes docs/P2_METRIC_SUITE.md + .json.
"""
import os, glob, pickle, json, re
import numpy as np

TEST_LABEL = "data/bdsl_si/test_label.pkl"
ARCHS = [   # (display, work_dir stem) — Track A (shared recipe), the Table-2 runs (NOT _bp = Track B)
    ("Adaptive GNN", "bdsl_adaptive_gnn_si"),
    ("ST-GCN", "bdsl_st_gcn_vanilla_si"),
    ("Attention GNN", "bdsl_gnn_si"),
    ("SLGTFormer", "bdsl_slgtformer_si"),
    ("GNN+Bi-LSTM", "bdsl_gnn_lstm_si"),
    ("BlockGCN", "bdsl_block_gcn_si"),
    ("Pose-LSTM", "bdsl_pose_lstm_si"),
    ("GNN+Transformer", "bdsl_gnn_transformer_si"),
    ("VideoMAE (RGB)", "rgb_videomae_bdsl_si"),
]

def load_labels():
    names, labels = pickle.load(open(TEST_LABEL, "rb"))
    return dict(zip(names, labels))

def find_seed_pkls(stem):
    pats = [f"work_dir/{stem}_seed*_test/eval_results/best_acc.pkl",
            f"work_dir/{stem}_test_seed*/eval_results/best_acc.pkl",
            f"work_dir/{stem}_seed*_test/eval_results/*.pkl",
            f"work_dir/{stem}_test_seed*/eval_results/*.pkl"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            # one per seed dir
            byseed = {}
            for h in hits:
                m = re.search(r"seed(\d+)", h)
                if m: byseed.setdefault(m.group(1), h)
            if byseed: return list(byseed.values())
    return []

def to_prob(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1: v = v.ravel()
    s = v.sum()
    if v.min() >= 0 and abs(s - 1) < 1e-3:      # already prob
        return v
    e = np.exp(v - v.max()); return e / e.sum()  # logits -> softmax

def ece_equal_mass(conf, correct, bins=15):
    order = np.argsort(conf); conf = conf[order]; correct = correct[order]
    n = len(conf); e = 0.0
    for b in np.array_split(np.arange(n), bins):
        if len(b) == 0: continue
        e += (len(b) / n) * abs(correct[b].mean() - conf[b].mean())
    return e * 100

def _metrics_one(P, y):
    """All scalar metrics for one probability matrix P (N,C) and labels y."""
    N, C = P.shape
    pred = P.argmax(1)
    ranks = np.array([(P[i] > P[i, y[i]]).sum() for i in range(N)])
    f1s = []
    for c in range(C):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        if tp + fp + fn == 0: continue
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    conf = P.max(1); correct = (pred == y).astype(float)
    onehot = np.zeros_like(P); onehot[np.arange(N), y] = 1
    return dict(top1=(pred == y).mean() * 100,
                top5=np.mean([y[i] in P[i].argsort()[-5:] for i in range(N)]) * 100,
                macrof1=np.mean(f1s) * 100, mrr=float(np.mean(1.0 / (ranks + 1))),
                r10=np.mean(ranks < 10) * 100, ece=ece_equal_mass(conf, correct),
                nll=-np.mean([np.log(max(P[i, y[i]], 1e-12)) for i in range(N)]),
                brier=float(np.mean(((P - onehot) ** 2).sum(1))), correct=correct)

def metrics_for(stem, lab):
    pkls = find_seed_pkls(stem)
    if not pkls: return None
    per = [{k: to_prob(v) for k, v in pickle.load(open(p, "rb")).items()} for p in pkls]
    common = sorted(set(lab) & set.intersection(*[set(d) for d in per]))
    if not common: return None
    y = np.array([lab[k] for k in common])
    # single-model: metrics per seed -> mean/std (matches Table 2 convention)
    keys = ["top1", "top5", "macrof1", "mrr", "r10", "ece", "nll", "brier"]
    per_seed = [_metrics_one(np.stack([d[k] for k in common]), y) for d in per]
    single = {k: (float(np.mean([m[k] for m in per_seed])), float(np.std([m[k] for m in per_seed]))) for k in keys}
    # ensemble (secondary)
    Pe = np.stack([np.mean([d[k] for d in per], axis=0) for k in common])
    ens = _metrics_one(Pe, y)
    # per-signer + hardest classes (from ensemble correctness)
    correct = ens["correct"]
    sig = {}
    for i, k in enumerate(common):
        mm = re.match(r"U(\d+)", k); s = mm.group(1) if mm else "?"
        sig.setdefault(s, []).append(correct[i])
    per_signer = {s: (float(np.mean(v)) * 100, len(v)) for s, v in sorted(sig.items())}
    cls_acc = {c: float(correct[np.where(y == c)[0]].mean()) * 100 for c in range(Pe.shape[1]) if (y == c).any()}
    hardest = sorted(cls_acc.items(), key=lambda kv: kv[1])[:8]
    return dict(seeds=len(pkls), n=len(common), single=single,
                ens={k: ens[k] for k in keys}, per_signer=per_signer, hardest=hardest)

def main():
    lab = load_labels()
    rows = {}
    for disp, stem in ARCHS:
        m = metrics_for(stem, lab)
        rows[disp] = m
        if m:
            s = m['single']
            print(f"{disp:18s} n={m['n']} seeds={m['seeds']}  "
                  f"Top1 {s['top1'][0]:.2f}±{s['top1'][1]:.2f} (ens {m['ens']['top1']:.2f})  "
                  f"F1 {s['macrof1'][0]:.2f}  ECE {s['ece'][0]:.2f}  NLL {s['nll'][0]:.3f}  Brier {s['brier'][0]:.3f}")
        else:
            print(f"{disp:18s} -- no score files found (stem {stem})")
    os.makedirs("docs", exist_ok=True)
    json.dump(rows, open("docs/P2_METRIC_SUITE.json", "w"), indent=2, default=float)
    with open("docs/P2_METRIC_SUITE.md", "w") as f:
        f.write("# P2 full metric suite (held-out TEST)\n\n")
        f.write("Primary = single-model **3-seed mean ± std** (matches Table 2 convention). "
                "Top-1(ens) = 3-seed softmax ensemble.\n\n")
        f.write("| Arch | Top-1 (mean±std) | Top-1(ens) | Top-5 | Macro-F1 | MRR | R@10 | ECE% | NLL | Brier |\n"
                "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for disp, _ in ARCHS:
            m = rows[disp]
            if not m: continue
            s = m['single']
            f.write(f"| {disp} | {s['top1'][0]:.2f}±{s['top1'][1]:.2f} | {m['ens']['top1']:.2f} | "
                    f"{s['top5'][0]:.2f} | {s['macrof1'][0]:.2f} | {s['mrr'][0]:.3f} | {s['r10'][0]:.2f} | "
                    f"{s['ece'][0]:.2f} | {s['nll'][0]:.3f} | {s['brier'][0]:.3f} |\n")
        f.write("\n## Per-signer TEST accuracy (Top-1, ensemble)\n")
        for disp, _ in ARCHS:
            m = rows[disp]
            if m: f.write(f"- **{disp}**: " + ", ".join(f"U{s} {a:.1f} (n={n})" for s,(a,n) in m['per_signer'].items()) + "\n")
        f.write("\n## Hardest classes (lowest Top-1, BlockGCN)\n")
        if rows.get("BlockGCN"):
            f.write(", ".join(f"cls{c}:{a:.0f}%" for c,a in rows["BlockGCN"]['hardest']) + "\n")
    print("[wrote docs/P2_METRIC_SUITE.md + .json]")

if __name__ == "__main__":
    main()
