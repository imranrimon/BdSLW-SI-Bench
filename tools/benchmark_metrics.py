"""Full benchmark metric suite — reviewer point #15.

From saved per-clip score dicts (logits) + labels, compute the standardized metric
panel a top-tier benchmark reviewer asked for, well beyond Top-1:

  Recognition   : Top-1 / Top-5 per-instance (P-I)
  Class balance : Top-1 / Top-5 per-class (P-C, macro recall), Macro-F1
  Calibration   : NLL, multiclass Brier, adaptive ECE (equal-count bins)
  Retrieval     : MRR, R@1, R@5, R@10  (rank of true class in the score vector)
  Generalization: per-signer accuracy — mean, SD, worst, 10th-percentile
  Efficiency    : (--config) params, batch-1 latency, peak VRAM, FLOPs (if thop)

Scores are ensembled over seeds by averaging softmax probabilities (correct for the
calibration metrics). Reads the same eval_results/best_acc.pkl files as the bootstrap
tools. Use --source test for held-out numbers, loso to pool folds.

Usage:
  python tools/benchmark_metrics.py --base bdsl_block_gcn_si --source test
  python tools/benchmark_metrics.py --base bdsl_block_gcn_si --source loso --json out.json
  python tools/benchmark_metrics.py --config config/bdsl_block_gcn_si.yaml --efficiency
"""
import argparse, glob, json, pickle, re, sys, time

try:
    import numpy.core as _npc
    sys.modules.setdefault("numpy._core", _npc)
    for _s in ("multiarray", "numeric", "umath", "_multiarray_umath", "numerictypes"):
        try:
            sys.modules.setdefault("numpy._core." + _s, __import__("numpy.core." + _s, fromlist=["x"]))
        except Exception:
            pass
except Exception:
    pass
import numpy as np

SIG_RE = re.compile(r"U(\d+)W")


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _glob(base, source):
    if source == "test":
        g = sorted(glob.glob(f"work_dir/{base}_seed*_test/eval_results/best_acc.pkl"))
    elif source == "loso":
        g = sorted(glob.glob(f"work_dir/{base}_loso_test*_seed*/eval_results/best_acc.pkl"))
    else:
        g = [p for p in sorted(glob.glob(f"work_dir/{base}_seed*/eval_results/best_acc.pkl"))
             if "_test/" not in p]
    return g


def load_probs(base, source):
    """-> (clips list, P (C,K) mean softmax over seeds/folds)."""
    pkls = _glob(base, source)
    if not pkls:
        return None, None, 0
    acc = {}
    for p in pkls:
        d = pickle.load(open(p, "rb"))
        for k, v in d.items():
            acc.setdefault(k, []).append(_softmax(np.asarray(v, dtype=np.float64)))
    clips = sorted(acc)
    P = np.stack([np.mean(acc[c], axis=0) for c in clips])
    return clips, P, len(pkls)


def load_labels(path):
    names, labels = pickle.load(open(path, "rb"))
    return {n: int(l) for n, l in zip(names, labels)}


def ece_adaptive(conf, correct, n_bins=15):
    """Adaptive (equal-count) ECE."""
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    e = 0.0
    for chunk_c, chunk_a in zip(np.array_split(conf, n_bins), np.array_split(correct, n_bins)):
        if len(chunk_c) == 0:
            continue
        e += (len(chunk_c) / len(conf)) * abs(chunk_a.mean() - chunk_c.mean())
    return e


def compute(clips, P, lab):
    keep = [i for i, c in enumerate(clips) if c in lab]
    clips = [clips[i] for i in keep]
    P = P[keep]
    y = np.array([lab[c] for c in clips])
    sig = np.array([int(m.group(1)) if (m := SIG_RE.search(c)) else -1 for c in clips])
    K = P.shape[1]
    order = np.argsort(-P, axis=1)                     # ranked class indices
    top1 = order[:, 0]
    rank_true = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])  # 0-based
    m = {}
    m["n_clips"] = len(y)
    m["top1_PI"] = float((top1 == y).mean() * 100)
    m["top5_PI"] = float((rank_true < 5).mean() * 100)
    # per-class recall
    cls = np.unique(y)
    r1 = [ (top1[y == c] == c).mean() for c in cls ]
    r5 = [ (rank_true[y == c] < 5).mean() for c in cls ]
    m["top1_PC"] = float(np.mean(r1) * 100)
    m["top5_PC"] = float(np.mean(r5) * 100)
    # macro-F1 on top-1
    f1s = []
    for c in cls:
        tp = np.sum((top1 == c) & (y == c)); fp = np.sum((top1 == c) & (y != c)); fn = np.sum((top1 != c) & (y == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    m["macroF1"] = float(np.mean(f1s) * 100)
    # calibration
    p_true = P[np.arange(len(y)), y]
    m["NLL"] = float(-np.log(np.clip(p_true, 1e-12, 1)).mean())
    onehot = np.zeros_like(P); onehot[np.arange(len(y)), y] = 1
    m["Brier"] = float(((P - onehot) ** 2).sum(axis=1).mean())
    conf = P.max(axis=1); correct = (top1 == y).astype(np.float64)
    m["ECE"] = float(ece_adaptive(conf, correct) * 100)
    # retrieval
    m["MRR"] = float((1.0 / (rank_true + 1)).mean())
    m["R@1"] = float((rank_true < 1).mean() * 100)
    m["R@5"] = float((rank_true < 5).mean() * 100)
    m["R@10"] = float((rank_true < 10).mean() * 100)
    # per-signer generalization
    per = np.array([correct[sig == s].mean() for s in np.unique(sig[sig >= 0])]) * 100
    m["signer_mean"] = float(per.mean()); m["signer_SD"] = float(per.std())
    m["signer_worst"] = float(per.min()); m["signer_p10"] = float(np.percentile(per, 10))
    m["n_signers"] = int(len(per)); m["n_classes"] = int(len(cls))
    return m


def efficiency(config_path):
    import os, yaml, torch, importlib
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    def imp(dotted):
        mod, _, cls = dotted.rpartition(".")
        return getattr(importlib.import_module(mod), cls)

    cfg = yaml.safe_load(open(config_path))
    Model = imp(cfg["model"])
    margs = dict(cfg.get("model_args", {}))
    model = Model(**margs).eval()
    n_params = sum(p.numel() for p in model.parameters())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    V = margs.get("num_point", 27); M = margs.get("num_person", 1); C = margs.get("in_channels", 3)
    T = 120
    x = torch.randn(1, C, T, V, M, device=dev)
    out = {"params_M": n_params / 1e6, "device": dev}
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,), verbose=False)
        out["GFLOPs"] = flops / 1e9
    except Exception as e:
        out["GFLOPs"] = None; out["flops_note"] = f"thop unavailable ({type(e).__name__})"
    with torch.no_grad():
        for _ in range(3):
            model(x)  # warmup
        if dev == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(20):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        out["latency_ms_bs1"] = (time.perf_counter() - t0) / 20 * 1000
        if dev == "cuda":
            out["peak_VRAM_MB"] = torch.cuda.max_memory_allocated() / 1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--source", choices=["test", "loso", "val"], default="test")
    ap.add_argument("--label", default="data/bdsl_si/test_label.pkl")
    ap.add_argument("--config", help="for --efficiency")
    ap.add_argument("--efficiency", action="store_true")
    ap.add_argument("--json", help="write metrics json here")
    a = ap.parse_args()

    result = {}
    if a.base:
        clips, P, nf = load_probs(a.base, a.source)
        if clips is None:
            print(f"missing scores for {a.base} (source={a.source})"); return
        lab = load_labels(a.label)
        m = compute(clips, P, lab)
        result["metrics"] = m; result["base"] = a.base; result["source"] = a.source; result["score_files"] = nf
        print(f"### {a.base}  [source={a.source}, {nf} score files, n={m['n_clips']}] ###")
        print(f"  Recognition : Top-1 P-I {m['top1_PI']:.2f}  Top-5 P-I {m['top5_PI']:.2f}")
        print(f"  Class-bal   : Top-1 P-C {m['top1_PC']:.2f}  Top-5 P-C {m['top5_PC']:.2f}  Macro-F1 {m['macroF1']:.2f}")
        print(f"  Calibration : NLL {m['NLL']:.4f}  Brier {m['Brier']:.4f}  ECE {m['ECE']:.2f}%")
        print(f"  Retrieval   : MRR {m['MRR']:.4f}  R@1 {m['R@1']:.2f}  R@5 {m['R@5']:.2f}  R@10 {m['R@10']:.2f}")
        print(f"  Signer gen. : mean {m['signer_mean']:.2f}  SD {m['signer_SD']:.2f}  worst {m['signer_worst']:.2f}  "
              f"p10 {m['signer_p10']:.2f}  ({m['n_signers']} signers)")
    if a.efficiency:
        if not a.config:
            ap.error("--efficiency needs --config")
        eff = efficiency(a.config)
        result["efficiency"] = eff
        print(f"### efficiency [{a.config}] ###")
        for k, v in eff.items():
            print(f"  {k:16s} = {v:.3f}" if isinstance(v, float) else f"  {k:16s} = {v}")
    if a.json and result:
        json.dump(result, open(a.json, "w"), indent=2)
        print(f"[wrote {a.json}]")


if __name__ == "__main__":
    main()
