"""ASL Citizen dictionary-retrieval metrics — the paper's standard eval panel.

ASL Citizen (Desai et al., NeurIPS D&B 2023) evaluates isolated-sign recognition
as DICTIONARY RETRIEVAL: rank all glosses for a query video, report Recall@k
(R@1 == Top-1 accuracy), Mean Reciprocal Rank (MRR), and Discounted Cumulative
Gain (DCG). This tool computes that panel from the saved per-clip score dicts
(work_dir/<base>_seed*/eval_results/best_acc.pkl), ensembling seeds by averaging
softmax probabilities, mirroring tools/benchmark_metrics.py.

Participant ids are recovered from the <user>__<basename> sample-name prefix
(written by preprocessing/generate_aslcitizen_split_npy.py), giving a
per-participant generalization breakdown on the disjoint TEST participants.

Usage:
  python tools/aslcitizen_retrieval.py --base aslcitizen_si_test \
      --label data/aslcitizen_si/test_label.pkl
  python tools/aslcitizen_retrieval.py --base aslcitizen_mixed_test \
      --label data/aslcitizen_mixed/test_label.pkl --json out.json
"""
import argparse
import glob
import json
import pickle
import sys

try:
    import numpy.core as _npc
    sys.modules.setdefault("numpy._core", _npc)
    for _s in ("multiarray", "numeric", "umath", "_multiarray_umath", "numerictypes"):
        try:
            sys.modules.setdefault("numpy._core." + _s,
                                   __import__("numpy.core." + _s, fromlist=["x"]))
        except Exception:
            pass
except Exception:
    pass
import numpy as np


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _glob(base):
    # accept both "<base>_seed*_test" (train configs) and "<base>_seed*" (test configs)
    pats = [
        "work_dir/{}_seed*/eval_results/best_acc.pkl".format(base),
        "work_dir/{}_seed*_test/eval_results/best_acc.pkl".format(base),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    return sorted(set(files))


def load_probs(base):
    pkls = _glob(base)
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


def user_of(clip):
    # sample name is "<user>__<videobasename>"
    return clip.split("__", 1)[0] if "__" in clip else "?"


def compute(clips, P, lab):
    keep = [i for i, c in enumerate(clips) if c in lab]
    clips = [clips[i] for i in keep]
    P = P[keep]
    y = np.array([lab[c] for c in clips])
    users = np.array([user_of(c) for c in clips])
    order = np.argsort(-P, axis=1)
    top1 = order[:, 0]
    rank_true = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])  # 0-based
    m = {}
    m["n_clips"] = int(len(y))
    m["n_classes"] = int(len(np.unique(y)))
    m["n_participants"] = int(len(np.unique(users)))
    # retrieval (ASL Citizen standard)
    m["R@1"] = float((rank_true < 1).mean() * 100)     # == Top-1 accuracy
    m["R@5"] = float((rank_true < 5).mean() * 100)
    m["R@10"] = float((rank_true < 10).mean() * 100)
    m["R@20"] = float((rank_true < 20).mean() * 100)
    m["MRR"] = float((1.0 / (rank_true + 1)).mean())
    m["DCG"] = float((1.0 / np.log2(rank_true + 2)).mean())
    m["top1_PI"] = m["R@1"]
    m["top5_PI"] = m["R@5"]
    # per-class (macro) recall
    cls = np.unique(y)
    m["top1_PC"] = float(np.mean([(top1[y == c] == c).mean() for c in cls]) * 100)
    m["top5_PC"] = float(np.mean([(rank_true[y == c] < 5).mean() for c in cls]) * 100)
    # per-participant generalization (disjoint test participants)
    correct = (top1 == y).astype(np.float64)
    per = np.array([correct[users == u].mean() for u in np.unique(users)]) * 100
    m["participant_mean"] = float(per.mean())
    m["participant_SD"] = float(per.std())
    m["participant_worst"] = float(per.min())
    m["participant_p10"] = float(np.percentile(per, 10))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="work_dir experiment stem, e.g. aslcitizen_si_test")
    ap.add_argument("--label", required=True,
                    help="test_label.pkl matching the eval split")
    ap.add_argument("--json", help="write metrics json here")
    a = ap.parse_args()

    clips, P, nf = load_probs(a.base)
    if clips is None:
        print("missing scores for {} (looked in work_dir/{}_seed*/)".format(a.base, a.base))
        return
    lab = load_labels(a.label)
    m = compute(clips, P, lab)
    print("### {}  [{} score files, n={} clips, {} classes, {} participants] ###".format(
        a.base, nf, m["n_clips"], m["n_classes"], m["n_participants"]))
    print("  Retrieval    : R@1 {:.2f}  R@5 {:.2f}  R@10 {:.2f}  R@20 {:.2f}  "
          "MRR {:.4f}  DCG {:.4f}".format(m["R@1"], m["R@5"], m["R@10"], m["R@20"],
                                          m["MRR"], m["DCG"]))
    print("  Class-bal    : Top-1 P-C {:.2f}  Top-5 P-C {:.2f}".format(m["top1_PC"], m["top5_PC"]))
    print("  Participant  : mean {:.2f}  SD {:.2f}  worst {:.2f}  p10 {:.2f}".format(
        m["participant_mean"], m["participant_SD"], m["participant_worst"], m["participant_p10"]))
    if a.json:
        json.dump({"base": a.base, "score_files": nf, "metrics": m},
                  open(a.json, "w"), indent=2)
        print("[wrote {}]".format(a.json))


if __name__ == "__main__":
    main()
