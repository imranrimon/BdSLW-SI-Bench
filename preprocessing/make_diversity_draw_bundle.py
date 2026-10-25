"""E5 -- signer-diversity draw bundle (mixed-accuracy multi-subset SMG(k)).

For an explicit signer SUBSET S (|S| = k, drawn randomly from the 8 canonical
train signers), this builds a training bundle from S's clips while class-
stratified reserving a SEEN-signer test holdout:

  data/bdsl_si_div/k<k>_d<draw>/
      train_data.npy / train_label.pkl      (S's clips minus the holdout)
      seentest_data.npy / seentest_label.pkl (held-out clips of the SAME S signers)

Model selection uses the canonical unseen val signer U15 (data/bdsl_si/val) and
the UNSEEN test is the canonical held-out TEST {2,13} (data/bdsl_si/test) -- both
reused, not rebuilt. Testing the ONE trained model on seentest vs the canonical
unseen test yields the operational per-k signer-mixing gap
    SMG(k) = acc(seen holdout) - acc(unseen TEST),
identical model, tested two ways -- cf. the paired-instance design of E1.

Pure index filter on the existing SI train bundle (no MediaPipe re-run).

Usage:
    python preprocessing/make_diversity_draw_bundle.py \\
        --signers 1,4,8 --out data/bdsl_si_div/k3_d0 --holdout 0.2 --seed 0
"""
import argparse, os, pickle, re
import numpy as np

SRC = "data/bdsl_si"
RX = re.compile(r"U(\d+)W")


def build(signers, out, holdout, seed, src=SRC, regex=None):
    rng = np.random.default_rng(seed)
    rx = re.compile(regex) if regex else RX
    names, labels = pickle.load(open(f"{src}/train_label.pkl", "rb"))
    labels = np.asarray(labels)
    print(f"loading {src}/train_data.npy into RAM ...", flush=True)
    data = np.load(f"{src}/train_data.npy")
    sig = np.array([int(rx.search(n).group(1)) for n in names])

    keep = set(signers)
    pool = np.where(np.isin(sig, list(keep)))[0]
    if len(pool) == 0:
        raise SystemExit(f"no clips for signers {sorted(keep)}")

    # class-stratified holdout: reserve a fraction of EACH class's subset clips as
    # the seen-signer test set; the rest train.
    test_idx, train_idx = [], []
    for c in np.unique(labels[pool]):
        ci = pool[labels[pool] == c]
        rng.shuffle(ci)
        n_hold = max(1, int(round(holdout * len(ci)))) if len(ci) >= 2 else 0
        test_idx.extend(ci[:n_hold].tolist())
        train_idx.extend(ci[n_hold:].tolist())
    train_idx = np.sort(np.array(train_idx))
    test_idx = np.sort(np.array(test_idx))

    tmp = f"{out}.{os.getpid()}.tmp"
    os.makedirs(tmp, exist_ok=True)
    np.save(f"{tmp}/train_data.npy", data[train_idx])
    with open(f"{tmp}/train_label.pkl", "wb") as f:
        pickle.dump(([names[i] for i in train_idx], [int(labels[i]) for i in train_idx]), f)
    np.save(f"{tmp}/seentest_data.npy", data[test_idx])
    with open(f"{tmp}/seentest_label.pkl", "wb") as f:
        pickle.dump(([names[i] for i in test_idx], [int(labels[i]) for i in test_idx]), f)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.replace(tmp, out)  # atomic
    nsig = len({int(rx.search(names[i]).group(1)) for i in train_idx})
    print(f"DONE signers={sorted(keep)} ({nsig} present) | train={len(train_idx)} "
          f"seentest={len(test_idx)} clips -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signers", required=True,
                    help="comma-separated signer ids, e.g. 1,4,8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="fraction of each class's subset clips reserved as seen-test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=SRC,
                    help="source SI bundle dir (default data/bdsl_si)")
    ap.add_argument("--regex", default=None,
                    help=r"signer-id regex, group(1)=id (default 'U(\d+)W'; "
                         r"AUTSL: 'signer(\d+)_')")
    a = ap.parse_args()
    signers = [int(s) for s in a.signers.split(",") if s.strip()]
    build(signers, a.out, a.holdout, a.seed, src=a.src, regex=a.regex)
