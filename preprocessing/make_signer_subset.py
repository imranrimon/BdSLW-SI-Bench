"""Build a signer-subset training bundle for the SI signer-scaling curve (P2).

Subsets the canonical SI TRAIN bundle (data/bdsl_si/train_{data.npy,label.pkl}) to the
first k of the fixed train-signer order [1,4,5,6,8,9,11,12] (nested: k=2 ⊂ k=4 ⊂ ...),
writing data/bdsl_si_k{k}/train_{data.npy,label.pkl}. Val/test are unchanged (the config
points them at the canonical data/bdsl_si), so only the number of TRAINING signers varies.
No MediaPipe re-run — pure index filter on the existing bundle.

Usage:  python preprocessing/make_signer_subset.py --k 2 4 6
"""
import argparse, os, pickle, re
import numpy as np

TRAIN_ORDER = [1, 4, 5, 6, 8, 9, 11, 12]   # canonical SI train signers, fixed nesting
RX = re.compile(r"U(\d+)W(\d+)F")

def build(ks, src="data/bdsl_si"):
    # load the full train bundle into RAM ONCE (fast; avoids slow repeated scratch mmap indexing)
    names, labels = pickle.load(open(f"{src}/train_label.pkl", "rb"))
    print(f"loading {src}/train_data.npy into RAM ...", flush=True)
    data = np.load(f"{src}/train_data.npy")          # full read, in memory
    labels = np.asarray(labels)
    signer_of = np.array([int(RX.search(n).group(1)) for n in names])
    for k in ks:
        keep = set(TRAIN_ORDER[:k])
        idx = np.where(np.isin(signer_of, list(keep)))[0]
        out = f"data/bdsl_si_k{k}.{os.getpid()}.tmp"   # unique tmp -> no cross-process collision
        os.makedirs(out, exist_ok=True)
        np.save(f"{out}/train_data.npy", data[idx])
        with open(f"{out}/train_label.pkl", "wb") as f:
            pickle.dump(([names[i] for i in idx], [int(labels[i]) for i in idx]), f)
        os.replace(out, f"data/bdsl_si_k{k}")        # atomic dir swap (final is always complete)
        print(f"k={k} signers={sorted(keep)}: {len(idx)}/{len(names)} clips -> data/bdsl_si_k{k}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=[2, 4, 6])
    a = ap.parse_args()
    build(a.k)
