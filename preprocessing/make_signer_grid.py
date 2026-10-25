"""P2 Tier-1 #12 — controlled 2D signer-diversity grid (disentangle signers vs clips).

Builds a train bundle with EXACTLY --clips total examples drawn from the first
--signers canonical train signers, stratified across the 60 classes. Fixing total
clips while varying signer count lets us regress
  A = b0 + b1*N_signers + b2*N_clips + b3*N_signers*N_clips
and ask whether signer DIVERSITY helps beyond sheer sample count.

Usage: python preprocessing/make_signer_grid.py --signers 4 --clips 800 --out data/bdsl_grid_s4_n800
Val/test are the canonical SI val/test (copied by reference in the config, unchanged).
"""
import argparse, os, pickle, re
import numpy as np

SRC = "data/bdsl_si"
TRAIN_ORDER = [1, 4, 5, 6, 8, 9, 11, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signers", type=int, required=True)
    ap.add_argument("--clips", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    X = np.load(f"{SRC}/train_data.npy", mmap_mode="r")
    names, labels = pickle.load(open(f"{SRC}/train_label.pkl", "rb"))
    labels = np.asarray(labels)
    keep = set(TRAIN_ORDER[:a.signers])
    sig = np.array([int(re.search(r"U(\d+)W", n).group(1)) for n in names])
    pool = np.where(np.isin(sig, list(keep)))[0]
    if a.clips > len(pool):
        raise SystemExit(f"requested {a.clips} > available {len(pool)} for {a.signers} signers")

    # stratify across classes present in the pool
    classes = np.unique(labels[pool])
    per = max(1, a.clips // len(classes))
    chosen = []
    for c in classes:
        ci = pool[labels[pool] == c]
        rng.shuffle(ci)
        chosen.extend(ci[:per].tolist())
    # top up / trim to exactly --clips
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < a.clips:
        rest = [i for i in pool.tolist() if i not in set(chosen)]
        rng.shuffle(rest); chosen += rest[:a.clips - len(chosen)]
    chosen = np.sort(np.array(chosen[:a.clips]))

    out = np.lib.format.open_memmap(f"{a.out}/train_data.npy", mode="w+",
                                    dtype=np.float32, shape=(len(chosen),) + X.shape[1:])
    for j, i in enumerate(chosen):
        out[j] = X[i]
    out.flush(); del out
    pickle.dump(([names[i] for i in chosen], [int(labels[i]) for i in chosen]),
                open(f"{a.out}/train_label.pkl", "wb"))
    nsig = len({int(re.search(r'U(\d+)W', names[i]).group(1)) for i in chosen})
    print(f"DONE signers={a.signers}({nsig} present) clips={len(chosen)} -> {a.out}")


if __name__ == "__main__":
    main()
