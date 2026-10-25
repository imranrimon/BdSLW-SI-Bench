"""Few-shot signer-personalization support/query bundle.

Splits a single held-out TEST signer F's clips into a k-shot ADAPTATION
(support) set and an EVAL (query) set, both drawn from the canonical SI test
bundle (data/bdsl_si/test_*). For EACH class present for F we reserve K support
clips (class-stratified, seeded) as the adaptation set; every remaining F clip
becomes the query/eval set. The support set is what an operator would enroll
after SI training; query accuracy measures how much of the SI gap that light
enrollment closes.

  support_data.npy / support_label.pkl   K clips-per-class of signer F
  query_data.npy   / query_label.pkl     all of F's remaining clips

K=0 -> empty support (nothing written for support beyond an empty bundle), and
ALL of F's clips become the query set (the pure SI-transfer baseline).

Pure index filter on the existing SI test npy, loaded into RAM once (cf.
make_diversity_draw_bundle.py); no MediaPipe re-run. Labels stay in the shared
60-class index space, so the SI checkpoint's classifier is reusable as-is.

Usage:
    python preprocessing/make_kshot_signer_bundle.py \\
        --signer 2 --k 3 --seed 0 --out data/bdsl_si_pers/U02_k3_seed0
"""
import argparse, os, pickle, re
import numpy as np

SRC = "data/bdsl_si"
RX = re.compile(r"U(\d+)W")


def build(signer, k, out, seed):
    rng = np.random.default_rng(seed)
    names, labels = pickle.load(open(f"{SRC}/test_label.pkl", "rb"))
    labels = np.asarray(labels)
    print(f"loading {SRC}/test_data.npy into RAM ...", flush=True)
    data = np.load(f"{SRC}/test_data.npy")
    sig = np.array([int(RX.search(n).group(1)) for n in names])

    pool = np.where(sig == signer)[0]
    if len(pool) == 0:
        raise SystemExit(f"no clips for signer U{signer:02d} in {SRC}/test")

    # class-stratified support draw: reserve up to K clips of EACH class present
    # for F as the adaptation set; the rest of F's clips are the query/eval set.
    support_idx, query_idx = [], []
    for c in np.unique(labels[pool]):
        ci = pool[labels[pool] == c]
        rng.shuffle(ci)
        n_sup = min(k, len(ci))
        support_idx.extend(ci[:n_sup].tolist())
        query_idx.extend(ci[n_sup:].tolist())
    support_idx = np.sort(np.array(support_idx, dtype=int))
    query_idx = np.sort(np.array(query_idx, dtype=int))

    tmp = f"{out}.{os.getpid()}.tmp"
    os.makedirs(tmp, exist_ok=True)
    np.save(f"{tmp}/support_data.npy", data[support_idx])
    with open(f"{tmp}/support_label.pkl", "wb") as f:
        pickle.dump(([names[i] for i in support_idx],
                     [int(labels[i]) for i in support_idx]), f)
    np.save(f"{tmp}/query_data.npy", data[query_idx])
    with open(f"{tmp}/query_label.pkl", "wb") as f:
        pickle.dump(([names[i] for i in query_idx],
                     [int(labels[i]) for i in query_idx]), f)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.isdir(out):
        import shutil
        shutil.rmtree(out)
    os.replace(tmp, out)  # atomic

    n_cls_F = len(np.unique(labels[pool]))
    n_cls_sup = len(set(int(labels[i]) for i in support_idx))
    print(f"DONE signer=U{signer:02d} k={k} seed={seed} | "
          f"F_clips={len(pool)} classes_present={n_cls_F} | "
          f"support={len(support_idx)} ({n_cls_sup} classes) "
          f"query={len(query_idx)} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", type=int, required=True,
                    help="held-out TEST signer id (canonical SI test = 2 or 13)")
    ap.add_argument("--k", type=int, required=True,
                    help="labeled support clips per class (0 -> empty support)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the class-stratified support draw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.signer, a.k, a.out, a.seed)
