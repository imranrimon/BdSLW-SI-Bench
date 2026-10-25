"""P2 Tier-1 — matched signer-MIXED random splits of BdSLW60 for Signer-Mixing-Gap CIs.

Concatenates the canonical SI bundles (train+val+test), reshuffles with a given seed
IGNORING signer, and re-partitions into the SAME sizes as the SI split
(5748 / 655 / 1365) so the only difference vs SI is signer mixing. Memmap-safe.

Usage: python preprocessing/make_bdsl_random_split.py --seed 42 --out data/bdsl_rand42
Run for seeds {42,43,44,45,46} to get 5 matched random splits.
"""
import argparse, os, pickle
import numpy as np

SRC = "data/bdsl_si"
SIZES = {"train": 5748, "val": 655, "test": 1365}   # match the canonical SI split exactly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    arrs, names, labels, order = {}, [], [], []
    total = 0
    for sp in ("train", "val", "test"):
        arr = np.load(f"{SRC}/{sp}_data.npy", mmap_mode="r")
        nm, lb = pickle.load(open(f"{SRC}/{sp}_label.pkl", "rb"))
        assert arr.shape[0] == len(nm) == len(lb)
        arrs[sp] = arr
        for i in range(arr.shape[0]):
            order.append((sp, i))
        names += list(nm); labels += list(lb); total += arr.shape[0]
    labels = np.asarray(labels)
    C, T, V, M = arrs["train"].shape[1:]
    assert total == sum(SIZES.values()), (total, SIZES)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(total)
    cuts = {"train": perm[:SIZES["train"]],
            "val":   perm[SIZES["train"]:SIZES["train"]+SIZES["val"]],
            "test":  perm[SIZES["train"]+SIZES["val"]:]}
    import re
    for sp, idxs in cuts.items():
        idxs = np.sort(idxs)
        out = np.lib.format.open_memmap(f"{a.out}/{sp}_data.npy", mode="w+",
                                        dtype=np.float32, shape=(len(idxs), C, T, V, M))
        onm, olb = [], []
        for j, g in enumerate(idxs):
            src_sp, li = order[g]
            out[j] = arrs[src_sp][li]
            onm.append(names[g]); olb.append(int(labels[g]))
        out.flush(); del out
        pickle.dump((onm, olb), open(f"{a.out}/{sp}_label.pkl", "wb"))
        sig = {int(re.search(r'U(\d+)W', n).group(1)) for n in onm if re.search(r'U(\d+)W', n)}
        print(f"  seed{a.seed} {sp}: {len(idxs)} clips, {len(sig)} signers")
    print(f"DONE seed{a.seed} -> {a.out}")


if __name__ == "__main__":
    main()
