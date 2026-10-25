"""P2 ★#1 — Build a SIGNER-MIXED random split of AUTSL from the existing
signer-disjoint (official SI) bundles, to quantify the signer-identity shortcut
(random-split acc  vs  SI acc) on a SECOND dataset, exactly as done for BdSLW60.

We DO NOT re-extract poses: we concatenate the official train/val/test bundles
and re-partition 80/10/10 with a fixed seed, IGNORING signer — so signers now
appear in all splits (the shortcut-inducing setup). Memmap-safe (never loads the
full 3.5 GB into RAM at once).

Output: data/autsl_random/{train,val,test}_{data.npy,label.pkl}
"""
import os, pickle
import numpy as np

SRC = "data/autsl_si"
DST = "data/autsl_random"
SEED = 42
FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def main():
    os.makedirs(DST, exist_ok=True)
    # gather (split, local_idx) index + labels + names across all official splits
    arrs, names, labels, offsets = {}, [], [], {}
    total = 0
    for sp in ("train", "val", "test"):
        a = np.load(f"{SRC}/{sp}_data.npy", mmap_mode="r")
        nm, lb = pickle.load(open(f"{SRC}/{sp}_label.pkl", "rb"))
        assert len(nm) == a.shape[0] == len(lb), (sp, a.shape, len(nm), len(lb))
        arrs[sp] = a
        offsets[sp] = total
        total += a.shape[0]
        names += list(nm); labels += list(lb)
    labels = np.asarray(labels)
    C, T, V, M = arrs["train"].shape[1:]
    print(f"combined {total} clips  shape (*,{C},{T},{V},{M})  classes={len(set(labels.tolist()))}")

    # global index -> (split, local_idx)
    order = []
    for sp in ("train", "val", "test"):
        for i in range(arrs[sp].shape[0]):
            order.append((sp, i))

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(total)
    n_tr = int(FRACS["train"] * total); n_va = int(FRACS["val"] * total)
    cut = {"train": perm[:n_tr], "val": perm[n_tr:n_tr+n_va], "test": perm[n_tr+n_va:]}

    for sp, idxs in cut.items():
        idxs = np.sort(idxs)   # sorted for cache-friendly reads
        out = np.lib.format.open_memmap(f"{DST}/{sp}_data.npy", mode="w+",
                                        dtype=np.float32, shape=(len(idxs), C, T, V, M))
        onm, olb = [], []
        for j, g in enumerate(idxs):
            src_sp, li = order[g]
            out[j] = arrs[src_sp][li]
            onm.append(names[g]); olb.append(int(labels[g]))
            if (j + 1) % 5000 == 0: print(f"  {sp}: {j+1}/{len(idxs)}")
        out.flush(); del out
        pickle.dump((onm, olb), open(f"{DST}/{sp}_label.pkl", "wb"))
        # signer-mix sanity: how many distinct signers land in this split
        import re
        sig = {int(re.search(r'signer(\d+)_', n).group(1)) for n in onm if re.search(r'signer(\d+)_', n)}
        print(f"  wrote {sp}: {len(idxs)} clips, {len(sig)} distinct signers")
    print("DONE ->", DST)


if __name__ == "__main__":
    main()
