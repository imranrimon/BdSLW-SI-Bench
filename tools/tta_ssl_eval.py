#!/usr/bin/env python
"""(iv) Label-free masked-pose SSL test-time adaptation to an unseen signer (BlockGCN only).

A short SHuBERT-style masked-prediction pass (model/shubert_pretrain.py objective:
learned mask token on masked time steps, linear head predicting a discrete k-means
pose code per masked step, CE over masked positions; targets are `pose_motion`
features as in preprocessing/compute_pretrain_targets.py) run on ONE target signer's
unlabeled clips, then the frozen classifier (`fc`) is applied. Transductive: uses all of
that signer's unlabeled clips; no labels anywhere.

Adaptation to the classification checkpoint: the SI BlockGCN strides time by 2 at
each stage transition (T 120 -> 15), so masking and targets live at OUTPUT resolution:
each of the 15 output steps covers an 8-frame input window; a masked step replaces its
8 input frames with the mask token, and its target is the k-means code (K=64, fitted
on this signer's window-mean pose_motion features, MiniBatchKMeans seed 0) of that
window. Windows with no valid (non-padded) frame are excluded from the loss.

Schedule (pre-declared, not tuned on labels): mask ratio 0.3; head warm-up 1 epoch
(backbone frozen); then E epochs (default 3) updating every backbone parameter + head +
mask token with Adam (backbone lr --lr, head 10x), batch 32, BN kept in eval mode
(source statistics) so the SSL effect is not confounded with BN re-estimation.
Methods written: `ssl_lr<lr>` (eval with source BN stats) and `ssl_lr<lr>_adabn` (AdaBN
re-estimation after SSL; compare with plain `adabn` from tools/tta_eval.py).

  python tools/tta_ssl_eval.py --protocol A loso --lrs 1e-4 1e-3 \
      --out docs/tta_grl_2026-10/raw/tta_block_gcn_ssl.jsonl
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import yaml
from sklearn.cluster import MiniBatchKMeans

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from tools.tta_eval import (DEV, runs, complete, load_bundle, build_model, forward_all, adabn,  # noqa: E402
                            summarize)

K, MASK_RATIO = 64, 0.3


def window_codes(X, M, Tout, seed):
    """(N,C,T,V,1) -> codes (N,Tout) and valid (N,Tout) from window-mean pose_motion features."""
    N, C, T, V, _ = X.shape
    w = T // Tout
    pose = X[..., 0].transpose(0, 2, 1, 3).reshape(N, T, C * V)
    motion = np.zeros_like(pose); motion[:, 1:] = pose[:, 1:] - pose[:, :-1]
    fr_valid = M[..., 0].any(-1)                                     # (N,T)
    feat = np.concatenate([pose, motion], -1) * fr_valid[..., None]
    fw = feat.reshape(N, Tout, w, -1).sum(2) / np.maximum(fr_valid.reshape(N, Tout, w).sum(2), 1)[..., None]
    valid = fr_valid.reshape(N, Tout, w).any(-1)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=1024, n_init=3)
    km.fit(fw[valid])
    codes = np.zeros((N, Tout), dtype=np.int64)
    codes[valid] = km.predict(fw[valid])
    return codes, valid


def frame_feats(model, x):
    """BlockGCN per-output-step features (N, T', C)."""
    N, C, T, V, Mp = x.shape
    h = x.permute(0, 4, 3, 1, 2).contiguous().view(N, Mp * V * C, T)
    h = model.data_bn(h)
    h = h.view(N, Mp, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * Mp, C, T, V)
    for layer in model.layers:
        h = layer(h)
    c, t = h.size(1), h.size(2)
    return h.view(N, Mp, c, t, V).mean(-1).mean(1).transpose(1, 2)    # (N, T', c)


def ssl_adapt(model, X, codes, valid, lr, epochs, bs, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    N, C, T, V, _ = X.shape
    Tout = codes.shape[1]; w = T // Tout
    head = nn.Linear(model.feat_dim, K).to(DEV)
    tok = nn.Parameter(torch.zeros(1, C, 1, V, 1, device=DEV)); nn.init.trunc_normal_(tok, std=0.02)
    model.eval()                                       # BN uses source running stats throughout
    bb = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    for p in model.fc.parameters():
        p.requires_grad_(False)
    traj = []

    def step(opt, train_bb):
        for p in bb:
            p.requires_grad_(train_bb)
        order = rng.permutation(N)
        for i in range(0, N, bs):
            b = order[i:i + bs]
            xb = torch.from_numpy(X[b]).to(DEV)
            vm = torch.from_numpy(valid[b]).to(DEV)
            m = torch.rand(len(b), Tout, device=DEV) < MASK_RATIO
            m = m & vm
            if m.sum() == 0:
                continue
            fm = m.repeat_interleave(w, dim=1)[:, None, :, None, None]       # (B,1,T,1,1)
            xm = torch.where(fm, tok.expand_as(xb), xb)
            logits = head(frame_feats(model, xm))                             # (B,T',K)
            tgt = torch.from_numpy(codes[b]).to(DEV)
            loss = Fnn.cross_entropy(logits[m], tgt[m])
            opt.zero_grad(); loss.backward(); opt.step()
            traj.append(dict(phase="joint" if train_bb else "head", loss=float(loss.item()),
                             mask_acc=float((logits[m].argmax(-1) == tgt[m]).float().mean().item())))

    step(torch.optim.Adam(list(head.parameters()) + [tok], lr=lr * 10), False)
    opt = torch.optim.Adam([{"params": bb, "lr": lr}, {"params": list(head.parameters()) + [tok], "lr": lr * 10}])
    for _ in range(epochs):
        step(opt, True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, traj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", nargs="+", default=["A", "loso"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 1e-3])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_name = "block_gcn"
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            r = json.loads(ln); done.add((r["protocol"], r["fold"], r["seed"], r["signer"], r["method"]))
    for prot in a.protocol:
        for r in runs(model_name, prot):
            if a.folds and r["fold"] != "A" and r["fold"] not in a.folds:
                continue
            if not complete(r["exp"]):
                print(f"[skip] truncated {r['exp']}"); continue
            cfg = yaml.safe_load(open(r["cfg"]))
            X_all, M_all, y_all, names, sig_all = load_bundle(cfg, *r["test"])
            for s in sorted(set(sig_all.tolist())):
                sel = np.where(sig_all == s)[0]
                X, M, y = X_all[sel], M_all[sel], y_all[sel]
                todo = [lr for lr in a.lrs if any((prot, r["fold"], r["seed"], s, f"ssl_lr{lr:g}{suf}") not in done
                                                  for suf in ("", "_adabn"))]
                if not todo:
                    continue
                base = build_model(cfg, r["ckpt"])
                with torch.no_grad():
                    Tout = frame_feats(base, torch.from_numpy(X[:2]).to(DEV)).shape[1]
                codes, valid = window_codes(X, M, Tout, seed=0)
                for lr in todo:
                    t1 = time.time()
                    model, traj = ssl_adapt(copy.deepcopy(base), X, codes, valid, lr, a.epochs, a.bs,
                                            seed=1234 + r["seed"])
                    for suf in ("", "_adabn"):
                        meth = f"ssl_lr{lr:g}{suf}"
                        m2 = adabn(copy.deepcopy(model), X, a.bs) if suf else model
                        lg = forward_all(m2, X, a.bs)
                        rec = dict(model=model_name, protocol=prot, fold=r["fold"], seed=r["seed"], signer=s,
                                   method=meth, n=int(len(sel)), ckpt=r["ckpt"], bs=a.bs, epochs=a.epochs,
                                   ckpt_mtime=os.path.getmtime(r["ckpt"]))
                        rec.update(summarize(lg, y))
                        rec["correct"] = (lg.argmax(1) == y).astype(int).tolist()
                        rec["ssl_traj"] = traj
                        rec["sec"] = round(time.time() - t1, 1)
                        with open(a.out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        done.add((prot, r["fold"], r["seed"], s, meth))
                        print(f"ssl {prot} f{r['fold']} s{r['seed']} U{s:02d} {meth:20s} acc {rec['acc']:.2f} "
                              f"ssl loss {traj[0]['loss']:.2f}->{traj[-1]['loss']:.2f} "
                              f"mask acc {traj[-1]['mask_acc']:.2f} ({rec['sec']}s)", flush=True)
                if DEV == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
