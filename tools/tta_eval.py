#!/usr/bin/env python
"""Label-free test-time adaptation (TTA) to an unseen signer -- EVAL ONLY, no retraining.

This is the k=0 point of the P2 few-shot enrollment curve (sec:pers): every method
below uses ONLY the target signer's UNLABELED clips. The protocol is TRANSDUCTIVE:
each method sees all of one signer's unlabeled clips (statistics / one adaptation
epoch) before (or, for online Tent, while) predicting them. Signers are adapted
separately; a fresh copy of the checkpoint is used for every (run, signer, method).

Methods (--methods):
  none        the SI checkpoint as is (model.eval(), training BN running stats).
  adabn       AdaBN (Li et al. 2016): reset every BatchNorm's running mean/var and
              re-estimate them (cumulative average, momentum=None) from one no-grad
              pass over the target signer's clips; then eval as usual.
  cmvnxy      per-signer input standardization (CMVN analogue) on the model input
              (after the feeder's centre crop + node-0 centring): for every
              (channel, joint) z = (x - mu_signer) / sd_signer, then mapped back to the
              training distribution x' = z * sd_train + mu_train so the frozen model sees
              inputs on its training scale. Statistics are over valid (non-padded,
              detected) frame-joints only; padded / missing entries are left untouched.
              mu/sd_train come from the run's own training bundle (fold-specific for LOSO).
              Only the x/y coordinate channels are standardized: channel 2 is MediaPipe
              visibility (body) / a constant 1.0 (hands), not a coordinate, and
              standardizing it destroys accuracy (the first run, methods `cmvn*`, did so
              and is superseded).
  cmvnxy_c    the same with per-channel (x, y) statistics only (pooled over joints).
  cmvnxy_adabn  cmvnxy then adabn.
  tent        Tent (Wang et al., ICLR 2021): BatchNorm layers normalise with test-batch
              statistics (track_running_stats off, as in the reference code), only the
              BN affine (weight/bias) parameters are trainable, Adam(lr, betas 0.9/0.999),
              loss = mean prediction entropy; ONE epoch over the signer's clips in a seeded
              shuffled order. Reported accuracy is ONLINE (each batch is predicted by the
              forward that computes its loss, before that batch's update -- the reference
              Tent protocol); an OFFLINE second pass (frozen adapted params, same batch-stat
              BN, seeded shuffled batches) is logged as `acc_offline`. One method per
              lr in --tent-lrs ("tent_lr<lr>"); lr 0 = test-batch BN statistics only.
              Per-step entropy and batch accuracy are logged to watch for collapse.

Protocols (--protocol):
  A      canonical Protocol-A TEST (signers 2 and 13, adapted separately) with the paper's
         Protocol-A checkpoints (3 seeds).
  loso   the 11 held-out-signer LOSO folds (data/bdsl_si_loso/test_U<F>_val_U*/test_*),
         each fold's best-on-fold-val checkpoint (as docs/loso_heldout/<model>.csv), 3 seeds.

Models (--model): block_gcn (lr 0.01 recipe, the paper's current BlockGCN),
adaptive_gnn (lr 0.01, the paper's current Table 7 checkpoints), slgtformer.

Output: one JSON line per (run, signer, method) appended to --out, with the per-clip
correctness vector in bundle order (for paired bootstraps; tools/tta_report.py).
Already-present (model, protocol, fold, seed, signer, method) keys are skipped, so a
requeued job resumes where it stopped.

  python tools/tta_eval.py --model block_gcn --protocol A loso \
      --out docs/tta_grl_2026-10/raw/tta_block_gcn.jsonl
"""
import argparse, copy, glob, json, os, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from feeders import tools as ftools  # noqa: E402
for _s in ("multiarray", "numeric", "_multiarray_umath"):   # numpy-2 pickled checkpoints on numpy-1
    try:
        import numpy.core as _nc
        sys.modules.setdefault("numpy._core", _nc)
        sys.modules.setdefault("numpy._core." + _s, getattr(_nc, _s))
    except Exception:
        pass

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIG_RE = re.compile(r"U(\d+)W")
LOSO_FOLDS = ["01", "02", "04", "05", "06", "08", "09", "11", "12", "13", "15"]
SEEDS = [0, 1, 2]


def import_class(name):
    mod, cls = name.rsplit(".", 1)
    return getattr(__import__(mod, fromlist=[cls]), cls)


# ----------------------------------------------------------------------------- run registry
def _wd(exp):
    return os.path.join("work_dir", exp)


def _weights_from_test_cfg(exp):
    """Protocol-A checkpoint actually scored by the paper's `<exp>_test` main.py run."""
    c = yaml.safe_load(open(os.path.join(_wd(exp + "_test"), "config.yaml")))
    return c["weights"]


def complete(exp):
    """Training reached its final epoch (log.txt 'Eval epoch: <num_epoch>'). An audit (2026-10-22)
    found truncated LOSO / Protocol-A runs (e.g. SLGTFormer seed 2 stopped at epoch 25); their
    checkpoints are not the paper's intended models, so they are skipped and listed as skipped."""
    try:
        ne = int(yaml.safe_load(open(os.path.join(_wd(exp), "config.yaml")))["num_epoch"])
        eps = [int(x) for x in re.findall(r"Eval epoch: ?(\d+)", open(os.path.join(_wd(exp), "log.txt")).read())]
    except FileNotFoundError:
        return False
    return bool(eps) and max(eps) >= ne


def runs(model, protocol):
    """Yield dicts: fold, seed, cfg, ckpt, test (data,label), train (data,label)."""
    out = []
    if protocol == "A":
        for s in SEEDS:
            if model == "block_gcn":
                exp = f"bdsl_block_gcn_si_lr01_scratch_seed{s}"
                ck = os.path.join(_wd(exp), f"{exp}_model-best.pt")
            elif model == "adaptive_gnn":
                exp = f"bdsl_adaptive_gnn_si_seed{s}"; ck = _weights_from_test_cfg(exp)
            elif model == "slgtformer":
                exp = f"bdsl_slgtformer_si_seed{s}"; ck = _weights_from_test_cfg(exp)
            else:
                raise ValueError(model)
            out.append(dict(fold="A", seed=s, exp=exp, cfg=os.path.join(_wd(exp), "config.yaml"), ckpt=ck,
                            test=("data/bdsl_si/test_data.npy", "data/bdsl_si/test_label.pkl"),
                            train=("data/bdsl_si/train_data.npy", "data/bdsl_si/train_label.pkl")))
    elif protocol == "loso":
        stem = {"block_gcn": "bdsl_block_gcn_si_lr01_loso", "adaptive_gnn": "bdsl_adaptive_gnn_si_loso",
                "slgtformer": "bdsl_slgtformer_si_loso"}[model]
        for f in LOSO_FOLDS:
            fd = glob.glob(f"data/bdsl_si_loso/test_U{f}_val_U*")
            assert len(fd) == 1, (f, fd)
            fd = fd[0]
            for s in SEEDS:
                exp = f"{stem}_test{f}_seed{s}"
                out.append(dict(fold=f, seed=s, exp=exp, cfg=os.path.join(_wd(exp), "config.yaml"),
                                ckpt=os.path.join(_wd(exp), f"{exp}_model-best.pt"),
                                test=(f"{fd}/test_data.npy", f"{fd}/test_label.pkl"),
                                train=(f"{fd}/train_data.npy", f"{fd}/train_label.pkl")))
    else:
        raise ValueError(protocol)
    return out


# ----------------------------------------------------------------------------- data
_DATA = {}


def load_bundle(cfg, data, label):
    """Feeder test-mode outputs for a whole bundle: X (N,C,T,V,M) float32, valid mask
    (N,T,V,M) bool (raw entry non-zero after the centre crop), labels, names, signers."""
    key = (data, label)
    if key in _DATA:
        return _DATA[key]
    Feeder = import_class(cfg["feeder"])
    fa = dict(cfg["test_feeder_args"]); fa["data_path"] = data; fa["label_path"] = label
    fa.pop("debug", None)
    ds = Feeder(**fa)
    ws = int(fa.get("window_size", -1))
    X, M = [], []
    for i in range(len(ds)):
        x, _, _ = ds[i][:3]
        raw = np.array(ds.data[i])
        if ws > 0:
            raw = ftools.center_choose(raw, ws)
        X.append(np.asarray(x, dtype=np.float32))
        M.append(np.abs(raw).sum(0) > 0)
    X = np.stack(X); M = np.stack(M)
    y = np.asarray(ds.label, dtype=np.int64)
    names = [str(n) for n in ds.sample_name]
    sig = np.array([int(SIG_RE.search(n).group(1)) for n in names])
    _DATA[key] = (X, M, y, names, sig)
    print(f"[data] {data}: X{X.shape} valid {M.mean():.3f} signers {sorted(set(sig.tolist()))}", flush=True)
    return _DATA[key]


def moments(X, M, per_joint=True):
    """Masked mean/sd over (clips, frames[, persons]) -> shape (C,V) (per_joint) or (C,1)."""
    C, V = X.shape[1], X.shape[3]
    w = np.broadcast_to(M[:, None], X.shape).astype(np.float64)       # (N,C,T,V,M)
    ax = (0, 2, 4) if per_joint else (0, 2, 3, 4)
    n = w.sum(axis=ax)
    mu = (X * w).sum(axis=ax) / np.maximum(n, 1)
    if per_joint:
        d = X - mu[None, :, None, :, None]
    else:
        d = X - mu[None, :, None, None, None]
    var = (d * d * w).sum(axis=ax) / np.maximum(n - 1, 1)
    sd = np.sqrt(var)
    if not per_joint:
        mu, sd, n = mu[:, None], sd[:, None], n[:, None]
    return mu.astype(np.float32), sd.astype(np.float32), n


_TRAIN_STATS = {}


def train_moments(cfg, train, per_joint):
    key = (train, per_joint)
    if key not in _TRAIN_STATS:
        X, M, *_ = load_bundle(cfg, *train)
        _TRAIN_STATS[key] = moments(X, M, per_joint)
        _DATA.pop(tuple(train), None)  # training bundle no longer needed in memory
    return _TRAIN_STATS[key]


def cmvn(X, M, tr, per_joint):
    mu_s, sd_s, n_s = moments(X, M, per_joint)
    mu_t, sd_t, _ = tr
    ok = (n_s >= 10) & (sd_s > 1e-6)                                   # (C,V|1)
    ok[2:] = False   # channel 2 is MediaPipe visibility (body) / constant 1.0 (hands), not a coordinate
    scale = np.where(ok, sd_t / np.maximum(sd_s, 1e-6), 1.0).astype(np.float32)
    shift = np.where(ok, mu_t - mu_s * scale, 0.0).astype(np.float32)
    Y = X * scale[None, :, None, :, None] + shift[None, :, None, :, None]   # (C,V) or (C,1) broadcast
    return np.where(M[:, None], Y, X).astype(np.float32)


# ----------------------------------------------------------------------------- model
def build_model(cfg, ckpt):
    model = import_class(cfg["model"])(**cfg["model_args"])
    ck = torch.load(ckpt, map_location="cpu")
    w = ck["weights"] if isinstance(ck, dict) and "weights" in ck else ck
    w = {k.replace("module.", ""): v for k, v in w.items()}
    missing, unexpected = model.load_state_dict(w, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt}: missing {missing[:5]} unexpected {unexpected[:5]}")
    return model.to(DEV).eval()


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def bns(model):
    return [m for m in model.modules() if isinstance(m, BN_TYPES)]


def _out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


@torch.no_grad()
def forward_all(model, X, bs, order=None):
    idx = np.arange(len(X)) if order is None else order
    out = np.zeros((len(X), 0), dtype=np.float32)
    chunks = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        chunks.append((b, _out(model(torch.from_numpy(X[b]).to(DEV))).float().cpu().numpy()))
    out = np.zeros((len(X), chunks[0][1].shape[1]), dtype=np.float32)
    for b, o in chunks:
        out[b] = o
    return out


@torch.no_grad()
def adabn(model, X, bs):
    for m in bns(model):
        m.reset_running_stats(); m.momentum = None; m.train()
    for i in range(0, len(X), bs):
        model(torch.from_numpy(X[i:i + bs]).to(DEV))
    model.eval()
    return model


def softmax_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def tent(model, X, y, lr, bs, seed):
    """Reference-Tent configuration (Wang et al. 2021, github.com/DequanWang/tent):
    model.train() is replaced by eval() + BN.train() so dropout/DropGraph stay off."""
    model.eval()
    model.requires_grad_(False)
    params = []
    for m in bns(model):
        m.train(); m.track_running_stats = False; m.running_mean = None; m.running_var = None
        if m.affine:
            m.weight.requires_grad_(True); m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0) if lr > 0 else None
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(X))
    logits = np.zeros((len(X), 0), dtype=np.float32); chunks = []
    traj = []
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        if len(b) < 2:  # a 1-clip batch cannot give batch statistics for BatchNorm1d
            b = order[max(0, i - 1):i + bs]
        xb = torch.from_numpy(X[b]).to(DEV)
        with torch.set_grad_enabled(opt is not None):
            o = _out(model(xb))
            ent = softmax_entropy(o)
            if opt is not None:
                loss = ent.mean(); opt.zero_grad(); loss.backward(); opt.step()
        o = o.detach().float().cpu().numpy()
        chunks.append((b, o))
        traj.append(dict(step=len(traj), entropy=float(ent.mean().item()),
                         batch_acc=float((o.argmax(1) == y[b]).mean()),
                         n_pred_classes=int(len(np.unique(o.argmax(1))))))
    logits = np.zeros((len(X), chunks[0][1].shape[1]), dtype=np.float32)
    for b, o in chunks:
        logits[b] = o
    model.requires_grad_(False)
    order2 = np.random.RandomState(seed + 1000).permutation(len(X))
    off = forward_all(model, X, bs, order=order2)   # BN still in batch-stat mode
    return logits, off, traj


# ----------------------------------------------------------------------------- main
def summarize(logits, y):
    pred = logits.argmax(1)
    lt = torch.from_numpy(logits)
    ent = softmax_entropy(lt).numpy()
    counts = np.bincount(pred, minlength=logits.shape[1])
    return dict(acc=float((pred == y).mean() * 100), mean_entropy=float(ent.mean()),
                n_pred_classes=int((counts > 0).sum()), max_class_share=float(counts.max() / len(pred)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["block_gcn", "adaptive_gnn", "slgtformer"])
    ap.add_argument("--protocol", nargs="+", default=["A", "loso"], choices=["A", "loso"])
    ap.add_argument("--methods", nargs="+", default=["none", "adabn", "cmvnxy", "cmvnxy_c", "cmvnxy_adabn", "tent"])
    ap.add_argument("--tent-lrs", nargs="+", type=float, default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--bs", type=int, default=32, help="adaptation batch size (AdaBN pass, Tent)")
    ap.add_argument("--folds", nargs="+", default=None, help="restrict LOSO folds")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                r = json.loads(ln)
                done.add((r["model"], r["protocol"], r["fold"], r["seed"], r["signer"], r["method"]))
            except Exception:
                pass
    methods = []
    for m in a.methods:
        methods += [f"tent_lr{lr:g}" for lr in a.tent_lrs] if m == "tent" else [m]

    t0 = time.time()
    for prot in a.protocol:
        for r in runs(a.model, prot):
            if a.folds and r["fold"] != "A" and r["fold"] not in a.folds:
                continue
            if a.seeds is not None and r["seed"] not in a.seeds:
                continue
            if (a.model, prot, r["fold"], r["seed"], -1, "skipped_truncated") in done:
                continue   # keep the snapshot: a run skipped as truncated earlier stays excluded
            if not complete(r["exp"]):
                print(f"[skip] {r['exp']}: truncated run (final epoch not reached) -- excluded", flush=True)
                key = (a.model, prot, r["fold"], r["seed"], -1, "skipped_truncated")
                if key not in done:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(dict(model=a.model, protocol=prot, fold=r["fold"], seed=r["seed"],
                                                signer=-1, method="skipped_truncated", exp=r["exp"])) + "\n")
                    done.add(key)
                continue
            cfg = yaml.safe_load(open(r["cfg"]))
            X_all, M_all, y_all, names, sig_all = load_bundle(cfg, *r["test"])
            for s in sorted(set(sig_all.tolist())):
                todo = [m for m in methods if (a.model, prot, r["fold"], r["seed"], s, m) not in done]
                if not todo:
                    continue
                sel = np.where(sig_all == s)[0]
                X, M, y = X_all[sel], M_all[sel], y_all[sel]
                base = build_model(cfg, r["ckpt"])
                for meth in todo:
                    t1 = time.time()
                    model = copy.deepcopy(base)
                    rec = dict(model=a.model, protocol=prot, fold=r["fold"], seed=r["seed"], signer=s,
                               method=meth, n=int(len(sel)), ckpt=r["ckpt"], bs=a.bs,
                               ckpt_mtime=os.path.getmtime(r["ckpt"]))
                    if meth == "none":
                        lg = forward_all(model, X, a.bs)
                    elif meth == "adabn":
                        lg = forward_all(adabn(model, X, a.bs), X, a.bs)
                    elif meth in ("cmvnxy", "cmvnxy_c", "cmvnxy_adabn"):
                        pj = meth != "cmvnxy_c"
                        Xc = cmvn(X, M, train_moments(cfg, r["train"], pj), pj)
                        if meth == "cmvnxy_adabn":
                            model = adabn(model, Xc, a.bs)
                        lg = forward_all(model, Xc, a.bs)
                    elif meth.startswith("tent_lr"):
                        lr = float(meth[len("tent_lr"):])
                        lg, off, traj = tent(model, X, y, lr, a.bs, seed=1234 + r["seed"])
                        so = summarize(off, y)
                        rec.update(acc_offline=so["acc"], offline_n_pred_classes=so["n_pred_classes"],
                                   offline_max_class_share=so["max_class_share"],
                                   offline_correct=(off.argmax(1) == y).astype(int).tolist(), traj=traj)
                    else:
                        raise ValueError(meth)
                    rec.update(summarize(lg, y))
                    rec["correct"] = (lg.argmax(1) == y).astype(int).tolist()
                    rec["sec"] = round(time.time() - t1, 1)
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done.add((a.model, prot, r["fold"], r["seed"], s, meth))
                    extra = f" offline {rec['acc_offline']:.2f}" if "acc_offline" in rec else ""
                    print(f"{a.model} {prot} f{r['fold']} s{r['seed']} U{s:02d} {meth:14s} "
                          f"acc {rec['acc']:.2f}{extra} H {rec['mean_entropy']:.3f} "
                          f"cls {rec['n_pred_classes']} ({rec['sec']}s)", flush=True)
                    del model
                del base
                if DEV == "cuda":
                    torch.cuda.empty_cache()
    print(f"[done] {a.model} {a.protocol} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
