#!/usr/bin/env python
"""Signer-decodability probe v2 (P2 leak-table fix, 2026-10).

What tools/decodability_probe.py (v1) did, and what v2 changes:
  v1 input   : raw 300-frame, uncentred, zero-padded arrays (not what any model sees);
  v2 input   : FEATURE-PROCESSED clips exactly as the test feeder gives them (feeders.feeder.Feeder
               with the encoder's test_feeder_args: centre crop/pad to 120 frames, root-x/y
               normalisation), i.e. the model's real test-time input.
  v1 clips   : the first 4000/6000 clips of the BdSLW60-SI TRAIN split = the supervised encoders'
               own training clips;
  v2 clips   : HELD-OUT clips no encoder trained on: SI test (U02, U13) + the labelled pretrain
               pool (U03, U07, U10, U14, U16, U17, U18) -> 9 signers, 2904 clips (--with-val adds U15,
               which only selected checkpoints).
  v1 CV      : StratifiedKFold over clips, so sibling trials of one recording (each signer x class is
               one video of ~12 trials) straddle train/test folds;
  v2 CV      : StratifiedGroupKFold grouped by source-recording stem (name before '_trial'), so a
               recording's trials never straddle folds. Scaler is fit inside each fold.

Metrics (L_norm = (BA - 1/S) / (1 - 1/S)):
  cc   class-conditioned: subtract the per-class mean feature, then one global signer probe (P2's L_norm)
  mrg  marginal: no conditioning
  wc   within-class: a signer probe inside each class, averaged (P2's L_norm^wc). In BdSLW60 a signer
       contributes ONE recording per class, so a recording-grouped within-class split cannot exist:
       wc is necessarily ungrouped (it measures same-recording trial similarity) and is flagged.
  cc_ungrouped  cc with v1's ungrouped CV on the same held-out features (isolates the grouping effect).
Each probe is averaged over CV seeds 0/1/2.

Reference rows (no learned encoder):
  rawmean_unmasked  per-joint mean of the feeder-processed (x, y, conf) over the 120 frames (81-d)
  rawmean_masked    the same, averaged only over frames where the joint is detected (conf > 0)
  missrate_len      per-joint missing rate over valid frames + valid clip length (28-d)
  randinit_*        randomly initialised (torch seed 0) BlockGCN / Adaptive GNN, eval mode
--protocol v1 reruns the reference rows (and any encoder) under v1's input/clips/CV for comparison.

  python tools/decodability_probe_v2.py --out docs/p2_missing_joint_fixes_2026-10/raw/probe_v2.jsonl
"""
import argparse, json, os, pickle, re, sys, time
import numpy as np
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.decodability_probe import build_backbone, _imp  # noqa: E402

W = "work_dir"
SSL_CFG = "config/bdsl_shubert_pretrain_xlingual.yaml"   # backbone args used by v1 for every SSL encoder
PB = "config/probe_block_gcn_si.yaml"
PA = "config/probe_adaptive_gnn_si.yaml"
# (tag, probe/backbone config, checkpoint, flag) -- the P2 leak table + the GRL arms + seed spread
ENCODERS = (
    [(f"sup_blockgcn_lr01_s{s}", PB, f"{W}/bdsl_block_gcn_si_lr01_scratch_seed{s}/bdsl_block_gcn_si_lr01_scratch_seed{s}_model-best.pt", "")
     for s in (0, 1, 2)]
    + [("sup_adaptive_lr0p01_s0_m75", PA, f"{W}/bdsl_adaptive_gnn_si_seed0/bdsl_adaptive_gnn_si_seed0_model-75.pt", "P2 row"),
       ("sup_adaptive_lr0p01_s0_m59", PA, f"{W}/bdsl_adaptive_gnn_si_seed0/bdsl_adaptive_gnn_si_seed0_model-59.pt", ""),
       ("sup_adaptive_lr0p01_s1_m59", PA, f"{W}/bdsl_adaptive_gnn_si_seed1/bdsl_adaptive_gnn_si_seed1_model-59.pt", ""),
       ("sup_adaptive_lr0p01_s2_m89", PA, f"{W}/bdsl_adaptive_gnn_si_seed2/bdsl_adaptive_gnn_si_seed2_model-89.pt", "")]
    + [(f"sup_adaptive_lr0p1_s{s}", PA, f"{W}/bdsl_adaptive_gnn_si_lr0p1_seed{s}/bdsl_adaptive_gnn_si_lr0p1_seed{s}_model-best.pt", "")
       for s in (0, 1, 2)]
    + [("ssl_bdsl_only", SSL_CFG, f"{W}/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt", ""),
       ("ssl_xlingual", SSL_CFG, f"{W}/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt", "FLAG: WLASL-scrambled pool"),
       ("ssl_dgxl", SSL_CFG, f"{W}/bdsl_shubert_pretrain_dgxl/pretrained_epoch30.pt", "FLAG: WLASL-scrambled pool")]
    + [(f"grl_lr01_l{l}", PB, f"{W}/bdsl_block_gcn_si_signeradv_lr01_gl{l}_seed0/bdsl_block_gcn_si_signeradv_lr01_gl{l}_seed0_model-best.pt", "")
       for l in ("01", "05", "10")]
    + [("randinit_blockgcn", PB, None, "reference"), ("randinit_adaptive", PA, None, "reference")]
)
TEST_ARGS = dict(window_size=120, random_choose=False, normalization=True)   # every encoder's test_feeder_args


def signer_of(n): return int(re.match(r"U(\d+)", n).group(1))
def rec_of(n): return n.split("_trial")[0]


def load_heldout(with_val):
    """Feeder-processed held-out clips: (X (N,3,120,27,1), names, cls, raw valid lengths, raw missing rates)."""
    from feeders.feeder import Feeder
    from feeders import tools as ft
    Xs, names, cls, lens, miss = [], [], [], [], []
    for sp in (["test", "pretrain"] + (["val"] if with_val else [])):
        fd = Feeder(f"data/bdsl_si/{sp}_data.npy", f"data/bdsl_si/{sp}_label.pkl", **TEST_ARGS)
        raw = np.load(f"data/bdsl_si/{sp}_data.npy", mmap_mode="r")
        for i in range(len(fd)):
            x, y, _ = fd[i]
            Xs.append(np.asarray(x, dtype=np.float32)); cls.append(int(y)); names.append(fd.sample_name[i])
            r = np.asarray(raw[i]); L = ft.valid_frame_length(r); lens.append(L)
            m = (r[0, :L] == 0) & (r[1, :L] == 0)                       # (L, V, M)
            miss.append(m.mean(axis=(0, 2)))
    return np.stack(Xs), names, np.array(cls), np.array(lens), np.stack(miss)


def load_v1(n=4000):
    """v1 protocol input: raw arrays of the first n SI-train clips."""
    raw = np.load("data/bdsl_si/train_data.npy", mmap_mode="r")
    names, cls = pickle.load(open("data/bdsl_si/train_label.pkl", "rb"))
    X = np.asarray(raw[:n], dtype=np.float32)
    from feeders import tools as ft
    lens = np.array([ft.valid_frame_length(x) for x in X])
    miss = np.stack([((x[0, :L] == 0) & (x[1, :L] == 0)).mean(axis=(0, 2)) for x, L in zip(X, lens)])
    return X, list(names[:n]), np.asarray(cls[:n]), lens, miss


def ref_features(X, lens, miss):
    conf = X[:, 2]                                          # (N, T, V, M): 0 where missing or padding
    det = (conf > 0).astype(np.float32)
    unm = X.mean(axis=2).reshape(len(X), -1)               # mean over T of (x,y,c) per joint
    num = (X * det[:, None]).sum(axis=2)
    msk = (num / np.maximum(det.sum(axis=1), 1)[:, None]).reshape(len(X), -1)
    return {"rawmean_unmasked": unm, "rawmean_masked": msk,
            "missrate_len": np.concatenate([miss, lens[:, None].astype(np.float32)], 1)}


@torch.no_grad()
def net_features(cfg, ckpt, X, dev, bs=64):
    if ckpt is None:
        torch.manual_seed(0)
        c = yaml.safe_load(open(os.path.join(ROOT, cfg)))
        kw = dict(c["backbone_args"]); kw["return_features"] = True
        bb = _imp(c["backbone"])(**kw).to(dev).eval()
    else:
        bb, _ = build_backbone(cfg, ckpt, dev)
    out = []
    for i in range(0, len(X), bs):
        f = bb(torch.from_numpy(X[i:i + bs]).to(dev))
        if f.dim() == 3: f = f.mean(dim=1)                  # Adaptive GNN: per-frame (N, T', C)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def _lr():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def lnorm(ba, S): return (ba - 1.0 / S) / (1.0 - 1.0 / S)


def probe(F, y, groups, seeds=(0, 1, 2), grouped=True, scale_all=False):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    if scale_all: F = StandardScaler().fit_transform(F)     # v1 behaviour
    S = len(np.unique(y)); res = []
    for sd in seeds:
        cv = (StratifiedGroupKFold(5, shuffle=True, random_state=sd) if grouped
              else StratifiedKFold(5, shuffle=True, random_state=sd))
        accs = []
        for tr, te in (cv.split(F, y, groups) if grouped else cv.split(F, y)):
            accs.append(balanced_accuracy_score(y[te], _lr().fit(F[tr], y[tr]).predict(F[te])))
        res.append(lnorm(np.mean(accs), S))
    return float(np.mean(res))


def probe_wc(F, y, cls, seeds=(0, 1, 2)):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    vals = []
    for sd in seeds:
        per = []
        for c in np.unique(cls):
            m = cls == c
            u, yv = np.unique(y[m], return_inverse=True)
            cnt = np.bincount(yv)
            if len(u) < 2 or cnt.min() < 3: continue
            skf = StratifiedKFold(int(min(5, cnt.min())), shuffle=True, random_state=sd)
            accs = [balanced_accuracy_score(yv[te], _lr().fit(F[m][tr], yv[tr]).predict(F[m][te]))
                    for tr, te in skf.split(F[m], yv)]
            per.append(lnorm(np.mean(accs), len(u)))
        vals.append(np.mean(per))
    return float(np.mean(vals)), len(per)


def residualize(F, cls):
    F = F.copy()
    for c in np.unique(cls):
        m = cls == c; F[m] -= F[m].mean(0, keepdims=True)
    return F


def evaluate(F, y, groups, cls, protocol):
    if protocol == "v1":   # v1: ungrouped, scaler fit on all clips
        return {"cc": probe(residualize(F, cls), y, groups, grouped=False, scale_all=True),
                "mrg": probe(F, y, groups, grouped=False, scale_all=True),
                "wc": probe_wc(F, y, cls)[0]}
    wc, nwc = probe_wc(F, y, cls)
    Fr = residualize(F, cls)
    return {"cc": probe(Fr, y, groups), "mrg": probe(F, y, groups), "wc": wc, "wc_classes": nwc,
            "cc_ungrouped": probe(Fr, y, groups, grouped=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=["v2", "v1"], default="v2")
    ap.add_argument("--with-val", action="store_true")
    ap.add_argument("--only", nargs="*", help="tags to run (default: all encoders + reference rows)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, names, cls, lens, miss = load_v1() if a.protocol == "v1" else load_heldout(a.with_val)
    y = np.unique([signer_of(n) for n in names], return_inverse=True)[1]
    groups = np.array([rec_of(n) for n in names])
    print(f"[{a.protocol}] {len(X)} clips, {len(np.unique(y))} signers, {len(np.unique(groups))} recordings, "
          f"input {X.shape} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["tag"] for l in open(a.out) if l.strip()}
    want = lambda t: (not a.only or t in a.only) and t not in done
    jobs = [(t, ("ref", f), "reference") for t, f in ref_features(X, lens, miss).items()]
    jobs += [(t, ("net", c, k), flag) for t, c, k, flag in ENCODERS]
    for tag, spec, flag in jobs:
        if not want(tag): continue
        if spec[0] == "ref":
            F = spec[1]
        else:
            if spec[2] is not None and not os.path.exists(spec[2]):
                print(f"[skip] {tag}: missing {spec[2]}", flush=True); continue
            F = net_features(spec[1], spec[2], X, dev)
        r = evaluate(F.astype(np.float64), y, groups, cls, a.protocol)
        rec = {"tag": tag, "protocol": a.protocol, "flag": flag, "n": len(X), "S": int(len(np.unique(y))),
               "dim": int(F.shape[1]), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
               "ckpt": spec[2] if spec[0] == "net" else None}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
