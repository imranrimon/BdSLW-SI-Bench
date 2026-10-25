#!/usr/bin/env python
"""Signer-decodability (L_norm) of signer-normalized encoders with probe v2's protocol.

tools/decodability_probe_v2.py builds its held-out inputs with feeders.feeder.Feeder and fixed test args,
which is wrong for encoders trained with feeders.feeder_signernorm (body frame, 5 channels). This wrapper
reuses v2's clip set (SI test U02/U13 + labelled pretrain pool, 9 signers), recording-grouped CV and
metrics (v2.evaluate: cc / mrg / wc / cc_ungrouped, CV seeds 0-2), but feeds each encoder ITS OWN
test-time input: the feeder + test_feeder_args from work_dir/<run>/config.yaml (data paths swapped).
Baseline encoders (base Feeder) go through the identical code path, so rows are directly comparable.

  python tools/signernorm_probe.py --runs bdsl_block_gcn_si_lr01_scratch_seed0 bdsl_block_gcn_si_lr01_sn_abcd_seed0 \
      --out docs/signernorm_2026-10/raw/probe_v2_signernorm.jsonl
"""
import argparse, json, os, sys, time
import numpy as np
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from tools.decodability_probe import _alias_numpy_core, _imp  # noqa: E402
from tools.decodability_probe_v2 import evaluate, rec_of, signer_of  # noqa: E402
from main import import_class  # noqa: E402

SPLITS = ("test", "pretrain")


def heldout_inputs(cfg):
    Feeder = import_class(cfg["feeder"])
    Xs, names, cls = [], [], []
    for sp in SPLITS:
        fa = dict(cfg["test_feeder_args"])
        fa.update(data_path=f"data/bdsl_si/{sp}_data.npy", label_path=f"data/bdsl_si/{sp}_label.pkl",
                  random_choose=False, window_size=120, random_mirror=False)
        fd = Feeder(**fa)
        for i in range(len(fd)):
            x, y, _ = fd[i]
            Xs.append(np.asarray(x, dtype=np.float32)); cls.append(int(y)); names.append(fd.sample_name[i])
    return np.stack(Xs), names, np.array(cls)


@torch.no_grad()
def features(cfg, ckpt, X, dev, bs=64):
    kw = dict(cfg["model_args"]); kw["return_features"] = True
    bb = _imp(cfg["model"])(**kw).to(dev).eval()
    _alias_numpy_core()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("weights", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = bb.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    if missing:
        raise RuntimeError(f"{ckpt}: missing backbone params {missing[:5]}")
    out = []
    for i in range(0, len(X), bs):
        f = bb(torch.from_numpy(X[i:i + bs]).to(dev))
        if f.dim() == 3: f = f.mean(dim=1)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="work_dir run names (best checkpoint is probed); run@ckpt.pt picks another checkpoint")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = {json.loads(l)["tag"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
    cache = {}
    for spec in a.runs:
        run, _, ck = spec.partition("@")          # run@<checkpoint basename> overrides <run>_model-best.pt
        tag = f"{run}@{ck}" if ck else run
        if tag in done:
            continue
        t0 = time.time()
        cfg = yaml.safe_load(open(f"work_dir/{run}/config.yaml"))
        key = json.dumps([cfg["feeder"], cfg["test_feeder_args"]], sort_keys=True, default=str)
        if key not in cache:
            cache[key] = heldout_inputs(cfg)
        X, names, cls = cache[key]
        y = np.unique([signer_of(n) for n in names], return_inverse=True)[1]
        groups = np.array([rec_of(n) for n in names])
        ckpt = f"work_dir/{run}/{ck or run + '_model-best.pt'}"
        F = features(cfg, ckpt, X, dev)
        r = evaluate(F.astype(np.float64), y, groups, cls, "v2")
        rec = {"tag": tag, "protocol": "v2-own-feeder", "n": len(X), "S": int(len(np.unique(y))),
               "input": list(X.shape[1:]), "dim": int(F.shape[1]),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, "ckpt": ckpt,
               "secs": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
