#!/usr/bin/env python
"""P2 missing-joint fixes: side-effect-free per-clip score dumps for the analysis (GPU).

Scores each needed checkpoint on TEST (and VAL where the fusion weight is chosen) with the run's
own config.yaml test_feeder_args (feeder exactly as main.py's test phase; only the npy path is
swapped). Writes docs/p2_missing_joint_fixes_2026-10/raw/scores/<key>.pkl ({clip: logits}) and
appends {key, top1, ckpt} to raw/scores/index.jsonl. Idempotent; every run is gated on
completeness (its final-epoch checkpoint must exist), never on a partial run.
  python tools/p2_mjfix_score.py [--sections bone cue rand]
"""
import argparse, glob, importlib, json, os, pickle, re, sys
import numpy as np, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = "docs/p2_missing_joint_fixes_2026-10/raw/scores"
W = "work_dir"
SEEDS = (0, 1, 2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if not hasattr(np, "_core"):
    import numpy.core as _c
    sys.modules.setdefault("numpy._core", _c)
    for _s in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        if f"numpy.core.{_s}" in sys.modules: sys.modules.setdefault(f"numpy._core.{_s}", sys.modules[f"numpy.core.{_s}"])


def imp(n):
    m, c = n.rsplit(".", 1); return getattr(importlib.import_module(m), c)


_FEED = {}


def feeder(cfg, data, label):
    fa = dict(cfg["test_feeder_args"]); fa["data_path"] = data; fa["label_path"] = label
    k = json.dumps(fa, sort_keys=True)
    if k not in _FEED:
        if len(_FEED) >= 4: _FEED.pop(next(iter(_FEED)))
        _FEED[k] = imp(cfg["feeder"])(**fa)
    return _FEED[k]


@torch.no_grad()
def score(cfg_path, ckpt, data, label):
    cfg = yaml.safe_load(open(cfg_path))
    model = imp(cfg["model"])(**cfg["model_args"]).to(DEV).eval()
    ck = torch.load(ckpt, map_location="cpu")
    w = ck["weights"] if isinstance(ck, dict) and "weights" in ck else ck
    miss, unexp = model.load_state_dict({k.replace("module.", ""): v for k, v in w.items()}, strict=False)
    assert not miss and not unexp, (ckpt, miss[:3], unexp[:3])
    ds = feeder(cfg, data, label)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    out, i = {}, 0
    for b in dl:
        s = model(b[0].float().to(DEV))
        s = (s[0] if isinstance(s, (tuple, list)) else s).cpu().numpy()
        for r in s: out[ds.sample_name[i]] = r; i += 1
    y = dict(zip(ds.sample_name, ds.label))
    return out, 100.0 * np.mean([np.argmax(v) == y[k] for k, v in out.items()])


def complete(run, n_ep=100):
    return os.path.exists(f"{W}/{run}/{run}_model-{n_ep - 1}.pt")


def best(run): return f"{W}/{run}/{run}_model-best.pt"


def dump(key, cfg, ckpt, data, label, gate_run=None, save=True):
    p = f"{OUT}/{key}.pkl"
    if gate_run and not complete(gate_run):
        print(f"[gate] {key}: {gate_run} not complete -> skipped", flush=True); return None
    if not os.path.exists(ckpt):
        print(f"[missing] {key}: {ckpt}", flush=True); return None
    if save and os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(ckpt):
        return None
    sc, acc = score(cfg, ckpt, data, label)
    if save: pickle.dump(sc, open(p, "wb"))
    rec = {"key": key, "top1": round(acc, 4), "ckpt": ckpt, "data": data}
    with open(f"{OUT}/index.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return acc


SI, BO, BN = "data/bdsl_si", "data/bdsl_si", "data/bdsl_si_bonemask"


def bone():
    L = {sp: f"{SI}/{sp}_label.pkl" for sp in ("val", "test")}
    for s in SEEDS:
        runs = {  # key -> (run dir holding config.yaml, ckpt, data dir, data suffix, gate)
            f"bgj_s{s}": (f"bdsl_block_gcn_si_lr01_scratch_seed{s}", None, SI, "", True),
            f"bgb_old_s{s}": (f"bdsl_block_gcn_si_bone_vec_lr01_seed{s}", None, BO, "_bone", True),
            f"bgb_new_s{s}": (f"bdsl_block_gcn_si_bonemask_vec_lr01_seed{s}", None, BN, "_bone", True),
            f"agj_s{s}": (f"bdsl_adaptive_gnn_si_seed{s}",
                          f"{W}/bdsl_adaptive_gnn_si_seed{s}/bdsl_adaptive_gnn_si_seed{s}_model-{[59, 59, 89][s]}.pt", SI, "", False),
            f"agb_old_s{s}": (f"bdsl_adaptive_gnn_bone_clip5_seed{s}", None, BO, "_bone", True),
            f"agb_new_s{s}": (f"bdsl_adaptive_gnn_bonemask_clip5_seed{s}", None, BN, "_bone", True),
            f"agj1_s{s}": (f"bdsl_adaptive_gnn_si_lr0p1_seed{s}", None, SI, "", True),
            f"agb1_old_s{s}": (f"bdsl_adaptive_gnn_bone_lr0p1_seed{s}", None, BO, "_bone", True),
            f"agb1_new_s{s}": (f"bdsl_adaptive_gnn_bonemask_lr0p1_seed{s}", None, BN, "_bone", True),
        }
        for key, (run, ck, d, suf, gate) in runs.items():
            for sp in ("test", "val"):
                dump(f"{key}_{sp}", f"{W}/{run}/config.yaml", ck or best(run), f"{d}/{sp}_data{suf}.npy", L[sp],
                     gate_run=run if gate else None)


def cue():
    for s in SEEDS:
        runs = {f"cuectl_s{s}": ("bdsl_block_gcn_si_lr01_scratch_seed{s}", SI),
                f"cue_speed_old_s{s}": ("bdsl_cue_speednorm_lr01_seed{s}", "data/bdsl_cue_speednorm"),
                f"cue_speed_new_s{s}": ("bdsl_cue_speednorm_masked_lr01_seed{s}", "data/bdsl_cue_speednorm_masked"),
                f"cue_root_old_s{s}": ("bdsl_cue_rootcenter_lr01_seed{s}", "data/bdsl_cue_rootcenter"),
                f"cue_root_new_s{s}": ("bdsl_cue_rootcenter_masked_lr01_seed{s}", "data/bdsl_cue_rootcenter_masked")}
        for key, (run, d) in runs.items():
            run = run.format(s=s)
            dump(key + "_test", f"{W}/{run}/config.yaml", f"{W}/{run}/{run}_model-99.pt", f"{d}/test_data.npy",
                 f"{d}/test_label.pkl", gate_run=run)


def rand():
    """Val-selected (best-on-val) and final checkpoints on the split's TEST; plus every saved checkpoint
    on TEST (= the paper's optimistic best-on-own-test estimator, recomputed on these runs)."""
    for r in range(42, 47):
        run = f"bdsl_adaptive_gnn_rand{r}_valsel_seed0"
        if not complete(run):
            print(f"[gate] {run} not complete", flush=True); continue
        d = f"./data/bdsl_rand{r}"
        dump(f"rand{r}_valsel_test", f"{W}/{run}/config.yaml", best(run), f"{d}/test_data.npy", f"{d}/test_label.pkl")
        scan = f"{OUT}/rand{r}_scan.json"
        if os.path.exists(scan): continue
        res = {}
        for ck in sorted(glob.glob(f"{W}/{run}/{run}_model-*.pt")):
            m = re.search(r"_model-(\d+)\.pt$", ck)
            if not m: continue
            _, acc = score(f"{W}/{run}/config.yaml", ck, f"{d}/test_data.npy", f"{d}/test_label.pkl")
            _, vacc = score(f"{W}/{run}/config.yaml", ck, f"{d}/val_data.npy", f"{d}/val_label.pkl")
            res[int(m.group(1))] = {"test": acc, "val": vacc}
            print(f"  {run} ep{m.group(1)} test {acc:.2f} val {vacc:.2f}", flush=True)
        json.dump(res, open(scan, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=["bone", "cue", "rand"], choices=["bone", "cue", "rand"])
    os.makedirs(OUT, exist_ok=True)
    for sec in ap.parse_args().sections:
        globals()[sec]()
