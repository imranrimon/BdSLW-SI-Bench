#!/usr/bin/env python
"""BdSL-SPOTER under its own released training recipe ("native recipe" row,
analogous to Track B; NOT the shared-recipe main-table row).

Recipe taken from the released notebook (github.com/sayad-dot/BDSLW_SPOTER,
BdSL_SPOTER_Phase4_RealData.ipynb, `ProductionSPOTER` + `AdvancedTrainingConfig`
+ `BdSLDataAugmentation`): 8 encoder layers, AdamW (lr 3e-4, wd 0.05, betas
0.9/0.95), OneCycleLR (max_lr 1e-3, pct_start 0.1) over 100 epochs, label
smoothing 0.1, grad-clip 1.0, batch 8 with 2-step accumulation, augmentation
with prob 0.7 {temporal rescale x0.8-1.2 (p .3), Gaussian coord noise sd 0.02
(p .4), in-plane rotation +-10 deg (p .3), hand-node jitter sd 0.05 (p .5)}.
Not reproduced: their 54-landmark 2D input (we keep the benchmark's identical
27-kpt x 3 input), the length curriculum (stages under-specified and
inconsistent between paper and code), and the paper's horizontal flip (absent
from the released augmentation class). Selection: best val-U15 epoch; that
epoch's model is scored on held-out TEST (or, with --mixed, on the sd val split
used for 'Mixed'). Writes <work_dir>/eval_results/{best_acc.pkl,test_scores.pkl}
and a result JSON; resumes from <work_dir>/last.pt on requeue.
"""
import argparse, json, os, pickle, random, sys, time
import numpy as np
import yaml
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from feeders.feeder import Feeder  # noqa: E402
from model.spoter import Model  # noqa: E402

HAND_NODES = list(range(7, 27))


class NativeAug(Dataset):
    """Wraps the repo Feeder (identical crop/normalisation) and adds the released augmentations."""

    def __init__(self, base, train):
        self.base, self.train = base, train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, idx = self.base[i]
        x = np.array(x, dtype=np.float32)          # (C, T, V, M)
        if self.train and random.random() < 0.7:
            C, T, V, M = x.shape
            valid = (np.abs(x).sum(axis=(0, 2, 3)) > 0)
            if random.random() < 0.3:              # temporal rescale, back onto T frames
                L = max(2, int(valid.sum()) or T)
                s = random.uniform(0.8, 1.2)
                src = np.linspace(0, L - 1, max(2, int(L * s)))
                lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, L - 1); w = (src - lo)[None, :, None, None]
                seq = (1 - w) * x[:, lo] + w * x[:, hi]
                out = np.zeros_like(x); n = min(T, seq.shape[1]); out[:, :n] = seq[:, :n]; x = out
                valid = (np.abs(x).sum(axis=(0, 2, 3)) > 0)
            nz = (x[:2] != 0)
            if random.random() < 0.4:
                x[:2] += np.random.normal(0, 0.02, x[:2].shape).astype(np.float32) * nz
            if random.random() < 0.3:
                a = np.radians(random.uniform(-10, 10)); c, s_ = np.cos(a), np.sin(a)
                xx, yy = x[0].copy(), x[1].copy()
                x[0] = np.where(nz[0], c * xx - s_ * yy, 0); x[1] = np.where(nz[1], s_ * xx + c * yy, 0)
            if random.random() < 0.5:
                j = np.random.normal(0, 0.05, x[:2][:, :, HAND_NODES].shape).astype(np.float32)
                x[:2][:, :, HAND_NODES] += j * nz[:, :, HAND_NODES]
        return x, y, idx


@torch.no_grad()
def score(model, loader, dev):
    model.eval(); out, ys = [], []
    for x, y, _ in loader:
        out.append(torch.softmax(model(x.float().to(dev)), 1).cpu()); ys.append(y)
    P = torch.cat(out).numpy(); Y = torch.cat(ys).numpy()
    return P, Y, float((P.argmax(1) == Y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["si", "sd"], default="si")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda")
    wd = os.path.join(ROOT, "work_dir", a.exp); os.makedirs(os.path.join(wd, "eval_results"), exist_ok=True)
    d = f"data/bdsl_{a.split}"
    fk = dict(window_size=120, normalization=True, random_mirror=False)
    tr = NativeAug(Feeder(f"{d}/train_data.npy", f"{d}/train_label.pkl", random_choose=True, random_shift=False, **fk), True)
    va = Feeder(f"{d}/val_data.npy", f"{d}/val_label.pkl", **fk)
    te = Feeder("data/bdsl_si/test_data.npy", "data/bdsl_si/test_label.pkl", **fk) if a.split == "si" else None
    g = torch.Generator(); g.manual_seed(a.seed)
    trl = DataLoader(tr, batch_size=8, shuffle=True, num_workers=4, drop_last=True, generator=g)
    val = DataLoader(va, batch_size=64, num_workers=2)
    tel = DataLoader(te, batch_size=64, num_workers=2) if te else None

    model_args = dict(num_class=60, num_point=27, num_person=1, in_channels=3,
                      variant="bdsl_spoter", num_layers=a.layers, dropout=0.15)
    # config.yaml lets tools/eval_ckpt_on_npy.py rebuild model + test feeder (sd-test Mixed scoring)
    with open(os.path.join(wd, "config.yaml"), "w") as f:
        yaml.safe_dump(dict(model="model.spoter.Model", model_args=model_args, feeder="feeders.feeder.Feeder",
                            test_feeder_args=dict(data_path=f"{d}/val_data.npy", label_path=f"{d}/val_label.pkl", **fk),
                            test_batch_size=64, seed=a.seed, split=a.split, recipe="bdsl_spoter_native"), f)
    model = Model(**model_args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8)
    steps = a.epochs * (len(trl) // 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    start, best, best_ep, log = 0, -1.0, -1, []
    last = os.path.join(wd, "last.pt")
    if os.path.exists(last):  # resume on requeue (never restart from 0)
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, best_ep, log = ck["epoch"] + 1, ck["best"], ck["best_ep"], ck["log"]
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at epoch {start} (best val {best:.4f} @ {best_ep})")
    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); opt.zero_grad(); tot = n = 0
        for it, (x, y, _) in enumerate(trl):
            loss = crit(model(x.float().to(dev)), y.long().to(dev)) / 2
            loss.backward(); tot += loss.item() * 2; n += 1
            if (it + 1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                if sched.last_epoch < steps - 1: sched.step()
        Pv, Yv, acc = score(model, val, dev)
        log.append((ep, tot / max(1, n), acc))
        if acc > best:
            best, best_ep = acc, ep
            torch.save(model.state_dict(), os.path.join(wd, "model-best.pt"))
            names = va.sample_name
            pickle.dump(dict(zip(names, Pv)), open(os.path.join(wd, "eval_results", "best_acc.pkl"), "wb"))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep,
                        best=best, best_ep=best_ep, log=log, py_rng=random.getstate(),
                        np_rng=np.random.get_state(), torch_rng=torch.get_rng_state()), last)
        print(f"ep {ep} loss {tot / max(1, n):.4f} val {acc:.4f} (best {best:.4f}@{best_ep}) {time.time() - t0:.0f}s", flush=True)

    res = dict(exp=a.exp, seed=a.seed, split=a.split, layers=a.layers, best_val=best, best_epoch=best_ep)
    if tel is not None:
        model.load_state_dict(torch.load(os.path.join(wd, "model-best.pt"), map_location=dev))
        Pt, Yt, tacc = score(model, tel, dev)
        pickle.dump(dict(zip(te.sample_name, Pt)), open(os.path.join(wd, "eval_results", "test_scores.pkl"), "wb"))
        res["test_top1"] = tacc
    json.dump(res, open(os.path.join(wd, "result.json"), "w"), indent=1)
    open(os.path.join(wd, ".done"), "w").write("ok\n")
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
