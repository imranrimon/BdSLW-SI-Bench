#!/usr/bin/env python
"""Resume-safe twin of main_signer_adv.py (GRL signer-adversarial training).

The training recipe is IDENTICAL to main_signer_adv.py -- the same feeder, model,
SGD+Nesterov optimizer, warm-up/step LR schedule (_lr_at), loss (class CE + signer
CE through the GRL), clipping rule (cfg grad_clip_norm, 0 = off), eval cadence,
best-on-val checkpoint (`<exp>_model-best.pt`, key 'weights') and results_final.csv
rows. The helpers are imported from main_signer_adv itself so they cannot drift.

The only addition: after every epoch it writes `<exp>_resume.pt` (model + optimizer
+ epoch + best_t1/t5 + python/numpy/torch/cuda RNG states) atomically, and on start
it resumes from that file if present (never from epoch 0 once a checkpoint exists).
Restoring the RNG states at the epoch boundary makes the shuffle / random-crop stream
identical to an uninterrupted run (up to cuDNN non-determinism). A final
`<exp>_model-<num_epoch-1>.pt` marks completion; a finished run is skipped.

  python tools/main_signer_adv_resumable.py \
      --config config/lr01_switch/bdsl_block_gcn_si_signeradv_lr01.yaml \
      --seed 1 --grl-lambda 0.1 -Experiment_name bdsl_block_gcn_si_signeradv_lr01_gl01_seed1
"""
import argparse
import csv
import datetime
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # main_signer_adv writes ./work_dir and ./results_final.csv relative to the repo root

from feeders.feeder import Feeder  # noqa: E402
from main_signer_adv import _import_class, _init_seed, _lr_at, evaluate  # noqa: E402


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-Experiment_name", dest="exp_override", default=None)
    ap.add_argument("--grl-lambda", type=float, default=None)
    args, _ = ap.parse_known_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.grl_lambda is not None:
        cfg["model_args"]["grl_lambda"] = args.grl_lambda

    exp = args.exp_override or cfg.get("Experiment_name", "bdsl_signer_adv")
    work_dir = os.path.join(".", "work_dir", exp)
    os.makedirs(work_dir, exist_ok=True)
    num_epoch = int(cfg.get("num_epoch", 100))
    final_ck = os.path.join(work_dir, f"{exp}_model-{num_epoch - 1}.pt")
    resume_ck = os.path.join(work_dir, f"{exp}_resume.pt")
    if os.path.exists(final_ck):
        print(f"[done] {exp}: final checkpoint exists, skipping", flush=True)
        return
    with open(os.path.join(work_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(dict(cfg, Experiment_name=exp, seed=args.seed), f, sort_keys=False)

    _init_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda {torch.cuda.is_available()}  exp {exp}", flush=True)

    tf = dict(cfg["train_feeder_args"]); tf["return_signer"] = True
    train_set = Feeder(**tf)
    val_set = Feeder(**cfg["test_feeder_args"])
    bs = int(cfg.get("batch_size", 32))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=int(cfg.get("num_worker", 0)), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=int(cfg.get("test_batch_size", 32)),
                            shuffle=False, num_workers=int(cfg.get("num_worker", 0)))
    num_signer = train_set.num_signer
    print(f"train {len(train_set)} val {len(val_set)} | num_signer {num_signer} "
          f"(map {train_set.signer_to_idx})", flush=True)

    margs = dict(cfg["model_args"])
    margs["num_signer"] = num_signer
    model = _import_class(cfg["model"])(**margs).to(device)
    ce = nn.CrossEntropyLoss()

    base_lr = float(cfg.get("base_lr", 0.1))
    warmup = int(cfg.get("warm_up_epoch", 5))
    steps = list(cfg.get("step", [60, 80]))
    clip = float(cfg.get("grad_clip_norm", 5.0))
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                nesterov=True, weight_decay=float(cfg.get("weight_decay", 1e-4)))

    best_t1 = best_t5 = 0.0
    start_epoch = 0
    if os.path.exists(resume_ck):
        st = torch.load(resume_ck, map_location="cpu")  # an unreadable file raises: never silently restart
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        start_epoch = int(st["epoch"]) + 1
        best_t1, best_t5 = float(st["best_t1"]), float(st["best_t5"])
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(f"[resume] {exp} from epoch {start_epoch} (best val Top1 {best_t1*100:.2f})", flush=True)

    for epoch in range(start_epoch, num_epoch):
        lr = _lr_at(epoch, base_lr, warmup, steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        closs = sloss = 0.0; nb = 0
        for data, label, signer, _idx in train_loader:
            data = data.float().to(device); label = label.long().to(device)
            signer = signer.long().to(device)
            class_logits, signer_logits = model(data)
            lc = ce(class_logits, label)
            ls = ce(signer_logits, signer)
            loss = lc + ls
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            closs += lc.item(); sloss += ls.item(); nb += 1
        if (epoch + 1) % int(cfg.get("eval_interval", 5)) == 0 or epoch == num_epoch - 1:
            t1, t5 = evaluate(model, val_loader, device)
            print(f"[epoch {epoch}] lr {lr:.4f} class_loss {closs/nb:.4f} "
                  f"signer_loss {sloss/nb:.4f} | val Top1 {t1*100:.2f} Top5 {t5*100:.2f}", flush=True)
            if t1 > best_t1:
                best_t1, best_t5 = t1, t5
                _atomic_save({"weights": model.state_dict(), "acc": t1, "epoch": epoch},
                             os.path.join(work_dir, f"{exp}_model-best.pt"))
            with open("results_final.csv", "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    exp, epoch, f"{t1:.4f}", f"{t5:.4f}",
                    "same_epoch_as_logged_top1", work_dir])
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "best_t1": best_t1, "best_t5": best_t5,
                      "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                     resume_ck)
    _atomic_save({"weights": model.state_dict(), "epoch": num_epoch - 1}, final_ck)
    print(f"DONE {exp} best Top1 {best_t1*100:.2f} Top5 {best_t5*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
