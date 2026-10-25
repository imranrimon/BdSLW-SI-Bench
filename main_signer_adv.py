"""Signer-adversarial supervised training (Option: direct shortcut mitigation).

Trains a `return_features` backbone with a class head + a GRL signer head
(model.signer_adv.Model) so the learned sign features are signer-invariant. Mirrors
the shared SI recipe (SGD+nesterov, warmup, step decay, grad-clip) for comparability
with the main.py 8-arch table; logs best-val Top-1/Top-5 to results_final.csv.

  python main_signer_adv.py --config config/bdsl_block_gcn_si_signeradv.yaml --seed 0
"""
import argparse
import csv
import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from feeders.feeder import Feeder


def _import_class(dotted):
    mod, cls = dotted.rsplit('.', 1)
    return getattr(__import__(mod, fromlist=[cls]), cls)


def _init_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _lr_at(epoch, base_lr, warmup, steps):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    lr = base_lr
    for s in steps:
        if epoch >= s:
            lr *= 0.1
    return lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    t1 = t5 = n = 0
    for data, label, *_ in loader:
        data = data.float().to(device); label = label.long().to(device)
        logits, _ = model(data)
        _, top5 = logits.topk(5, 1)
        t1 += (top5[:, 0] == label).sum().item()
        t5 += (top5 == label.unsqueeze(1)).any(1).sum().item()
        n += label.numel()
    return t1 / n, t5 / n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-Experiment_name", dest="exp_override", default=None)
    ap.add_argument("--grl-lambda", type=float, default=None,
                    help="override model_args.grl_lambda (adversarial strength sweep)")
    args, _ = ap.parse_known_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.grl_lambda is not None:
        cfg["model_args"]["grl_lambda"] = args.grl_lambda

    exp = args.exp_override or cfg.get("Experiment_name", "bdsl_signer_adv")
    work_dir = os.path.join(".", "work_dir", exp)
    os.makedirs(work_dir, exist_ok=True)
    _init_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda {torch.cuda.is_available()}  exp {exp}")

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
          f"(map {train_set.signer_to_idx})")

    margs = dict(cfg["model_args"])
    margs["num_signer"] = num_signer
    model = _import_class(cfg["model"])(**margs).to(device)
    ce = nn.CrossEntropyLoss()

    base_lr = float(cfg.get("base_lr", 0.1))
    warmup = int(cfg.get("warm_up_epoch", 5))
    steps = list(cfg.get("step", [60, 80]))
    num_epoch = int(cfg.get("num_epoch", 100))
    clip = float(cfg.get("grad_clip_norm", 5.0))
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                nesterov=True, weight_decay=float(cfg.get("weight_decay", 1e-4)))

    best_t1 = best_t5 = 0.0
    for epoch in range(num_epoch):
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
            ls = ce(signer_logits, signer)      # GRL reverses grad to backbone
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
                  f"signer_loss {sloss/nb:.4f} | val Top1 {t1*100:.2f} Top5 {t5*100:.2f}")
            if t1 > best_t1:
                best_t1, best_t5 = t1, t5
                torch.save({"weights": model.state_dict()},
                           os.path.join(work_dir, f"{exp}_model-best.pt"))
            with open("results_final.csv", "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    exp, epoch, f"{t1:.4f}", f"{t5:.4f}",
                    "same_epoch_as_logged_top1", work_dir])
    print(f"DONE {exp} best Top1 {best_t1*100:.2f} Top5 {best_t5*100:.2f}")


if __name__ == "__main__":
    main()
