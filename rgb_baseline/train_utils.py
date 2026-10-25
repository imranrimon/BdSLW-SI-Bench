"""Small training helpers shared by the RGB baselines.

Same learning-rate schedule family as ``main.py`` (linear warm-up, then step decay)
and a Top-k accuracy over a score matrix.
"""
import numpy as np


def adjust_learning_rate(optimizer, epoch, base_lr, step, warm_up_epoch):
    # Same schedule family as main.py: linear warm-up then step decay.
    if epoch < warm_up_epoch:
        lr = base_lr * (epoch + 1) / warm_up_epoch
    else:
        lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def topk_from_scores(scores, labels, k):
    rank = scores.argsort(axis=1)
    hit = [labels[i] in rank[i, -k:] for i in range(len(labels))]
    return sum(hit) / max(1, len(hit))
