"""Signer-adversarial wrapper for supervised SI training.

Motivation: the SI-vs-SD study shows the signer-identity shortcut is *pervasive*
(even pose kinematics leak it), so signer-invariance must be trained in, not assumed
from the representation. This wraps any `return_features` backbone with:
  * a class head (the task), and
  * a signer head behind a Gradient Reversal Layer (GRL).
The GRL makes the backbone learn features from which the signer CANNOT be predicted,
while the class head still solves the sign task — i.e. sign features that are
signer-invariant. Trained via `main_signer_adv.py`.

forward(x) -> (class_logits, signer_logits).
"""
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


def _import_class(dotted):
    mod, cls = dotted.rsplit('.', 1)
    return getattr(__import__(mod, fromlist=[cls]), cls)


class Model(nn.Module):
    def __init__(self, backbone, backbone_args, num_class, num_signer,
                 feat_dim=256, grl_lambda=1.0, signer_hidden=256):
        super().__init__()
        Backbone = _import_class(backbone)
        bargs = dict(backbone_args or {})
        bargs['return_features'] = True
        self.backbone = Backbone(**bargs)
        self.grl_lambda = float(grl_lambda)
        self.class_head = nn.Linear(feat_dim, num_class)
        self.signer_head = nn.Sequential(
            nn.Linear(feat_dim, signer_hidden), nn.ReLU(inplace=True),
            nn.Linear(signer_hidden, num_signer),
        )

    def forward(self, x, keep_prob=1.0):
        feat = self.backbone(x, keep_prob)            # (N, feat_dim)
        if isinstance(feat, tuple):
            feat = feat[0]
        class_logits = self.class_head(feat)
        signer_logits = self.signer_head(grad_reverse(feat, self.grl_lambda))
        return class_logits, signer_logits
