"""Pose-motion clip descriptor and RBF-kernel MMD used by the signer-covariate analysis.

clip_descriptor: per-joint mean and std of frame-to-frame (x, y) speed over the
non-padding frames of one (C, T, V, M) clip, i.e. a motion signature that is
independent of absolute position.
mmd_rbf: unbiased squared MMD between two descriptor samples with an RBF kernel
whose bandwidth is the median pairwise squared distance (median heuristic).
"""
import numpy as np


def clip_descriptor(arr):
    # arr: (C,T,V,M) -> use x,y ; speed_t = |p_{t+1}-p_t|
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]                      # (2,T,V)
    # drop all-zero (padding) frames
    nz = np.abs(xy).sum((0, 2)) > 0
    xy = xy[:, nz, :] if nz.any() else xy
    if xy.shape[1] < 2:
        return None
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=0)   # (T-1, V)
    return np.concatenate([vel.mean(0), vel.std(0)])


def mmd_rbf(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sigma2 = np.median(d2[d2 > 0]) + 1e-9
    K = np.exp(-d2 / sigma2)
    n, m = len(X), len(Y)
    Kxx = K[:n, :n]; Kyy = K[n:, n:]; Kxy = K[:n, n:]
    return (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) \
         - 2 * Kxy.mean()
