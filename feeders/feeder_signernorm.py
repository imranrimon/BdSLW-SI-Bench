"""Signer-normalized pose feeder (label-free) for the signer-independent BdSLW60 benchmark.

Subclasses feeders.feeder.Feeder WITHOUT modifying it (running jobs import feeder.py). Every option
defaults to off; with all options off the output is identical to Feeder (tests/test_feeder_signernorm.py).

27-node layout (preprocessing/preprocess_bdsl.py; MediaPipe Holistic, subject's left/right):
  0 nose | 1 R-shoulder 2 L-shoulder | 3 R-elbow 4 L-elbow | 5 R-wrist 6 L-wrist (pose)
  7-16  RIGHT hand (MediaPipe right_hand_landmarks): wrist, thumbTip, indexMCP, indexTip, midMCP,
        midTip, ringMCP, ringTip, pinkyMCP, pinkyTip          (graph edge 5-7)
  17-26 LEFT hand, same order                                  (graph edge 6-17)
  (feeder.py's keypoint_drop comment calls 7-16 "left hand"; that label is wrong but flip_index is
   a symmetric swap, so mirroring is unaffected.)
Channels: 0 x, 1 y (image-normalized, 0 = missing), 2 visibility (1 body / 0 hands).

Options
  (a) body_frame:  per-frame shoulder-midpoint centering, scale by shoulder width
                   (body_scale='frame' per-frame width | 'clip' clip-median width). Frames lacking
                   shoulders borrow the nearest valid frame's frame; a clip with no shoulders falls
                   back to its centroid and `fallback_shoulder_width`. Missing points/padded frames stay 0.
                   body_scale='none' = centring-only control (shoulder-midpoint centring, no scaling,
                   padding/missing kept at 0) isolating the fix of Feeder's centring artefact.
                   Replaces Feeder's per-clip nose-mean centering (`normalization`), which is skipped.
  (b) hand_local:  2 extra channels (C 3 -> 5): wrist-relative hand coordinates divided by palm size
                   (max wrist->MCP distance; hand_scale='clip' clip-median | 'frame'). Body nodes 0.
  (c) random_mirror (inherited): flip_index L/R swap + x -> 1-x, before (a).
  oracle_hand_canon: ORACLE upper bound using the _L/_R filename tag (metadata, not deployable):
                   mirror '_L' clips to right-hand at train and test.
  (d) bone_aug='signer': with prob bone_aug_p, retarget every bone of the kinematic tree rooted at the
                   shoulder midpoint to a randomly chosen OTHER training signer's mean bone lengths
                   (vector scaled by L_target/L_source, signer means in shoulder-width units, so
                   per-frame foreshortening is kept). bone_aug='jitter': per-bone U(1-s, 1+s) scale.
"""
import random
import re

import numpy as np

from feeders import tools
from feeders.feeder import Feeder, flip_index  # noqa: F401  (flip_index re-exported for tests)

R_SH, L_SH = 1, 2
R_HAND = list(range(7, 17))
L_HAND = list(range(17, 27))
HAND_WRISTS = (7, 17)
MCP_OFFSETS = (2, 4, 6, 8)          # indexMCP, midMCP, ringMCP, pinkyMCP relative to hand wrist
ROOT = -1                           # virtual root = shoulder midpoint
# (parent, child, retarget?) in topological order. Shoulders are fixed by the body frame
# (half shoulder width == 0.5 in shoulder-width units for every clip), pose-wrist -> hand-wrist is a
# cross-detector offset (rigid), the rest are retargeted.
_HAND_TREE = lambda w: [(w, w + 1, True), (w, w + 2, True), (w + 2, w + 3, True), (w, w + 4, True),
                        (w + 4, w + 5, True), (w, w + 6, True), (w + 6, w + 7, True),
                        (w, w + 8, True), (w + 8, w + 9, True)]
KIN_TREE = ([(ROOT, 0, True), (ROOT, R_SH, False), (ROOT, L_SH, False),
             (R_SH, 3, True), (3, 5, True), (L_SH, 4, True), (4, 6, True),
             (5, 7, False), (6, 17, False)] + _HAND_TREE(7) + _HAND_TREE(17))
RETARGET_BONES = [(p, c) for p, c, r in KIN_TREE if r]


def _signer_of(name):
    m = re.match(r'U(\d+)', str(name))
    return int(m.group(1)) if m else -1


def _valid(xy):
    """xy (2, T, V[, M]) -> point present mask (T, V[, M])."""
    return (xy[0] != 0) | (xy[1] != 0)


def _fill_nearest(vals, ok):
    """vals (..., T), ok (T,) bool -> vals with ~ok frames copied from the nearest ok frame."""
    idx = np.nonzero(ok)[0]
    T = ok.shape[0]
    near = idx[np.clip(np.searchsorted(idx, np.arange(T)), 0, len(idx) - 1)]
    left = idx[np.clip(np.searchsorted(idx, np.arange(T)) - 1, 0, len(idx) - 1)]
    near = np.where(np.abs(left - np.arange(T)) <= np.abs(near - np.arange(T)), left, near)  # tie -> earlier
    return vals[..., near]


def body_frame_normalize(data, scale='frame', fallback_width=0.253, eps=1e-6):
    """data (C>=2, T, V, M) image coords -> copy with x,y in a per-frame shoulder frame."""
    out = np.array(data, dtype=np.float32, copy=True)
    for m in range(out.shape[3]):
        xy = out[:2, :, :, m]                                    # (2, T, V)
        pv = _valid(xy)                                          # (T, V)
        fr = pv.any(1)                                           # frames with any data
        if not fr.any():
            continue
        sh_ok = pv[:, R_SH] & pv[:, L_SH]
        center = 0.5 * (xy[:, :, R_SH] + xy[:, :, L_SH])         # (2, T)
        width = np.linalg.norm(xy[:, :, R_SH] - xy[:, :, L_SH], axis=0)
        sh_ok &= width > eps
        if sh_ok.any():
            if not sh_ok[fr].all():
                center = _fill_nearest(center, sh_ok)
                width = _fill_nearest(width, sh_ok)
            if scale == 'clip':
                width = np.full_like(width, np.median(width[sh_ok]))
            elif scale == 'none':                                # centring-only control
                width = np.ones_like(width)
        else:                                                    # no shoulders anywhere: clip centroid
            c = np.array([xy[0][pv].mean(), xy[1][pv].mean()], dtype=np.float32)
            center = np.repeat(c[:, None], xy.shape[1], 1)
            width = np.full(xy.shape[1], 1.0 if scale == 'none' else fallback_width, dtype=np.float32)
        nxy = (xy - center[:, :, None]) / np.maximum(width, eps)[None, :, None]
        out[:2, :, :, m] = np.where(pv[None], nxy, 0.0)
    return out


def hand_local_channels(data, scale='clip', eps=1e-6):
    """data (C>=2, T, V, M) -> (2, T, V, M) wrist-relative hand coords / palm size (0 elsewhere)."""
    C, T, V, M = data.shape
    loc = np.zeros((2, T, V, M), dtype=np.float32)
    for m in range(M):
        xy = data[:2, :, :, m]
        pv = _valid(xy)
        for w in HAND_WRISTS:
            nodes = list(range(w, w + 10))
            ok = pv[:, w]                                        # hand present (all-or-none)
            if not ok.any():
                continue
            rel = xy[:, :, nodes] - xy[:, :, w:w + 1]            # (2, T, 10)
            palm = np.max(np.stack([np.linalg.norm(xy[:, :, w + o] - xy[:, :, w], axis=0)
                                    for o in MCP_OFFSETS]), axis=0)          # (T,)
            if scale == 'clip':
                good = ok & (palm > eps)
                s = np.full(T, np.median(palm[good]) if good.any() else 1.0, dtype=np.float32)
            else:
                s = palm
            lv = rel / np.maximum(s, eps)[None, :, None]
            keep = ok[:, None] & pv[:, nodes]                    # (T, 10)
            loc[:, :, nodes, m] = np.where(keep[None], lv, 0.0)
    return loc


def clip_bone_lengths(data, eps=1e-6):
    """(N, C, T, V, M) -> (N, B) per-clip median of per-frame bone length / shoulder width (nan if absent)."""
    xy = np.asarray(data[:, :2, :, :, 0], dtype=np.float32)      # (N, 2, T, V)
    pv = (xy[:, 0] != 0) | (xy[:, 1] != 0)                      # (N, T, V)
    mid = 0.5 * (xy[..., R_SH] + xy[..., L_SH])                  # (N, 2, T)
    sw = np.linalg.norm(xy[..., R_SH] - xy[..., L_SH], axis=1)   # (N, T)
    sh_ok = pv[..., R_SH] & pv[..., L_SH] & (sw > eps)
    out = np.full((xy.shape[0], len(RETARGET_BONES)), np.nan, dtype=np.float64)
    for b, (p, c) in enumerate(RETARGET_BONES):
        pp = mid if p == ROOT else xy[..., p]
        pok = sh_ok if p == ROOT else pv[..., p]
        L = np.linalg.norm(xy[..., c] - pp, axis=1) / np.maximum(sw, eps)
        L = np.where(pok & pv[..., c] & sh_ok, L, np.nan)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                out[:, b] = np.nanmedian(L, axis=1)
    return out


def retarget_bones(data, scales, eps=1e-6):
    """Scale each retargeted bone vector by scales[b] along KIN_TREE; offsets propagate to children.

    data (C, T, V, M) image coords; scales (len(RETARGET_BONES),). Missing points stay 0; frames without
    shoulders are left unchanged."""
    out = np.array(data, dtype=np.float32, copy=True)
    sc = dict(zip(RETARGET_BONES, scales))
    for m in range(out.shape[3]):
        xy = data[:2, :, :, m].astype(np.float32)
        pv = _valid(xy)
        sh_ok = pv[:, R_SH] & pv[:, L_SH]
        mid = 0.5 * (xy[:, :, R_SH] + xy[:, :, L_SH])
        off = {ROOT: np.zeros_like(mid)}
        for p, c, r in KIN_TREE:
            pp = mid if p == ROOT else xy[:, :, p]
            pok = sh_ok if p == ROOT else pv[:, p]
            both = pok & pv[:, c] & sh_ok
            s = sc[(p, c)] if r else 1.0
            o = off[p] + (s - 1.0) * (xy[:, :, c] - pp)
            off[c] = np.where(both[None], o, off[p])
        new = xy.copy()
        for v in range(xy.shape[2]):
            new[:, :, v] = np.where(pv[None, :, v] & sh_ok[None], xy[:, :, v] + off[v], xy[:, :, v])
        out[:2, :, :, m] = np.where(pv[None], new, 0.0)
    return out


def mirror(data, mirror_x):
    """flip_index L/R swap + x mirror (Feeder's exact transform)."""
    d = np.array(data[:, :, flip_index, :], copy=True)
    d[0] = mirror_x(d[0])
    return d


class SignerNormFeeder(Feeder):
    def __init__(self, data_path, label_path, body_frame=False, body_scale='frame',
                 fallback_shoulder_width=0.253, hand_local=False, hand_scale='clip',
                 bone_aug=None, bone_aug_p=0.5, bone_jitter=0.1, bone_stats_path=None,
                 oracle_hand_canon=False, **kwargs):
        if kwargs.get('lap_pe'):
            raise ValueError('SignerNormFeeder does not support lap_pe')
        if kwargs.get('is_vector'):
            raise ValueError('SignerNormFeeder expects joint coordinates (is_vector=False)')
        if body_frame and kwargs.get('normalization'):
            raise ValueError('body_frame replaces normalization; set normalization: False')
        if body_scale not in ('frame', 'clip', 'none') or hand_scale not in ('frame', 'clip'):
            raise ValueError('body_scale/hand_scale must be frame|clip')
        if bone_aug not in (None, 'none', 'signer', 'jitter'):
            raise ValueError(f'bone_aug={bone_aug!r}')
        if body_frame and kwargs.get('random_shift') and kwargs.get('random_shift_limit') is None:
            # Feeder's 0.05 (image units) expressed in shoulder widths (median width 0.253)
            kwargs['random_shift_limit'] = 0.05 if body_scale == 'none' else 0.05 / fallback_shoulder_width
        super().__init__(data_path, label_path, **kwargs)
        self.body_frame = body_frame
        self.body_scale = body_scale
        self.fallback_shoulder_width = fallback_shoulder_width
        self.hand_local = hand_local
        self.hand_scale = hand_scale
        self.bone_aug = None if bone_aug in (None, 'none') else bone_aug
        self.bone_aug_p = bone_aug_p
        self.bone_jitter = bone_jitter
        if self.bone_aug == 'signer':
            self._init_signer_bones(bone_stats_path)
        # ORACLE (uses filename metadata, NOT deployable): mirror clips tagged '_L' (left-hand recording)
        # to the right-hand convention at train AND test -> upper bound for handedness canonicalization.
        self.oracle_hand_canon = oracle_hand_canon
        if oracle_hand_canon:
            tags = [re.search(r'_([LR])(?:\.mp4)?$', str(n)) for n in self.sample_name]
            if any(t is None for t in tags):
                raise ValueError('oracle_hand_canon needs _L/_R sample-name tags')
            self.is_left = np.array([t.group(1) == 'L' for t in tags])
            print(f'[SignerNormFeeder] ORACLE handedness canonicalization: {int(self.is_left.sum())}/'
                  f'{len(self.is_left)} _L clips mirrored')

    # ---- (d) per-signer mean bone lengths of THIS (training) split ----
    def _init_signer_bones(self, stats_path=None):
        signers = np.array([_signer_of(n) for n in self.sample_name])
        if (signers < 0).any():
            raise ValueError('bone_aug=signer needs U<id> sample names')
        N = len(signers)
        L = np.concatenate([clip_bone_lengths(np.asarray(self.data[i:i + 256]))
                            for i in range(0, N, 256)], 0)
        self.signer_ids = sorted(set(signers.tolist()))
        if len(self.signer_ids) < 2:
            raise ValueError('bone_aug=signer needs >= 2 training signers')
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.signer_bone = {s: np.nanmean(L[signers == s], 0) for s in self.signer_ids}
        glob_mean = np.nanmean(L, 0)
        for s in self.signer_ids:                     # a bone never seen for a signer -> global mean
            bad = ~np.isfinite(self.signer_bone[s])
            self.signer_bone[s][bad] = glob_mean[bad]
        self.clip_signer = signers
        ratios = np.stack([self.signer_bone[s] / glob_mean for s in self.signer_ids])
        print('[SignerNormFeeder] bone retarget: {} signers, {} bones; signer/global ratio range '
              '[{:.3f}, {:.3f}], per-bone max spread {:.3f}'.format(
                  len(self.signer_ids), len(RETARGET_BONES), np.nanmin(ratios), np.nanmax(ratios),
                  np.nanmax(np.nanmax(ratios, 0) - np.nanmin(ratios, 0))))
        if stats_path:
            import json
            json.dump({'bones': [list(map(int, b)) for b in RETARGET_BONES],
                       'signer_bone': {str(s): self.signer_bone[s].tolist() for s in self.signer_ids}},
                      open(stats_path, 'w'), indent=1)

    def _bone_scales(self, index):
        if self.bone_aug == 'jitter':
            return np.array([random.uniform(1 - self.bone_jitter, 1 + self.bone_jitter)
                             for _ in RETARGET_BONES], dtype=np.float32)
        src = int(self.clip_signer[index])
        tgt = random.choice([s for s in self.signer_ids if s != src])
        return (self.signer_bone[tgt] / self.signer_bone[src]).astype(np.float32)

    def _shift_limit(self, data_numpy):
        if self.body_frame:
            if self.random_shift_limit is not None:
                return self.random_shift_limit
            return 0.05 if self.body_scale == 'none' else 0.05 / self.fallback_shoulder_width
        return super()._shift_limit(data_numpy)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.center_choose(data_numpy, self.window_size)

        if self.bone_aug and random.random() < self.bone_aug_p:          # (d)
            data_numpy = retarget_bones(data_numpy, self._bone_scales(index))

        if self.oracle_hand_canon and self.is_left[index]:                # oracle (metadata)
            data_numpy = data_numpy[:, :, flip_index, :]
            data_numpy[0, :, :, :] = self._mirror_x(data_numpy[0, :, :, :])

        if self.random_mirror:                                            # (c), as Feeder
            if random.random() > self.random_mirror_p:
                assert data_numpy.shape[2] == 27
                data_numpy = data_numpy[:, :, flip_index, :]
                data_numpy[0, :, :, :] = self._mirror_x(data_numpy[0, :, :, :])

        # (b) from image coords (translation/scale invariant; before any centering, which would make
        # missing points non-zero)
        loc = hand_local_channels(data_numpy, self.hand_scale) if self.hand_local else None

        if self.body_frame:                                               # (a)
            data_numpy = body_frame_normalize(data_numpy, self.body_scale,
                                              self.fallback_shoulder_width)
        elif self.normalization:                                          # Feeder's centering
            data_numpy[0, :, :, :] = data_numpy[0, :, :, :] - data_numpy[0, :, 0, 0].mean(axis=0)
            data_numpy[1, :, :, :] = data_numpy[1, :, :, :] - data_numpy[1, :, 0, 0].mean(axis=0)

        if self.random_shift:
            limit = self._shift_limit(data_numpy)
            x_mask = data_numpy[0, :, :, :] != 0
            y_mask = data_numpy[1, :, :, :] != 0
            data_numpy[0, :, :, :][x_mask] += random.random() * 2 * limit - limit
            data_numpy[1, :, :, :][y_mask] += random.random() * 2 * limit - limit

        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        if loc is not None:                                               # (b)
            data_numpy = np.concatenate([data_numpy, loc], 0)

        if self._drop_idx:
            data_numpy[:, :, self._drop_idx, :] = 0

        data_numpy = np.ascontiguousarray(data_numpy)
        if self.return_signer:
            return data_numpy, label, self.signer[index], index
        return data_numpy, label, index
