"""Unit tests for feeders/feeder_signernorm.py (signer-normalized pose recipe).

Real-clip tests use a 64-clip subset of data/bdsl_si/train_data.npy (skipped if the bundle is absent).
"""
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from feeders import feeder_signernorm as fs
from feeders.feeder import Feeder, flip_index

TRAIN = ROOT / "data" / "bdsl_si" / "train_data.npy"
TRAIN_LABEL = ROOT / "data" / "bdsl_si" / "train_label.pkl"


@pytest.fixture(scope="module")
def subset(tmp_path_factory):
    if not TRAIN.exists():
        pytest.skip("data/bdsl_si not available")
    d = np.load(TRAIN, mmap_mode="r")
    names, labels = pickle.load(open(TRAIN_LABEL, "rb"))
    idx = np.linspace(0, len(names) - 1, 64).astype(int)
    tmp = tmp_path_factory.mktemp("sn")
    np.save(tmp / "d.npy", np.asarray(d[idx]))
    pickle.dump(([names[i] for i in idx], [int(labels[i]) for i in idx]), open(tmp / "l.pkl", "wb"))
    assert len({fs._signer_of(names[i]) for i in idx}) >= 2
    return str(tmp / "d.npy"), str(tmp / "l.pkl")


def _clip(subset, i=0):
    return np.array(np.load(subset[0])[i])


BASE = dict(window_size=120, random_choose=True, random_shift=True, normalization=True,
            random_mirror=True, random_mirror_p=0.5, is_vector=False, lap_pe=False, use_mmap=False)


def test_all_off_matches_base_feeder(subset):
    a = Feeder(*subset, **BASE)
    b = fs.SignerNormFeeder(*subset, **BASE)
    for i in range(len(a)):
        random.seed(i); xa, ya, _ = a[i]
        random.seed(i); xb, yb, _ = b[i]
        assert ya == yb
        np.testing.assert_array_equal(xa, xb)


@pytest.mark.parametrize("opts,C", [
    (dict(body_frame=True), 3),
    (dict(body_frame=True, hand_local=True), 5),
    (dict(body_frame=True, hand_local=True, random_mirror=True, random_mirror_p=0.5), 5),
    (dict(body_frame=True, hand_local=True, random_mirror=True, random_mirror_p=0.5,
          bone_aug="signer", bone_aug_p=1.0), 5),
    (dict(body_frame=True, hand_local=True, bone_aug="jitter", bone_aug_p=1.0, bone_jitter=0.2), 5),
    (dict(body_frame=True, body_scale="clip", hand_local=True, hand_scale="frame"), 5),
    (dict(body_frame=True, body_scale="none"), 3),
])
def test_shapes_finite_and_missing_stay_zero(subset, opts, C):
    kw = dict(BASE, normalization=False, random_mirror=False)
    kw.update(opts)
    f = fs.SignerNormFeeder(*subset, **kw)
    raw = np.load(subset[0])
    for i in range(len(f)):
        random.seed(100 + i)
        x, y, idx = f[i]
        assert x.shape == (C, 120, 27, 1) and np.issubdtype(x.dtype, np.floating)
        assert np.isfinite(x).all()
        # padded frames stay all-zero; the number of non-empty frames is preserved
        nvalid = int((np.abs(raw[i, :2]).sum((0, 2, 3)) > 0).sum())
        nz = int((np.abs(x[:2]).sum((0, 2, 3)) > 0).sum())
        assert nz == min(nvalid, 120)
    # body-frame coords are O(shoulder widths)
    assert np.abs(x[:2]).max() < 20


def test_mirror_twice_is_identity(subset):
    f = fs.SignerNormFeeder(*subset, **dict(BASE, random_mirror=True))
    for i in range(8):
        d = _clip(subset, i)
        np.testing.assert_allclose(fs.mirror(fs.mirror(d, f._mirror_x), f._mirror_x), d, atol=1e-6)
    assert (flip_index[flip_index] == np.arange(27)).all()


def test_flip_index_matches_graph_and_data(subset):
    """flip_index swaps L/R body pairs and the two hands; graph edges and data agree on sidedness."""
    from graph.sign_27 import Graph
    assert list(flip_index[:7]) == [0, 2, 1, 4, 3, 6, 5]
    assert list(flip_index[7:17]) == list(range(17, 27)) and list(flip_index[17:]) == list(range(7, 17))
    edges = set(Graph().inward)
    assert (0, 2) in edges and (1, 3) in edges and (3, 5) in edges and (5, 7) in edges and (6, 17) in edges
    # every graph edge maps to an edge under the flip (mirror is a graph automorphism)
    assert all((int(flip_index[i]), int(flip_index[j])) in edges for i, j in edges)
    d = np.load(subset[0])
    xy = d[:, :2, :, :, 0]

    def dist(a, b):
        m = (xy[:, 0, :, a] != 0) & (xy[:, 0, :, b] != 0)
        return np.linalg.norm(xy[:, :, :, a] - xy[:, :, :, b], axis=1)[m].mean()
    # pose wrist 5 sits on hand 7-16, pose wrist 6 on hand 17-26
    assert dist(5, 7) < 0.3 * dist(5, 17) and dist(6, 17) < 0.3 * dist(6, 7)


@pytest.mark.parametrize("scale", ["frame", "clip"])
def test_body_frame_invariant_to_global_scale_translation(subset, scale):
    for i in range(8):
        d = _clip(subset, i)
        v = fs._valid(d[:2])
        g = d.copy()
        g[0] = np.where(v, 1.7 * d[0] + 0.31, 0)
        g[1] = np.where(v, 1.7 * d[1] - 0.12, 0)
        a = fs.body_frame_normalize(d, scale)
        b = fs.body_frame_normalize(g, scale)
        np.testing.assert_allclose(a, b, atol=1e-4)
        np.testing.assert_allclose(fs.hand_local_channels(d), fs.hand_local_channels(g), atol=1e-4)
        # shoulders land at (-0.5, 0) / (0.5, 0) along the shoulder axis: width 1
        fr = v[:, 1] & v[:, 2]
        if scale == "frame" and fr.any():
            w = np.linalg.norm(a[:2, :, 1, 0] - a[:2, :, 2, 0], axis=0)[fr[:, 0]]
            np.testing.assert_allclose(w, 1.0, atol=1e-4)
            np.testing.assert_allclose((a[:2, :, 1, 0] + a[:2, :, 2, 0])[:, fr[:, 0]], 0, atol=1e-5)


def test_missing_shoulder_frames_borrow_nearest():
    d = np.zeros((3, 6, 27, 1), np.float32)
    rng = np.random.RandomState(0)
    d[:2, :5] = rng.uniform(0.2, 0.8, (2, 5, 27, 1))
    d[:2, 2, [1, 2]] = 0                       # frame 2 lost its shoulders
    out = fs.body_frame_normalize(d)
    assert np.isfinite(out).all() and (out[:, 5] == 0).all()      # padded frame stays 0
    ref = fs.body_frame_normalize(np.stack([d[:, 1]] * 1, 1))      # frame 1 alone
    c = 0.5 * (d[:2, 1, 1] + d[:2, 1, 2]); w = np.linalg.norm(d[:2, 1, 1] - d[:2, 1, 2], axis=0)
    np.testing.assert_allclose(out[:2, 2, 0], (d[:2, 2, 0] - c) / w, atol=1e-5)
    assert ref.shape == (3, 1, 27, 1)
    # no shoulders anywhere -> centroid fallback, finite
    e = d.copy(); e[:2, :, [1, 2]] = 0
    assert np.isfinite(fs.body_frame_normalize(e)).all()


def test_retarget_identity_and_lengths(subset):
    nb = len(fs.RETARGET_BONES)
    d = _clip(subset, 3)
    np.testing.assert_allclose(fs.retarget_bones(d, np.ones(nb)), d, atol=1e-6)
    s = np.random.RandomState(1).uniform(0.7, 1.3, nb)
    r = fs.retarget_bones(d, s)
    v = fs._valid(d[:2])[..., 0]
    assert (fs._valid(r[:2])[..., 0] == v).all()
    sh = v[:, 1] & v[:, 2]
    for b, (p, c) in enumerate(fs.RETARGET_BONES):
        if p == fs.ROOT:
            P0 = 0.5 * (d[:2, :, 1, 0] + d[:2, :, 2, 0]); P1 = 0.5 * (r[:2, :, 1, 0] + r[:2, :, 2, 0]); ok = sh
        else:
            P0, P1, ok = d[:2, :, p, 0], r[:2, :, p, 0], v[:, p] & sh
        ok = ok & v[:, c]
        if not ok.any():
            continue
        l0 = np.linalg.norm(d[:2, :, c, 0] - P0, axis=0)[ok]
        l1 = np.linalg.norm(r[:2, :, c, 0] - P1, axis=0)[ok]
        np.testing.assert_allclose(l1, s[b] * l0, rtol=1e-4, atol=1e-6)
    # shoulders never move; channel 2 untouched
    np.testing.assert_array_equal(r[:, :, [1, 2]], d[:, :, [1, 2]])
    np.testing.assert_array_equal(r[2], d[2])


def test_signer_bone_stats(subset):
    f = fs.SignerNormFeeder(*subset, **dict(BASE, normalization=False, body_frame=True,
                                            bone_aug="signer", bone_aug_p=1.0))
    assert len(f.signer_ids) >= 2
    for s in f.signer_ids:
        assert np.isfinite(f.signer_bone[s]).all() and (f.signer_bone[s] > 0).all()
    for i in range(5):
        sc = f._bone_scales(i)
        assert sc.shape == (len(fs.RETARGET_BONES),) and np.isfinite(sc).all()


def test_blockgcn_accepts_five_channels(subset):
    from model.block_gcn import Model
    f = fs.SignerNormFeeder(*subset, **dict(BASE, normalization=False, random_mirror=False,
                                            body_frame=True, hand_local=True))
    x = torch.from_numpy(np.stack([f[i][0] for i in range(4)]))
    m = Model(num_class=60, num_point=27, num_person=1, graph="graph.sign_27.Graph",
              graph_args={"labeling_mode": "spatial"}, in_channels=5,
              num_blocks=[1, 1, 1, 1], stage_channels=[16, 16, 16, 16], max_distance=8).eval()
    with torch.no_grad():
        out = m(x)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (4, 60) and torch.isfinite(out).all()


def test_centring_only_control(subset):
    d = _clip(subset, 2)
    v = fs._valid(d[:2])
    a = fs.body_frame_normalize(d, "none")
    g = d.copy(); g[0] = np.where(v, d[0] + 0.3, 0); g[1] = np.where(v, d[1] - 0.1, 0)
    np.testing.assert_allclose(a, fs.body_frame_normalize(g, "none"), atol=1e-5)   # translation-invariant
    fr = v[:, 1, 0] & v[:, 2, 0]
    sw = np.linalg.norm(d[:2, :, 1, 0] - d[:2, :, 2, 0], axis=0)[fr]
    np.testing.assert_allclose(np.linalg.norm(a[:2, :, 1, 0] - a[:2, :, 2, 0], axis=0)[fr], sw, atol=1e-6)  # unscaled
    assert (a[:2][~np.broadcast_to(v[None], a[:2].shape)] == 0).all()


def test_oracle_hand_canon(subset):
    names, _ = pickle.load(open(subset[1], "rb"))
    f = fs.SignerNormFeeder(*subset, **dict(BASE, normalization=False, random_mirror=False, random_shift=False,
                                            body_frame=True, oracle_hand_canon=True))
    g = fs.SignerNormFeeder(*subset, **dict(BASE, normalization=False, random_mirror=False, random_shift=False,
                                            body_frame=True))
    assert f.is_left.tolist() == [n.endswith("_L.mp4") for n in names]
    for i in range(len(f)):
        random.seed(i); a = f[i][0]
        random.seed(i); b = g[i][0]
        if f.is_left[i]:
            # mirrored in image space == swap nodes + negate x in the (symmetric) shoulder frame
            np.testing.assert_allclose(a[0], -b[0][:, flip_index], atol=1e-4)
            np.testing.assert_allclose(a[1], b[1][:, flip_index], atol=1e-4)
        else:
            np.testing.assert_array_equal(a, b)
