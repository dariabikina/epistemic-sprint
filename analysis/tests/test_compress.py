# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

import mlx.core as mx
import numpy as np
import pytest

from wu.compress.prune import parse_level, prune_by_score, shortgpt_drop
from wu.compress.quant import pack


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_matches_mlx_layout(bits):
    w = mx.random.normal((8, 256), key=mx.random.key(0))
    packed, s, b = mx.quantize(w, bits=bits, group_size=128)
    q = mx.clip(mx.round((w.reshape(8, 2, 128) - b[..., None]) / s[..., None]), 0, 2**bits - 1).reshape(8, 256)
    assert np.array_equal(np.array(packed), pack(np.array(q).astype(np.uint32), bits))


def test_unstructured_sparsity_per_row():
    W = mx.random.normal((16, 64), key=mx.random.key(1))
    P = prune_by_score(W, mx.abs(W), 0.5, None, per_row=True)
    assert np.all((np.array(P) == 0).sum(axis=1) == 32)
    kept = np.abs(np.array(W))[np.array(P) != 0].min()  # survivors are the larger weights of their row
    assert kept > 0


@pytest.mark.parametrize("n,m", [(2, 4), (4, 8)])
def test_n_of_m(n, m):
    W = mx.random.normal((8, 64), key=mx.random.key(2))
    P = np.array(prune_by_score(W, mx.abs(W), None, (n, m), per_row=True)).reshape(8, 64 // m, m)
    assert np.all((P != 0).sum(axis=-1) == n)


def test_layerwise_magnitude_fraction():
    W = mx.random.normal((32, 32), key=mx.random.key(3))
    P = np.array(prune_by_score(W, mx.abs(W), 0.3, None, per_row=False))
    assert (P == 0).sum() == int(0.3 * 1024)


def test_levels_and_drop():
    assert parse_level("50") == (0.5, None) and parse_level("2:4") == (None, (2, 4))
    assert shortgpt_drop([0.5, 0.1, 0.3, 0.2], 50) == [1, 2]  # block 3 is last: protected
    assert shortgpt_drop([0.0, 0.9, 0.1, 0.8, 0.0], 40) == [2, 3]


@pytest.mark.parametrize("nm", [(2, 4), (4, 8)])
def test_n_of_m_keeps_the_highest_scores(nm):
    keep, m = nm
    W = mx.random.normal((8, 64), key=mx.random.key(4))
    score = mx.abs(W)
    P = np.array(prune_by_score(W, score, None, nm, per_row=True)).reshape(8, -1, m)
    s = np.abs(np.array(W)).reshape(8, -1, m)
    kept_min = np.where(P != 0, s, np.inf).min(-1)
    dropped_max = np.where(P == 0, s, -np.inf).max(-1)
    assert np.all(kept_min >= dropped_max)


def test_per_row_keeps_the_highest_scores():
    W = mx.random.normal((8, 64), key=mx.random.key(5))
    P = np.array(prune_by_score(W, mx.abs(W), 0.5, None, per_row=True))
    s = np.abs(np.array(W))
    assert np.all(np.where(P != 0, s, np.inf).min(1) >= np.where(P == 0, s, -np.inf).max(1))
