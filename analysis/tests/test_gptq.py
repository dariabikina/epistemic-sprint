# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from wu.compress.quant import GROUP_SIZE, _inverse_hessian_factor, gptq_linear, pack


def reference_gptq_linear(linear: nn.Linear, H: mx.array, bits: int) -> nn.QuantizedLinear:
    """GPTQ on one layer: quantize columns left to right, pushing each column's error onto the columns
    not yet quantized, weighted by the inverse Hessian of the layer's inputs."""
    dtype = linear.weight.dtype
    W = linear.weight.astype(mx.float32)
    dead = mx.diag(H) == 0  # inputs that never fire: their weights don't matter
    H = H + mx.diag(dead.astype(mx.float32))
    W = mx.where(dead[None], 0.0, W)
    Hinv = _inverse_hessian_factor(H)
    top = 2**bits - 1
    scales, biases, levels = [], [], []
    for i in range(0, W.shape[1], GROUP_SIZE):
        j = i + GROUP_SIZE
        _, s, b = mx.quantize(W[:, i:j], bits=bits, group_size=GROUP_SIZE)  # this group's grid, after earlier updates
        err = mx.zeros((W.shape[0], GROUP_SIZE), dtype=mx.float32)
        q_group = []
        for k in range(i, j):
            w = W[:, k : k + 1]
            q = mx.clip(mx.round((w - b) / s), 0, top)
            e = (w - (q * s + b)) / Hinv[k, k]
            W[:, k:j] = W[:, k:j] - e @ Hinv[k : k + 1, k:j]
            err[:, k - i : k - i + 1] = e
            q_group.append(q)
            mx.eval(W, err, q)
        W[:, j:] = W[:, j:] - err @ Hinv[i:j, j:]
        mx.eval(W)
        scales.append(s)
        biases.append(b)
        levels.append(mx.concatenate(q_group, axis=1))
    layer = linear.to_quantized(group_size=GROUP_SIZE, bits=bits)
    layer.weight = mx.array(pack(np.array(mx.concatenate(levels, axis=1)).astype(np.uint32), bits))
    layer.scales = mx.concatenate(scales, axis=1).astype(dtype)
    layer.biases = mx.concatenate(biases, axis=1).astype(dtype)
    mx.eval(layer)
    return layer


def test_gptq_restructured_matches_reference():
    mx.random.seed(0)
    lin = nn.Linear(256, 32, bias=False)
    X = mx.random.normal((512, 256)) * (1 + 3 * (mx.arange(256) % 37 == 0))
    H = (X.T @ X) / 512
    a, b = gptq_linear(lin, H, 3), reference_gptq_linear(lin, H, 3)
    assert np.array_equal(np.array(a.weight), np.array(b.weight))
    assert np.allclose(np.array(a.scales), np.array(b.scales)) and np.allclose(np.array(a.biases), np.array(b.biases))


def test_hessian_recorder_chunking():
    from wu.compress.blockwise import Recorder

    lin = nn.Linear(64, 8, bias=False)
    x = mx.random.normal((3, 1500, 64))
    r = Recorder(lin, hessian=True, sq_norm=True)
    r(x)
    xf = x.reshape(-1, 64)
    assert np.allclose(np.array(r.H), np.array(xf.T @ xf), rtol=1e-4, atol=1e-2)
    assert r.n == 4500
