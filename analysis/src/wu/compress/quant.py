# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Weight-only quantization, as in Wu Table 1: group size 128, lm_head (and embeddings) kept in full precision,
calibration on 128 C4 windows of 512 tokens, seed 42.

    rtn   round-to-nearest affine quantization; no data. Not in Wu: the floor GPTQ and AWQ should beat.
    gptq  Frantar et al. 2023, block by block (see blockwise.py). Differences from Wu's setup: MLX's
          asymmetric (affine) format instead of symmetric, no activation ordering, and blocks rather
          than single linears processed in sequence. MLX-LM's own gptq.py would quantize lm_head, can't
          pack 3 bits, and keeps every layer's Hessian at once (~35 GB for 8B), hence this version.
    awq   Lin et al., using MLX-LM's scale and clip search on our own block loop, so that the
          embeddings and lm_head stay unquantized (MLX-LM's awq_quantize makes both 4-bit).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from . import blockwise

GROUP_SIZE, SAMPLES, LENGTH, SEED = 128, 128, 512, 42


def _target(path: str, module) -> bool:
    return isinstance(module, nn.Linear) and "lm_head" not in path


def rtn(model, bits: int):
    nn.quantize(model, group_size=GROUP_SIZE, bits=bits, class_predicate=_target)
    return model


def pack(q: np.ndarray, bits: int) -> np.ndarray:
    """Integer levels [rows, cols] -> MLX's packed uint32 layout (a little-endian bit stream per row).
    Verified equal to mx.quantize's packing for 2, 3, 4 and 8 bits (tests/test_compress.py)."""
    q = q.astype(np.uint64)
    rows, cols = q.shape
    bitplanes = ((q[:, :, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(bitplanes.reshape(rows, cols * bits), axis=1, bitorder="little").view(np.uint32)


def _inverse_hessian_factor(H: mx.array) -> mx.array:
    with mx.stream(mx.cpu):
        H = H + 1e-2 * mx.mean(mx.diag(H)) * mx.eye(H.shape[0])  # GPTQ's 1% dampening
        L = mx.linalg.cholesky(H)
        return mx.linalg.cholesky(mx.linalg.cholesky_inv(L), upper=True)


def gptq_linear(linear: nn.Linear, H: mx.array, bits: int) -> nn.QuantizedLinear:
    """GPTQ on one layer: quantize columns left to right, pushing each column's error onto the columns
    not yet quantized, weighted by the inverse Hessian of the layer's inputs.

    Works on one 128-column group at a time (a small copy), then updates the not-yet-quantized remainder once
    per group. Same arithmetic as updating the full matrix after every column, but each step touches a
    rows x 128 array instead of the whole matrix (which, on 8B's down_proj, is 200 MB per column)."""
    dtype = linear.weight.dtype
    W = linear.weight.astype(mx.float32)
    dead = mx.diag(H) == 0  # inputs that never fire: their weights don't matter
    H = H + mx.diag(dead.astype(mx.float32))
    W = mx.where(dead[None], 0.0, W)
    mx.eval(W, H)
    Hinv = _inverse_hessian_factor(H)  # upper triangular
    mx.eval(Hinv)  # finish the CPU-side Cholesky before GPU work is queued behind it: on 8B (12,288-wide) the
    # GPU otherwise waits on the CPU stream long enough to trip the Metal watchdog
    top, cols = 2**bits - 1, W.shape[1]
    scales, biases, levels = [], [], []
    rest = W  # the columns not quantized yet
    for i in range(0, cols, GROUP_SIZE):
        j = i + GROUP_SIZE
        group, Hg = rest[:, :GROUP_SIZE], Hinv[i:j, i:j]
        _, s, b = mx.quantize(group, bits=bits, group_size=GROUP_SIZE)  # this group's grid, after earlier updates
        q_cols, e_cols = [], []
        for k in range(GROUP_SIZE):
            w = group[:, k : k + 1]
            q = mx.clip(mx.round((w - b) / s), 0, top)
            e = (w - (q * s + b)) / Hg[k, k]
            group = group - e @ Hg[k : k + 1, :]  # Hg is upper triangular: only columns >= k change
            q_cols.append(q)
            e_cols.append(e)
            if k % 16 == 15:
                mx.eval(group)
        err = mx.concatenate(e_cols, axis=1)
        rest = rest[:, GROUP_SIZE:] - err @ Hinv[i:j, j:] if j < cols else None
        level = mx.concatenate(q_cols, axis=1)
        mx.eval(level, s, b, *([rest] if rest is not None else []))
        scales.append(s)
        biases.append(b)
        levels.append(level)
    layer = linear.to_quantized(group_size=GROUP_SIZE, bits=bits)
    layer.weight = mx.array(pack(np.array(mx.concatenate(levels, axis=1)).astype(np.uint32), bits))
    layer.scales = mx.concatenate(scales, axis=1).astype(dtype)
    layer.biases = mx.concatenate(biases, axis=1).astype(dtype)
    mx.eval(layer)
    return layer


def gptq(model, tokenizer, bits: int, progress=print, calibration=None, blocks=None):
    """`calibration`: token windows to use instead of C4 (the ground-truth worlds pass their own).
    `blocks`: quantize only these block indices (see blockwise.walk)."""
    from ..data import c4_calibration

    tokens = calibration if calibration is not None else c4_calibration(tokenizer, SAMPLES, LENGTH, SEED)

    def quantize_block(index, block, x):
        recorders = blockwise.record(block, x, hessian=True)
        new = [(k, gptq_linear(r.module, r.H / r.n, bits)) for k, r in recorders.items()]
        del recorders
        block.update_modules(tree_unflatten(new))

    blockwise.walk(model, tokens, quantize_block, progress, only=blocks)
    return model


def awq(model, tokenizer, bits: int, n_grid: int = 20, progress=print, calibration=None, blocks=None):
    from mlx.utils import tree_map
    from mlx_lm.quant.awq import AWQ_MODEL_CONFIGS, clip_block, mse, run_layer, scale_block

    from ..data import c4_calibration

    config = AWQ_MODEL_CONFIGS[model.model_type]
    tokens = calibration if calibration is not None else c4_calibration(tokenizer, SAMPLES, LENGTH, SEED)

    def quantize_func(w):
        return mx.dequantize(*mx.quantize(w, bits=bits, group_size=GROUP_SIZE), bits=bits, group_size=GROUP_SIZE)

    def capture(module):  # as MLX-LM's awq_quantize: keep each linear's inputs on the module
        if not isinstance(module, nn.Linear):
            return module

        class Catcher(nn.Module):
            def __call__(self, x, *args, **kwargs):
                module.input_feat = mx.concatenate([module.input_feat, x]) if hasattr(module, "input_feat") else x
                return module(x, *args, **kwargs)

        return Catcher()

    def block_error(block, x, reference):
        return mse(reference, run_layer(block, x, mask="causal")).mean().item()

    def quantize_block(index, block, x):
        originals, before = block.leaf_modules(), block.parameters()
        block.update_modules(tree_map(capture, originals, is_leaf=nn.Module.is_module))
        reference = run_layer(block, x, mask="causal")
        block.update_modules(originals)
        nn.quantize(block, group_size=GROUP_SIZE, bits=bits)
        rtn_error = block_error(block, x, reference)
        block.update_modules(originals)
        scale_block(
            block=block,
            configs=config.scale_configs,
            quantize_func=quantize_func,
            n_grid=n_grid,
            layer_kwargs={"mask": "causal"},
        )
        clip_block(
            block=block, no_clip_keys=config.no_clip, quantize_func=quantize_func, group_size=GROUP_SIZE, n_grid=n_grid
        )
        for _, m in blockwise.linears(block):
            if "input_feat" in m:
                del m["input_feat"]
        nn.quantize(block, group_size=GROUP_SIZE, bits=bits)
        awq_error = block_error(block, x, reference)
        if awq_error > rtn_error:  # as MLX-LM: keep plain RTN for a block that AWQ made worse
            block.update_modules(originals)
            block.update(before)
            nn.quantize(block, group_size=GROUP_SIZE, bits=bits)
            progress(f"  block {index}: AWQ error {awq_error:.3g} > RTN {rtn_error:.3g}; kept RTN")
        mx.eval(block.parameters())

    blockwise.walk(model, tokens, quantize_block, progress, only=blocks)
    return model
