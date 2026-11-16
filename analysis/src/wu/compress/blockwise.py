# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Walk a decoder one block at a time over calibration data, as GPTQ, AWQ, WANDA and SparseGPT do.

The calibration tokens are embedded once. Then, for each block: `fn(index, block, inputs)` may change
the block (quantize or prune it) using the hidden states that enter it, and the *changed* block's
outputs become the next block's inputs. So each block is compressed knowing what the compressed blocks
before it produce ("sequential" in GPTQ's terms), and only one block's statistics are alive at a time:
needed for an 8B model under 48 GB.
"""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class _Captured(Exception):
    pass


def embed(model, tokens: list[list[int]], batch: int = 8) -> mx.array:
    """The hidden states that enter block 0, however the architecture makes them (Gemma, for one, scales
    its embeddings by sqrt(hidden)): run the model with block 0 replaced by a catcher, and stop there."""
    layers, first, caught = model.layers, model.layers[0], []

    class Catcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_sliding = getattr(first, "use_sliding", False)  # MLX-LM's Llama reads it from every layer

        def __call__(self, x, *args, **kwargs):
            caught.append(x)
            raise _Captured

    layers[0] = Catcher()
    try:
        for i in range(0, len(tokens), batch):
            try:
                model(mx.array(tokens[i : i + batch]))
            except _Captured:
                pass
    finally:
        layers[0] = first
    h = mx.concatenate(caught, axis=0)
    mx.eval(h)
    return h


def run_block(block, x: mx.array, batch: int = 8) -> mx.array:
    """The block on calibration windows. "causal" is exact for every block here, including sliding-window
    ones, as long as the window (Gemma: 1,024 or more) is longer than the 512-token calibration windows."""
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(block(x[i : i + batch], "causal", None))
        mx.eval(out[-1])
    return mx.concatenate(out, axis=0)


def linears(block) -> list[tuple[str, nn.Linear]]:
    return [
        (k, m) for k, m in tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module) if isinstance(m, nn.Linear)
    ]


class Recorder(nn.Module):
    """Wraps a Linear and accumulates statistics of its inputs: X^T X (`hessian`) and/or the
    squared norm of each input feature (`sq_norm`, for WANDA). Counts tokens in `n`."""

    def __init__(self, module: nn.Linear, hessian: bool, sq_norm: bool):
        super().__init__()
        self.module, self.want_h, self.want_n = module, hessian, sq_norm
        d = module.weight.shape[1]
        self.H = mx.zeros((d, d), dtype=mx.float32) if hessian else None
        self.sq = mx.zeros((d,), dtype=mx.float32) if sq_norm else None
        self.n = 0

    def __call__(self, x):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        for c in range(
            0, xf.shape[0], 1024
        ):  # small matmuls: one 4,096 x 12,288 product per call can trip the GPU watchdog
            part = xf[c : c + 1024]
            if self.want_h:
                self.H = self.H + part.T @ part
            if self.want_n:
                self.sq = self.sq + (part * part).sum(axis=0)
            mx.eval(*[a for a in (self.H, self.sq) if a is not None])
        self.n += xf.shape[0]
        return self.module(x)


def record(block, x: mx.array, *, hessian: bool = False, sq_norm: bool = False) -> dict[str, Recorder]:
    """Input statistics for every Linear in the block, from one pass over x (the block is left unchanged)."""
    originals = linears(block)
    recorders = {k: Recorder(m, hessian, sq_norm) for k, m in originals}
    block.update_modules(_nest(list(recorders.items())))
    run_block(block, x)
    block.update_modules(_nest(originals))
    return recorders


def _nest(pairs):
    from mlx.utils import tree_unflatten

    return tree_unflatten(pairs)


def walk(
    model,
    tokens: list[list[int]],
    fn: Callable[[int, nn.Module, mx.array], None],
    progress=print,
    only: set[int] | None = None,
):
    """`only`: apply fn to these blocks alone; the others stay as they are and pass their outputs on."""
    x = embed(model, tokens)
    layers = model.layers
    for i, block in enumerate(layers):
        if only is not None and i > max(only):
            break
        if only is None or i in only:
            fn(i, block, x)
        x = run_block(block, x)
        mx.clear_cache()
        if progress and (i % 4 == 3 or i == len(layers) - 1):
            progress(f"  block {i + 1}/{len(layers)}")
