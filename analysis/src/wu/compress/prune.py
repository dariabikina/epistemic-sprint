# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Pruning, as in Wu Table 2: attention and MLP linears only (never lm_head or embeddings); no recovery fine-tuning.

    magnitude-<s>   zero the s% smallest |W| of each linear. No data.
    wanda-<s>       zero the s% smallest |W_ij| * ||X_j|| within each output row (Sun et al. 2024); X are the
                    layer's inputs on 32 C4 windows of 512 tokens, collected block by block on the already
                    pruned model (as the WANDA code does).
    wanda-<n>:<m>   the same score, keeping n of every m consecutive weights in a row.
    shortgpt-<p>    remove the p% of decoder blocks with the lowest block influence
                    BI = 1 - mean cos(block input, block output) (Men et al. 2024), on the same C4 windows.

A pruned model is saved as a *recipe* (input norms or the dropped block list: kilobytes), and applied to the
base weights at load time. The masks are deterministic given the recipe, so this is the same model.
"""

import json
from pathlib import Path

import mlx.core as mx

from . import blockwise

SAMPLES, LENGTH, SEED = 32, 512, 42


def parse_level(level: str) -> tuple[float | None, tuple[int, int] | None]:
    """'50' -> (0.5, None); '2:4' -> (None, (2, 4))."""
    if ":" in level:
        n, m = map(int, level.split(":"))
        return None, (n, m)
    return int(level) / 100, None


def prune_by_score(
    W: mx.array, score: mx.array, sparsity: float | None, nm: tuple[int, int] | None, per_row: bool
) -> mx.array:
    """W with the lowest-scoring weights set to zero."""
    rows, cols = W.shape
    if nm:
        keep, m = nm
        grouped = score.reshape(rows, cols // m, m)
        drop = mx.argpartition(grouped, kth=m - keep - 1, axis=-1)[..., : m - keep]
        mask = mx.put_along_axis(mx.ones(grouped.shape, dtype=mx.bool_), drop, mx.array(False), axis=-1)
        return W * mask.reshape(rows, cols)
    if per_row:
        k = int(cols * sparsity)
        drop = mx.argpartition(score, kth=k - 1, axis=1)[:, :k]
        mask = mx.put_along_axis(mx.ones(score.shape, dtype=mx.bool_), drop, mx.array(False), axis=1)
        return W * mask
    k = int(W.size * sparsity)
    flat = score.reshape(-1)
    drop = mx.argpartition(flat, kth=k - 1)[:k]
    mask = mx.ones(flat.shape, dtype=mx.bool_)
    mask[drop] = False
    return W * mask.reshape(rows, cols)


def _linear_pairs(model):
    for i, block in enumerate(model.layers):
        for key, lin in blockwise.linears(block):
            yield f"layers.{i}.{key}", lin


def magnitude(model, level: str):
    sparsity, nm = parse_level(level)
    for _, lin in _linear_pairs(model):
        lin.weight = prune_by_score(lin.weight, mx.abs(lin.weight).astype(mx.float32), sparsity, nm, per_row=False)
        mx.eval(lin.weight)
    return model


def wanda_recipe(model, tokenizer, level: str, progress=print, calibration=None, blocks=None) -> dict[str, mx.array]:
    """Prunes the model in place while recording each linear's input norms; returns the norms (the recipe)."""
    from ..data import c4_calibration

    sparsity, nm = parse_level(level)
    norms = {}

    def prune_block(index, block, x):
        for key, r in blockwise.record(block, x, sq_norm=True).items():
            norm = mx.sqrt(r.sq)
            norms[f"layers.{index}.{key}"] = norm
            lin = r.module
            score = mx.abs(lin.weight).astype(mx.float32) * norm[None]
            lin.weight = prune_by_score(lin.weight, score, sparsity, nm, per_row=True)
            mx.eval(lin.weight)

    tokens = calibration if calibration is not None else c4_calibration(tokenizer, SAMPLES, LENGTH, SEED)
    blockwise.walk(model, tokens, prune_block, progress, only=blocks)
    return norms


def wanda_apply(model, level: str, norms: dict[str, mx.array]):
    sparsity, nm = parse_level(level)
    for name, lin in _linear_pairs(model):
        score = mx.abs(lin.weight).astype(mx.float32) * norms[name][None]
        lin.weight = prune_by_score(lin.weight, score, sparsity, nm, per_row=True)
        mx.eval(lin.weight)
    return model


def block_influence(model, tokenizer) -> list[float]:
    """BI per decoder block on the unpruned model."""
    from ..data import c4_calibration

    influence = []

    def measure(index, block, x):
        y = blockwise.run_block(block, x)
        cos = (x * y).sum(-1) / (
            mx.linalg.norm(x.astype(mx.float32), axis=-1) * mx.linalg.norm(y.astype(mx.float32), axis=-1) + 1e-6
        )
        influence.append(1 - cos.astype(mx.float32).mean().item())

    blockwise.walk(model, c4_calibration(tokenizer, SAMPLES, LENGTH, SEED), measure, progress=None)
    return influence


def shortgpt_drop(influence: list[float], percent: int) -> list[int]:
    """The round(percent% of all blocks) lowest-influence blocks, never the first or the last (Wu Table 2)."""
    n = round(len(influence) * percent / 100)
    interior = range(1, len(influence) - 1)
    return sorted(sorted(interior, key=lambda i: influence[i])[:n])


def shortgpt_apply(model, dropped: list[int]):
    kept = [b for i, b in enumerate(model.layers) if i not in set(dropped)]
    model.layers[:] = kept  # in place: model.layers is the inner model's list
    return model


def save_recipe(directory: Path, recipe: dict):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in recipe.items() if isinstance(v, mx.array)}
    if arrays:
        mx.save_safetensors(str(directory / "recipe.safetensors"), arrays)
    rest = {k: v for k, v in recipe.items() if not isinstance(v, mx.array)}
    (directory / "recipe.json").write_text(json.dumps(rest, indent=1) + "\n")


def load_recipe(directory: Path) -> dict:
    recipe = json.loads((directory / "recipe.json").read_text())
    if (directory / "recipe.safetensors").exists():
        recipe.update(mx.load(str(directory / "recipe.safetensors")))
    return recipe
