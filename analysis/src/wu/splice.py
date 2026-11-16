# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Where in the network does compression destroy a fact? Splice compressed and uncompressed blocks.

`hybrid(model, setting, lo, hi)` is the uncompressed model with only blocks lo..hi-1 compressed (as `setting`
compresses them). Two sweeps over a boundary L:

    prefix   blocks [0, L) compressed, the rest uncompressed: does an early error survive the clean later blocks?
    suffix   blocks [L, n) compressed, the rest uncompressed: how much damage do late blocks alone do?

Evaluated on the questions the fully compressed model *lost* (base right, compressed wrong) and an equal-sized
sample it *kept*. On lost facts: recovery = share answered correctly by the hybrid. On kept facts: the hybrid
should stay correct; if it doesn't, the splice itself is doing damage (a mismatch between clean and compressed
blocks), which is the control.
"""

import random

from .compress import prune
from .models import load, parse, setting_dir


def hybrid(model_name: str, setting: str, lo: int, hi: int, _cache: dict = {}):  # noqa: B006 (deliberate memo across calls)
    """The uncompressed model with blocks [lo, hi) taken from the compressed `setting`."""
    method, level = parse(setting)
    m, tok = load(model_name, "base")
    n = len(m.layers)
    region = set(range(lo, hi))
    if method in ("gptq", "awq", "rtn"):
        key = (model_name, setting)
        if key not in _cache:
            _cache.clear()
            _cache[key] = load(model_name, setting)[0]
        comp = _cache[key]
        for i in region:
            m.layers[i] = comp.layers[i]
    elif method == "wanda":
        norms = prune.load_recipe(setting_dir(model_name, setting))
        sparsity, nm = prune.parse_level(level)
        for name, lin in prune._linear_pairs(m):
            if int(name.split(".")[1]) in region:
                import mlx.core as mx

                score = mx.abs(lin.weight).astype(mx.float32) * norms[name][None]
                lin.weight = prune.prune_by_score(lin.weight, score, sparsity, nm, per_row=True)
    elif method == "shortgpt":
        dropped = [i for i in prune.load_recipe(setting_dir(model_name, setting))["dropped"] if i in region]
        prune.shortgpt_apply(m, dropped)
    else:
        raise ValueError(f"splicing not implemented for {setting}")
    assert len(m.layers) <= n
    return m, tok


def questions(
    base: list[dict], comp: list[dict], rows: list[dict], per_group: int, seed: int = 0
) -> tuple[list[dict], list[str]]:
    """Up to `per_group` lost and `per_group` kept questions, with their label."""
    lost = [r for r, b, c in zip(rows, base, comp) if b["correct"] and not c["correct"]]
    kept = [r for r, b, c in zip(rows, base, comp) if b["correct"] and c["correct"]]
    rng = random.Random(seed)
    lost, kept = rng.sample(lost, min(per_group, len(lost))), rng.sample(kept, min(per_group, len(kept)))
    return lost + kept, ["lost"] * len(lost) + ["kept"] * len(kept)
