# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Logit lens: at which layer does the answer appear, and where does compression make it vanish?

For each question, take the residual stream at the last prompt token after every block, apply the model's final
norm and output layer, and read the log-probability (and rank) of the first token of the *base model's correct
answer*. Done for the base and a compressed model on the same questions (lost and kept facts), this shows where
in depth the answer forms, and where the compressed model stops forming it.

(The logit lens reads intermediate layers through the last layer's decoder, so early layers are read poorly;
compare base and compressed at the same layer, not layers with each other.)
"""

from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .generate import MODES, prompt_tokens


class _Keep(nn.Module):
    def __init__(self, block, sink: list):
        super().__init__()
        self.block, self.sink = block, sink

    def __call__(self, x, *args, **kwargs):
        y = self.block(x, *args, **kwargs)
        self.sink.append(y[:, -1, :])
        return y


def decode(model, h: mx.array) -> mx.array:
    h = model.model.norm(h)
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def lens(model, tokenizer, rows: list[dict], targets: list[int], batch: int = 16) -> dict:
    """Per question and layer: log p(target) and its rank among the vocabulary."""
    prompts = [prompt_tokens(tokenizer, MODES["short"][0](r["question"])) for r in rows]
    n = len(model.layers)
    logp = np.zeros((len(rows), n), dtype=np.float32)
    rank = np.zeros((len(rows), n), dtype=np.int32)
    originals = list(model.layers)
    sinks = [[] for _ in range(n)]
    for i in range(n):
        model.layers[i] = _Keep(originals[i], sinks[i])
    try:
        by_len = defaultdict(list)
        for k, p in enumerate(prompts):
            by_len[len(p)].append(k)
        for ks_all in by_len.values():
            for s in range(0, len(ks_all), batch):
                ks = ks_all[s : s + batch]
                for sink in sinks:
                    sink.clear()
                model.model(mx.array([prompts[k] for k in ks]))
                tgt = mx.array([targets[k] for k in ks])
                for i in range(n):
                    lg = decode(model, sinks[i][0]).astype(mx.float32)
                    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                    t = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
                    r = (lp > t[:, None]).sum(axis=-1)
                    mx.eval(t, r)
                    logp[ks, i], rank[ks, i] = np.array(t), np.array(r)
    finally:
        for i in range(n):
            model.layers[i] = originals[i]
    return {"logp": logp, "rank": rank}
