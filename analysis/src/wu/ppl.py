# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""WikiText-2 perplexity, Wu A.1.2: the first 8,192 test tokens, in windows of 2,048. Compared with Wu Table 3."""

import math

import mlx.core as mx
import mlx.nn as nn

from .data import wikitext2_test

MAX_TOKENS, SEQ_LEN = 8192, 2048

# Wu Table 3, Qwen-3-8B column: what our settings should come close to.
PAPER_QWEN3_8B = {
    "base": 9.5888,
    "gptq-4": 9.9511,
    "gptq-3": 11.2652,
    "gptq-2": 141.3107,
    "awq-4": 9.9949,
    "awq-3": 11.3750,
    "awq-2": 16508.5254,
    "magnitude-30": 11.5522,
    "magnitude-50": 28.9189,
    "magnitude-70": 138372.7513,
    "wanda-30": 11.0325,
    "wanda-50": 12.7984,
    "wanda-70": 153.0605,
    "wanda-4:8": 14.7886,
    "wanda-2:4": 18.4150,
    "sparsegpt-30": 11.0395,
    "sparsegpt-50": 13.9930,
    "sparsegpt-70": 862.9209,
    "shortgpt-5": 15.1473,
    "shortgpt-10": 35.1417,
    "shortgpt-15": 43.5930,
    "shortgpt-20": 80.9849,
    "shortgpt-25": 617.8208,
}


def perplexity(model, tokenizer) -> dict:
    ids = tokenizer.encode(wikitext2_test(), add_special_tokens=False)[:MAX_TOKENS]
    nll, count = 0.0, 0
    for start in range(0, len(ids) - 1, SEQ_LEN):
        window = mx.array(ids[start : start + SEQ_LEN + 1])[None]
        logits = model(window[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, window[:, 1:], reduction="sum")
        nll, count = nll + loss.item(), count + window.shape[1] - 1
    return {"ppl": math.exp(nll / count), "tokens": count, "seq_len": SEQ_LEN}


def window_nll(model, tokenizer, windows: int = 32) -> list[dict]:
    """Summed NLL and token count for each of the first `windows` 2,048-token windows of WikiText-2 test.
    The paper's protocol uses the first 4; more windows show how much the estimate depends on which text is read."""
    ids = tokenizer.encode(wikitext2_test(), add_special_tokens=False)
    out = []
    for w in range(windows):
        start = w * SEQ_LEN
        window = mx.array(ids[start : start + SEQ_LEN + 1])[None]
        if window.shape[1] < 2:
            break
        logits = model(window[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, window[:, 1:], reduction="sum")
        out.append({"window": w, "nll": loss.item(), "tokens": window.shape[1] - 1})
    return out
