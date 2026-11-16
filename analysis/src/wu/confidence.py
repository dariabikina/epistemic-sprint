# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Two readings of a language model's confidence, with the truth known: facts and marbles.

The language-model counterpart of the ground-truth experiments (`wu.gt`, E1). There, a small model guesses
the next symbol; c is the exact probability that the guess is right, m the probability the model put on its
guess, r its answer to a query token. Here:

    facts    PopQA questions, with the answer each model already gave (eval-short-<n>.jsonl).
             truth       the answer is right or wrong (Wu's matching): 0 or 1.
             knowledge   log p_gold: log of the summed probability of the accepted replies, each followed by
                         the end-of-turn token. Accepted: up to MAX_ALIASES aliases, each also capitalized
                         and in title case, and the base model's own reply where that was right. The same set
                         for every setting. The counterpart of log p_model of the true symbol. It is a lower
                         bound on P(a correct answer): other correct phrasings are not counted.
             prediction  the stored log-probability of the model's own reply (eval-short: logprob_sum), the
                         counterpart of m. logp_own re-scores the reply from its text, as a check: it differs
                         where the text does not tokenize back to the generated tokens (3 of 200 in a smoke run).
             stated      the model's reply shown as a proposed answer (P(True), Kadavath et al. 2022):
                         P(Yes) to "Is the proposed answer correct?" and P(No) to "... incorrect?", for its
                         own reply (`own`), for a gold answer (`gold`: does it still accept the right
                         answer?) and for the base model's reply (`base`: the same strings judged by every
                         setting, so a change is in the judging, not in what is judged). The counterpart
                         of r. Two polarities, because models lean towards one of Yes and No.
                         `self_correct`: "Is your answer correct?" asked in the same chat after its own
                         reply; kept as a record of that lean (qwen3-1.7b says Yes with probability 1.00 to
                         this and to "Is your answer wrong?").
    marbles  `wu.urn`: texts whose next-colour probability is exact.
             prediction  log-probability of each of the 8 colour words as the next token of the plain text.
             stated      a distribution over the digits 0-9 for "how likely is <colour>?", per colour.

Everything is read at the last token of a prompt. For each read we keep, besides the output probabilities:
    lens   the same probabilities read through the logit lens (final norm and output layer applied to the
           residual stream) after each recorded block;
    H      the residual stream itself after each recorded block (float16),
so that the probabilities can be related to the hidden states that produce them (`wu.inside_confidence`).

Batches are right-padded. Attention is causal, so padding to the right of a sequence does not change
anything at or before its last token (tested in tests/test_confidence.py).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import urn
from .generate import MODES, prompt_tokens
from .match import normalize
from .represent import record_layers

MAX_ALIASES = 8
PROPOSED = "Question: {q}\nProposed answer: {a}\nIs the proposed answer {what}? Answer only Yes or No."
ASK_SELF = "Is your answer correct? Answer only Yes or No."
YES, NO = ("Yes", " Yes", "yes", " yes"), ("No", " No", "no", " no")


class _KeepAll(nn.Module):
    """A block that also stores its output (every position)."""

    def __init__(self, block, sink: list):
        super().__init__()
        self.block, self.sink = block, sink
        self.use_sliding = getattr(block, "use_sliding", False)  # read by mlx_lm's llama model

    def __call__(self, x, *args, **kwargs):
        y = self.block(x, *args, **kwargs)
        self.sink.append(y)
        return y


def _batches(seqs: list[list[int]], batch: int):
    """Indices in batches of similar length, with the padded token array and each row's last position."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch):
        part = order[s : s + batch]
        width = max(len(seqs[i]) for i in part)
        ids = np.zeros((len(part), width), dtype=np.int32)
        for row, i in enumerate(part):
            ids[row, : len(seqs[i])] = seqs[i]
        yield part, mx.array(ids), mx.array([len(seqs[i]) - 1 for i in part])


def _logits(model, h: mx.array) -> mx.array:
    """Output logits from hidden states that have been through the final norm. The states are cast to
    float32 first: in bf16 a logit near 20 is rounded to steps of 0.125."""
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    return head(h.astype(mx.float32)).astype(mx.float32)


def _group_logp(logits: mx.array, groups: list[list[int]]) -> mx.array:
    """(B, V) logits -> (B, G): log of the total probability of each group of tokens."""
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    return mx.stack([mx.logsumexp(lp[:, mx.array(g)], axis=-1) for g in groups], axis=-1)


def read(model, seqs: list[list[int]], groups: list[list[int]], dropped=(), batch: int = 32, states: bool = True):
    """At each sequence's last token: the log-probability of each token group, the same through the logit
    lens after each recorded block, and (if `states`) the residual stream there.

    Layers are named by the original block index; `dropped` lists removed blocks (ShortGPT), as in
    `represent.states`. Returns {"logp": (n, G), "lens": (n, L, G), "H": (n, L, d), "layers": (L,)}."""
    n_orig = len(model.layers) + len(dropped)
    kept = [i for i in range(n_orig) if i not in set(dropped)]
    layers = record_layers(n_orig)
    where = {L: max(j for j, k in enumerate(kept) if k <= L) for L in layers}
    n, G = len(seqs), len(groups)
    out = {
        "logp": np.zeros((n, G), np.float32),
        "lens": np.zeros((n, len(layers), G), np.float32),
        "layers": np.array(layers),
    }
    if states:
        out["H"] = np.zeros((n, len(layers), model.args.hidden_size), np.float16)
    originals = list(model.layers)
    sinks = {j: [] for j in set(where.values())}
    for j in sinks:
        model.layers[j] = _KeepAll(originals[j], sinks[j])
    try:
        for part, ids, last in _batches(seqs, batch):
            for sink in sinks.values():
                sink.clear()
            rows = mx.arange(len(part))
            logp = _group_logp(_logits(model, model.model(ids)[rows, last]), groups)  # model.model ends in the norm
            at = {j: sinks[j][0][rows, last] for j in sinks}
            lens = mx.stack(
                [_group_logp(_logits(model, model.model.norm(at[where[L]])), groups) for L in layers], axis=1
            )
            mx.eval(logp, lens)
            out["logp"][part], out["lens"][part] = np.array(logp), np.array(lens)
            if states:
                H = mx.stack([at[where[L]] for L in layers], axis=1).astype(mx.float32)
                out["H"][part] = np.array(H).astype(np.float16)
    finally:
        for j in sinks:
            model.layers[j] = originals[j]
    return out


def score(model, prompts: list[list[int]], continuations: list[list[int]], stop: list[int], batch: int = 32):
    """log p(continuation | prompt), and log p(a stop token comes next), for each pair. An empty
    continuation scores 0 and the stop probability is read at the prompt's last token."""
    seqs = [p + c for p, c in zip(prompts, continuations)]
    logp, logp_stop = np.zeros(len(seqs), np.float64), np.zeros(len(seqs), np.float64)
    stop_ids = mx.array(sorted(stop))
    for part, ids, last in _batches(seqs, batch):
        lp = _logits(model, model.model(ids))
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        nxt = mx.take_along_axis(lp[:, :-1], ids[:, 1:, None], axis=-1)[..., 0]  # [b, t] = log p(ids[t+1] | ..t)
        end = mx.logsumexp(lp[mx.arange(len(part)), last][:, stop_ids], axis=-1)
        mx.eval(nxt, end)
        nxt, end = np.array(nxt, dtype=np.float64), np.array(end)
        for row, i in enumerate(part):
            a, b = len(prompts[i]), len(seqs[i])
            logp[i] = nxt[row, a - 1 : b - 1].sum()
            logp_stop[i] = end[row]
    return logp, logp_stop


def _fork(cache, rows: list[int]):
    """New caches holding the keys and values of the given rows of `cache` (rows may repeat). Each is
    exactly full, so the next token allocates new arrays and `cache` itself is never written to."""
    from mlx_lm.models.cache import KVCache

    idx, out = mx.array(rows), []
    for c in cache:
        k = KVCache()
        k.state = (c.keys[..., : c.offset, :][idx], c.values[..., : c.offset, :][idx], c.offset)
        out.append(k)
    return out


def prefix_cache(model, tokens: list[int]):
    """The key-value cache after reading `tokens`, to be shared by every prompt that starts with them."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    model.model(mx.array([tokens]), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _similar_lengths(lengths: list[int], order: list[int], positions: int):
    """Split indices (sorted by length) into batches of similar length, so little of a batch is padding,
    each of at most `positions` tokens with its padding."""
    part = []
    for i in order:
        if part and (lengths[i] > 1.5 * lengths[part[0]] + 1 or (len(part) + 1) * lengths[i] > positions):
            yield part
            part = []
        part.append(i)
    if part:
        yield part


def continuation_logps(model, prompts: list[list[int]], continuations: list[list[list[int]]], prefix=None):
    """Prompts of equal length, each with its own continuations: the log-probability of every
    continuation token (one array per continuation, in a list per prompt), and the full next-token
    log-probabilities at the end of each prompt, float32 (B, V).

    The prompts are read once, together; their key-value caches are then shared by their continuations,
    so the cost is the prompts plus the continuation tokens, in a few forward passes (on an 8B model a
    pass costs about 0.1 s however short it is). `prefix` = (n, cache from `prefix_cache`) when the
    first n tokens are the same for all prompts and have been read already. `score` reads prompt +
    continuation whole each time; the two agree (tests/test_probs.py)."""
    from mlx_lm.models.cache import make_prompt_cache

    assert len({len(p) for p in prompts}) == 1, "prompts in a batch must be of equal length"
    n, cache = (prefix[0], _fork(prefix[1], [0] * len(prompts))) if prefix else (0, make_prompt_cache(model))
    assert n < len(prompts[0])
    first = _logits(model, model.model(mx.array([p[n:] for p in prompts]), cache=cache)[:, -1])
    first = first - mx.logsumexp(first, axis=-1, keepdims=True)
    mx.eval(first, [c.state for c in cache])
    first = np.array(first)
    flat = [(b, c) for b, conts in enumerate(continuations) for c in conts]
    assert all(c for _, c in flat), "an empty continuation has no tokens to score"
    got = [np.array([first[b, c[0]]], np.float64) for b, c in flat]
    inputs = [len(c) - 1 for _, c in flat]  # the first token is predicted by the prompt
    order = sorted((i for i, w in enumerate(inputs) if w), key=lambda i: inputs[i])
    for part in _similar_lengths(inputs, order, positions=1536):
        width = max(inputs[i] for i in part)
        ids = np.zeros((len(part), width), np.int32)  # token t, which predicts token t + 1
        nxt = np.zeros((len(part), width), np.int32)
        for row, i in enumerate(part):
            c = flat[i][1]
            ids[row, : len(c) - 1], nxt[row, : len(c) - 1] = c[:-1], c[1:]
        lp = _logits(model, model.model(mx.array(ids), cache=_fork(cache, [flat[i][0] for i in part])))
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        later = mx.take_along_axis(lp, mx.array(nxt)[..., None], axis=-1)[..., 0]
        mx.eval(later)
        later = np.array(later)
        for row, i in enumerate(part):
            got[i] = np.concatenate([got[i], later[row, : inputs[i]]])
    out, at = [], 0
    for conts in continuations:
        out.append(got[at : at + len(conts)])
        at += len(conts)
    return out, first


def first_tokens(tokenizer, words) -> list[int]:
    return sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in words})


def aliases(answers: list[str]) -> list[str]:
    """The accepted answers, without duplicates under the matcher's normalization, at most MAX_ALIASES."""
    seen, out = set(), []
    for a in answers:
        key = normalize(a)
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out[:MAX_ALIASES]


def accepted(answers: list[str], base_reply: str | None) -> list[str]:
    """The replies counted as the gold answer: each alias as given, capitalized and in title case (PopQA
    lists many aliases in lower case), and the base model's reply if it was right."""
    out = []
    for a in aliases(answers):
        out += [a, a[0].upper() + a[1:], a.title()]
    if base_reply:
        out.append(base_reply)
    return list(dict.fromkeys(out))


def _p_yes(logp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From log p of the Yes and No groups: P(Yes) among the two, and the probability on either."""
    both = np.logaddexp(logp[:, 0], logp[:, 1])
    return np.exp(logp[:, 0] - both), np.exp(both)


def facts(model, tokenizer, rows, records, base_records, dropped=(), batch: int = 32) -> dict:
    """`rows`: PopQA questions; `records`: this model's graded replies to them; `base_records`: the base
    model's. All in the same order."""
    assert [r["id"] for r in rows] == [r["id"] for r in records] == [r["id"] for r in base_records]
    short = MODES["short"][0]
    prompts = [prompt_tokens(tokenizer, short(r["question"])) for r in rows]
    stop = sorted(tokenizer.eos_token_ids)
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)

    # knowledge: the probability of each accepted reply
    owner, reply_tokens = [], []
    for i, (r, b) in enumerate(zip(rows, base_records)):
        for a in accepted(r["answers"], b["response"] if b["correct"] else None):
            owner.append(i)
            reply_tokens.append(enc(a))
    lp, lp_stop = score(model, [prompts[i] for i in owner], reply_tokens, stop, batch)
    logp_gold = np.full(len(rows), -np.inf)
    for i, v in zip(owner, lp + lp_stop):
        logp_gold[i] = np.logaddexp(logp_gold[i], v)

    # a check on the stored prediction confidence: the model's own reply, re-scored from its text
    logp_own, logp_own_stop = score(model, prompts, [enc(r["response"]) for r in records], stop, batch)

    # stated: its own reply, and a gold answer, shown as a proposed answer
    yes_no = [first_tokens(tokenizer, YES), first_tokens(tokenizer, NO)]
    assert not set(yes_no[0]) & set(yes_no[1])
    out = {
        "ids": np.array([r["id"] for r in rows]),
        "logp_gold": logp_gold,
        "n_accepted": np.bincount(owner, minlength=len(rows)),
        "logp_own": logp_own,
        "logp_own_stop": logp_own_stop,
    }
    gold = [aliases(r["answers"])[0] for r in rows]
    own = [r["response"] for r in records]
    base = [b["response"] for b in base_records]
    for who, answers in (("own", own), ("gold", gold), ("base", base)):
        for name, what in (("correct", "correct"), ("wrong", "incorrect")):
            asks = [
                prompt_tokens(
                    tokenizer, [{"role": "user", "content": PROPOSED.format(q=r["question"], a=a, what=what)}]
                )
                for r, a in zip(rows, answers)
            ]
            keep = who == "own" and name == "correct"  # hidden states for one of the six reads
            got = read(model, asks, yes_no, dropped, batch, states=keep)
            out[f"{who}_{name}_p_yes"], out[f"{who}_{name}_mass"] = _p_yes(got["logp"])
            if keep:
                out |= {"layers": got["layers"], "lens_yes_no": got["lens"], "H": got["H"]}
    asks = [
        prompt_tokens(
            tokenizer,
            [*short(r["question"]), {"role": "assistant", "content": a}, {"role": "user", "content": ASK_SELF}],
        )
        for r, a in zip(rows, own)
    ]
    out["self_correct_p_yes"], out["self_correct_mass"] = _p_yes(
        read(model, asks, yes_no, dropped, batch, states=False)["logp"]
    )
    return out


def marbles(model, tokenizer, items: list[dict], dropped=(), batch: int = 32) -> dict:
    colour_groups = [first_tokens(tokenizer, [" " + c]) for c in urn.COLOURS]
    digit_groups = [first_tokens(tokenizer, [str(d)]) for d in range(10)]
    assert len({g[0] for g in colour_groups}) == len(urn.COLOURS)
    predict = read(model, [tokenizer.encode(it["predict"]) for it in items], colour_groups, dropped, batch)
    prefill = tokenizer.encode(urn.STATE_PREFILL + "7", add_special_tokens=False)[:-1]  # ends in the space token
    item, colour, asks = [], [], []
    for i, it in enumerate(items):
        for j, c in enumerate(it["colours"]):
            item.append(i)
            colour.append(j)
            asks.append(prompt_tokens(tokenizer, urn.state_message(it, c)) + prefill)
    state = read(model, asks, digit_groups, dropped, batch)
    return {
        "ids": np.array([it["id"] for it in items]),
        "layers": predict["layers"],
        "predict_logp": predict["logp"],  # (items, 8 colours in urn.COLOURS order)
        "predict_lens": predict["lens"],
        "predict_H": predict["H"],
        "state_item": np.array(item),
        "state_colour": np.array(colour),  # index into the item's own `colours`
        "state_logp": state["logp"],  # (asks, 10 digits)
        "state_lens": state["lens"],
        "state_H": state["H"],
    }
