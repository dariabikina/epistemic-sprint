# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Ask PopQA questions, greedily, and keep each answer's token log-probs.

Modes (the same questions in each; Wu does not publish its prompt, so these are ours):
    short     the Wu reproduction: a bare answer, at most 32 new tokens. Conf (Eq. 4) comes from here.
    chat      the question as a user would ask it, at most 256 tokens: material for the claim/hedge/IDK judge.
    chat-idk  as chat, with a system prompt that makes "I don't know" an acceptable answer.
"""

import math
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from .match import loose_match, match

IDK_SYSTEM = "If you are not sure of the answer, it is fine to say that you don't know."

MODES = {  # name -> (messages for a question, max new tokens)
    "short": (
        lambda q: [
            {"role": "user", "content": f"Answer the question with only the answer, no explanation.\nQuestion: {q}"}
        ],
        32,
    ),
    "chat": (lambda q: [{"role": "user", "content": q}], 256),
    "chat-idk": (lambda q: [{"role": "system", "content": IDK_SYSTEM}, {"role": "user", "content": q}], 256),
}


def prompt_tokens(tokenizer, messages: list[dict]) -> list[int]:
    """Chat template up to the assistant's turn, thinking disabled (Qwen3 is a hybrid reasoning model; other
    templates ignore the flag). Templates without a system role (Gemma) get the system text at the start of
    the first user message, which is what those models' own documentation recommends."""
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    except Exception:
        if messages[0]["role"] != "system":
            raise
        system, user, *rest = messages
        merged = [{"role": "user", "content": f"{system['content']}\n\n{user['content']}"}, *rest]
        return tokenizer.apply_chat_template(merged, add_generation_prompt=True, enable_thinking=False, tokenize=True)


@dataclass
class Completion:
    text: str
    tokens: list[int]  # generated ids, without the stop token
    logprobs: list[float]  # log p(token | prompt, earlier tokens) at temperature 1, aligned with tokens
    finished: bool  # False when cut off at max_tokens


def generate(model, tokenizer, prompts: list[list[int]], *, max_tokens: int, batch_size: int = 64) -> list[Completion]:
    """Greedy completions with MLX-LM's batched generator."""
    stop = set(tokenizer.eos_token_ids)
    gen = BatchGenerator(
        model,
        max_tokens=max_tokens,
        stop_tokens=[[t] for t in stop],
        sampler=make_sampler(temp=0.0),
        completion_batch_size=batch_size,
        prefill_batch_size=min(batch_size, 16),
    )
    uids = gen.insert(prompts, [max_tokens] * len(prompts))
    tokens, logprobs, finished = {u: [] for u in uids}, {u: [] for u in uids}, {}
    try:
        while responses := gen.next_generated():
            for r in responses:
                tokens[r.uid].append(r.token)
                logprobs[r.uid].append(r.logprobs[r.token])
                if r.finish_reason is not None:
                    finished[r.uid] = r.finish_reason == "stop"
    finally:
        gen.close()
    out = []
    for u in uids:
        t, lp = tokens[u], mx.stack(logprobs[u]).tolist()
        if finished[u]:
            t, lp = t[:-1], lp[:-1]  # the answer is what comes before the stop token
        out.append(Completion(tokenizer.decode(t), t, lp, finished[u]))
    return out


def confidence(logprobs: list[float]) -> float:
    """Wu Eq. 4: length-normalized sequence probability, p(answer | question)^(1/T)."""
    return math.exp(sum(logprobs) / len(logprobs)) if logprobs else float("nan")


def ask(
    model,
    tokenizer,
    rows: list[dict],
    mode: str,
    *,
    batch_size: int = 64,
    chunk: int = 256,
    progress=lambda m: print(m, flush=True),
) -> list[dict]:
    """One graded record per PopQA row."""
    messages, max_tokens = MODES[mode]
    records, start = [], time.time()
    for i in range(0, len(rows), chunk):
        part = rows[i : i + chunk]
        completions = generate(
            model,
            tokenizer,
            [prompt_tokens(tokenizer, messages(r["question"])) for r in part],
            max_tokens=max_tokens,
            batch_size=batch_size,
        )
        for r, c in zip(part, completions):
            records.append(
                {
                    "id": r["id"],
                    "bucket": r["bucket"],
                    "s_pop": r["s_pop"],
                    "question": r["question"],
                    "response": c.text,
                    "finished": c.finished,
                    "n_tokens": len(c.tokens),
                    "logprob_sum": sum(c.logprobs),
                    "conf": confidence(c.logprobs),
                    "correct": match(c.text, r["answers"]),
                    "correct_loose": loose_match(c.text, r["answers"]),
                }
            )
        done, elapsed = len(records), time.time() - start
        progress(
            f"  {done}/{len(rows)} questions, {60 * done / elapsed:.0f}/min, "
            f"accuracy so far {sum(x['correct'] for x in records) / done:.3f}"
        )
    return records
