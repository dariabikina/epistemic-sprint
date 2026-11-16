# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""wu.probs: scoring with a shared prompt cache agrees with reading each sequence whole."""

import json

import numpy as np
import pytest

from wu import probs
from wu.confidence import continuation_logps, prefix_cache, score


def _tiny():
    import mlx.core as mx
    from mlx_lm.models import llama

    from wu.gt.train import SHAPES

    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(SHAPES["gt-1m-q"].config()))
    model.set_dtype(mx.float32)
    return model


def test_cached_scores_equal_whole_sequence_scores():
    import mlx.core as mx

    model = _tiny()
    rng = np.random.default_rng(2)
    head = rng.integers(1, 250, size=9).tolist()
    prompts = [head + rng.integers(1, 250, size=14).tolist() for _ in range(3)]  # equal length, shared start
    conts = [[rng.integers(1, 250, size=n).tolist() for n in sizes] for sizes in ((1, 4, 4, 9, 2, 1), (3,), (12, 1))]
    shared = (9, prefix_cache(model, head))
    for prefix in (None, shared, shared):  # prompts read whole; their first 9 tokens read before (used twice)
        lps, first = continuation_logps(model, prompts, conts, prefix)
        for b, p in enumerate(prompts):
            whole, _ = score(model, [p] * len(conts[b]), conts[b], [7], batch=3)
            assert [len(lp) for lp in lps[b]] == [len(c) for c in conts[b]]
            # MLX on the GPU rounds the cached and the plain pass differently (about 3e-3 in a log-probability)
            assert [lp.sum() for lp in lps[b]] == pytest.approx(whole, abs=5e-3)
            lp = model(mx.array([p]))[0, -1].astype(mx.float32)
            lp = np.array(lp - mx.logsumexp(lp))
            assert first[b] == pytest.approx(lp, abs=5e-3)
            assert np.exp(first[b]).sum() == pytest.approx(1, abs=1e-4)
            assert [x[0] for x in lps[b]] == pytest.approx([lp[c[0]] for c in conts[b]], abs=5e-3)


def test_passes_add_the_null_prompt_and_keys_are_unique():
    rows = [
        {"id": "a", "lang": "en", "condition": "scoring", "variant": "question", "exemplar_set": "A", "prompt": "p"}
        | {"null_prompt": "q"},
        {"id": "a", "lang": "en", "condition": "scoring", "variant": "paraphrase", "exemplar_set": "A", "prompt": "p"},
        {"id": "a", "lang": "en", "condition": "knows", "variant": "question", "prompt": "k"},
    ]
    got = probs.passes(rows, "scoring")
    assert [(n, p) for _, n, p in got] == [(False, "p"), (True, "q"), (False, "p")]
    assert len({probs.pass_key(r, n) for r, n, _ in got}) == 3
    assert [p for _, _, p in probs.passes(rows, "knows")] == ["k"]


@pytest.mark.skipif(not probs.PROMPTS.exists(), reason="the epistemic-probabilities folder is not here")
def test_the_prompt_file_is_as_its_readme_says():
    rows = probs.load_prompts()
    count = lambda c: sum(r["condition"] == c for r in rows)
    assert (count("short"), count("scoring"), count("knows")) == (1994, 3988, 2000)
    assert len(probs.passes(rows, "scoring")) == 3988 + 1594
    assert all(json.dumps(r["continuations"]) for r in rows if r["condition"] != "short")
    assert {r["id"] for r in probs.load_prompts(n=40)} < {r["id"] for r in rows}
