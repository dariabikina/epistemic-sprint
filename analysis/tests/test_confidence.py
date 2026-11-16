# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""wu.urn (exact marble probabilities) and wu.confidence (batched reads) on a tiny random model."""

import random
import re

import numpy as np
import pytest

from wu import urn
from wu.confidence import accepted, aliases


def test_marble_probabilities_follow_from_the_text():
    items = urn.marbles()
    assert len(items) == 900
    for it in items:
        assert abs(sum(it["p"]) - 1) < 1e-12 and len(it["p"]) == it["k"] == len(set(it["colours"]))
        seen = it["text"].split("Colours drawn so far: ")[1].rstrip(".").split(", ")
        assert it["predict"].endswith(",") and it["predict"][:-1] == it["text"][:-1]
        assert set(seen) <= set(it["colours"])
        if it["kind"] == "bag":
            said = {c: int(n) for n, c in re.findall(r"(\d+) (\w+) marble", it["text"])}
            assert [said[c] / 10 for c in it["colours"]] == pytest.approx(it["p"])
            assert len(seen) == urn.BAG_SHOWN
        else:
            assert len(seen) == it["n"]
            expect = [(1 + seen.count(c)) / (it["k"] + it["n"]) for c in it["colours"]]
            assert expect == pytest.approx(it["p"])


def test_urn_rule_matches_its_own_simulation():
    # P(next = c | draws) = (1 + count_c) / (K + n): check one history by simulating the stated rule.
    rng, hits, total = random.Random(0), 0, 0
    for _ in range(200_000):
        bag = ["a", "b"]
        history = []
        for _ in range(3):
            m = rng.choice(bag)
            history.append(m)
            bag.append(m)
        if history == ["a", "a", "b"]:
            total += 1
            hits += rng.choice(bag) == "a"
    assert hits / total == pytest.approx((1 + 2) / (2 + 3), abs=0.01)


def test_stated_probability_and_aliases():
    one_hot = [0.0] * 10
    one_hot[7] = 1.0
    assert urn.stated_probability(one_hot) == pytest.approx(0.75)
    assert urn.stated_probability([0.2] * 10) == pytest.approx(0.5)  # unnormalized input is normalized
    assert aliases(["Paris", "paris", "The Paris", "", "Lutetia"]) == ["Paris", "Lutetia"]
    assert len(aliases([f"name {i}" for i in range(30)])) == 8
    assert accepted(["barack obama"], "Obama") == ["barack obama", "Barack obama", "Barack Obama", "Obama"]
    assert accepted(["Paris"], None) == ["Paris"]


def _tiny():
    import mlx.core as mx
    from mlx_lm.models import llama

    from wu.gt.train import SHAPES

    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(SHAPES["gt-1m-q"].config()))
    model.set_dtype(mx.float32)
    return model


def test_read_is_unchanged_by_padding_and_its_last_layer_is_the_output():
    import mlx.core as mx

    from wu.confidence import read

    model = _tiny()
    rng = np.random.default_rng(0)
    seqs = [rng.integers(1, 250, size=n).tolist() for n in (5, 9, 9, 17, 30)]
    groups = [[3], [10, 11], [200]]
    together = read(model, seqs, groups, batch=8)
    for i, s in enumerate(seqs):
        alone = read(model, [s], groups, batch=1)
        # MLX on the GPU rounds a one-row product differently (about 2e-3 in a logit; exact on the CPU)
        assert alone["logp"][0] == pytest.approx(together["logp"][i], abs=5e-3)
        assert alone["H"][0].astype(np.float32) == pytest.approx(together["H"][i].astype(np.float32), abs=2e-2)
        lp = model(mx.array([s]))[0, -1].astype(mx.float32)
        lp = np.array(lp - mx.logsumexp(lp))
        assert together["logp"][i, 0] == pytest.approx(lp[3], abs=1e-4)
        assert together["logp"][i, 1] == pytest.approx(np.logaddexp(lp[10], lp[11]), abs=1e-4)
    assert together["layers"][-1] == len(model.layers) - 1
    assert together["lens"][:, -1] == pytest.approx(together["logp"], abs=1e-4)  # the lens at the last block


def test_score_sums_the_continuation_and_reads_the_stop():
    import mlx.core as mx

    from wu.confidence import score

    model = _tiny()
    rng = np.random.default_rng(1)
    prompts = [rng.integers(1, 250, size=n).tolist() for n in (4, 7, 12)]
    conts = [rng.integers(1, 250, size=n).tolist() for n in (3, 0, 5)]
    stop = [7, 8]
    logp, logp_stop = score(model, prompts, conts, stop, batch=3)
    for i, (p, c) in enumerate(zip(prompts, conts)):
        lp = model(mx.array([p + c]))[0].astype(mx.float32)
        lp = np.array(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
        expect = sum(lp[len(p) - 1 + j, t] for j, t in enumerate(c))
        assert logp[i] == pytest.approx(expect, abs=1e-3)
        assert logp_stop[i] == pytest.approx(np.logaddexp(lp[-1, 7], lp[-1, 8]), abs=1e-4)
