# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

import math

import numpy as np

from wu.generate import confidence
from wu.match import loose_match, match
from wu.metrics import compare, ece, isotonic_fit


def test_match_word_boundaries():
    assert match("He was a politician.", ["politician", "pol"])
    assert not match("He studied politics", ["pol"])
    assert loose_match("He studied politics", ["pol"])
    assert match("Baton Rouge", ["Baton Rouge, LA", "Baton Rouge"])
    assert match("The answer is The Beatles!", ["Beatles"])
    assert not match("", ["x"])


def test_confidence_is_geometric_mean_probability():
    assert math.isclose(confidence([math.log(0.5), math.log(0.5)]), 0.5)
    assert math.isnan(confidence([]))


def test_isotonic_is_monotone_and_fits_steps():
    f = isotonic_fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1]))
    out = f(np.array([0.1, 0.2, 0.3, 0.4, 0.9]))
    assert np.all(np.diff(out) >= 0)
    assert np.allclose(out, [0, 0.5, 0.5, 1, 1])


def test_ece_perfect_and_worst():
    assert ece(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0
    assert ece(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == 1


def _records(correct_by_bucket):
    out, i = [], 0
    for bucket, flags in correct_by_bucket.items():
        for f in flags:
            out.append({"id": str(i), "bucket": bucket, "correct": f, "conf": 0.9 if f else 0.2})
            i += 1
    return out


def test_retention_shift_and_loss_rate():
    base = _records({"head": [1] * 10, "middle": [1] * 5 + [0] * 5, "tail": [1] * 2 + [0] * 8})
    comp = _records({"head": [1] * 5 + [0] * 5, "middle": [1] * 5 + [0] * 5, "tail": [1] * 2 + [0] * 8})
    c = compare(base, comp)
    assert math.isclose(c["retention"]["all"], 12 / 17)
    assert math.isclose(c["rs_pp"]["head"], (0.5 - 12 / 17) * 100)
    assert math.isclose(c["rs_pp"]["tail"], (1 - 12 / 17) * 100)
    assert c["loss_rate"]["head"] == 0.5 and c["loss_rate"]["tail"] == 0
    assert c["loss_rate_of_all"]["head"] == 0.5


def test_jsonl_survives_unicode_line_separators(tmp_path):
    import json

    from wu.cli import read_jsonl

    rows = [{"response": "a b\x85c\x1cd"}, {"response": "ok"}]
    p = tmp_path / "x.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert read_jsonl(p) == rows


def test_bayes_posterior_centres_on_the_counts():
    from wu.bayes import posterior, summarize

    base = _records({"head": [1] * 400, "middle": [1] * 400, "tail": [1] * 400})
    comp = _records({"head": [1] * 200 + [0] * 200, "middle": [1] * 300 + [0] * 100, "tail": [1] * 400})
    d = posterior(base, comp, draws=4000)
    assert abs(summarize(d["LR_head"])["median"] - 0.5) < 0.02
    assert abs(summarize(d["r"])["median"] - 0.75) < 0.02
    assert summarize(d["RS_tail-head"])["p_positive"] > 0.999


def test_auc_and_probe():
    import numpy as np

    from wu.represent import Probe, auc, cosine_drift, knn_overlap

    assert auc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0])) == 1.0
    assert auc(np.array([1.0, 1.0]), np.array([1, 0])) == 0.5
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 50))
    y = (X[:, 0] > 0).astype(float)
    assert auc(Probe().fit(X[:300], y[:300]).score(X[300:]), y[300:]) > 0.9
    assert np.allclose(cosine_drift(X, X), 0)
    assert knn_overlap(X, X).mean() == 1.0
