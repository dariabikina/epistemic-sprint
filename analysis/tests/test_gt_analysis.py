# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""The analyses of the ground-truth models: intervals, decompositions, Jacobians, probes, the collapse test."""

import json

import mlx.core as mx
import numpy as np

from wu.gt import control, measures, stats
from wu.gt import priors as P
from wu.represent import auc as reference_auc


def test_cluster_bootstrap_mean_and_coverage():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 20))
    ci = stats.mean_over(v)
    assert np.isclose(ci["est"], v.mean()) and ci["lo"] < ci["est"] < ci["hi"]
    hits = 0
    for r in range(100):  # sequences with a shared offset: positions within a sequence are correlated
        x = np.random.default_rng(r).normal(size=(40, 1)) + 0.3 * np.random.default_rng(1000 + r).normal(size=(40, 16))
        c = stats.mean_over(x, B=300)
        hits += c["lo"] <= 0 <= c["hi"]
    assert 85 <= hits <= 100


def test_masked_mean_is_a_ratio_over_selected_positions():
    v = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    m = np.array([[True, False, True], [False, True, False]])
    assert np.isclose(stats.mean_over(v, m)["est"], (1 + 3 + 20) / 3)


def test_fast_auc_matches_the_reference():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 300).astype(float)  # many ties
    y = (rng.random(300) < 0.4).astype(float)
    assert np.isclose(stats.auc(s, y), reference_auc(s, y))


def test_token_kinds():
    x = np.array([[5, 7, 5, 9, 7]], dtype=np.int16)
    p0 = np.linspace(1, 0.1, P.NBINS)
    p0 /= p0.sum()  # symbol 0 commonest, rank = index
    k = measures.token_kinds(x, p0)
    assert k["new"].tolist() == [[True, True, False, True, False]]
    assert k["rank"].tolist() == [[5, 7, 5, 9, 7]]


def test_excess_splits_exactly_into_meta_and_content():
    rng = np.random.default_rng(2)
    n, T = 8, 30
    new = rng.random((n, T)) < 0.3

    def fake():
        return {
            "logp": np.log(rng.uniform(0.01, 0.5, (n, T))),
            "novel_model": rng.uniform(0.05, 0.9, (n, T)),
            "bayes_logp": np.log(rng.uniform(0.01, 0.5, (n, T))),
            "novel_bayes": rng.uniform(0.05, 0.9, (n, T)),
        }

    e = fake()
    kind, which = measures.split_logp(e, new, model=True)
    assert np.allclose(kind + which, e["logp"])
    b, c = fake(), fake()
    c["bayes_logp"], c["novel_bayes"] = b["bayes_logp"], b["novel_bayes"]
    kb, wb = measures.split_logp(b, new, True)
    kc, wc = measures.split_logp(c, new, True)
    total = (-c["logp"] + c["bayes_logp"]) - (-b["logp"] + b["bayes_logp"])
    assert np.allclose(total, (kb - kc) + (wb - wc))


def test_jacobian_matches_finite_differences():
    from mlx_lm.models import llama

    from wu.gt.inside import forward_states, jacobian
    from wu.gt.train import SHAPES, initialize

    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(SHAPES["gt-1m-q"].config()))
    initialize(model, 4)
    tok = mx.array([[P.BOS, 3, 17, 3, 200, 5]])
    hs = forward_states(model, tok)
    l, t = 1, 4
    J = jacobian(model, hs[l], l, t)

    def final(h):
        for layer in model.layers[l:]:
            h = layer(h, "causal", None)
        return np.array(h[0, t].astype(mx.float32))

    h0 = hs[l].astype(mx.float32)
    eps, cols = 1e-2, [0, 7, 100]
    for c in cols:
        bump = mx.zeros(h0.shape).at[0, t, c].add(eps)
        fd = (final(h0 + bump) - final(h0 - bump)) / (2 * eps)
        assert np.allclose(J[:, c], fd, atol=2e-3, rtol=2e-2)


def test_cross_fitted_probe_finds_signal_and_not_noise():
    from wu.gt.inside import cross_fitted_ridge

    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(60), 20)
    X = rng.normal(size=(1200, 30))
    y = X[:, 0] * 2 + 0.3 * rng.normal(size=1200)
    assert stats.r2_over(y, cross_fitted_ridge(X, y, groups), groups)["est"] > 0.9
    noise = rng.normal(size=1200)
    assert stats.r2_over(noise, cross_fitted_ridge(X, noise, groups), groups)["est"] < 0.05


def test_collapse_test_tells_one_line_from_many():
    rng = np.random.default_rng(4)
    one = {f"r{i}": (x := rng.uniform(1, 5, 200), 0.5 - 0.1 * x + 0.02 * rng.normal(size=200)) for i in range(3)}
    f = control.collapse(one, B=100)
    assert f["r2_separate_lines"] - f["r2_pooled"] < 0.01
    assert f["slope_per_decade"]["lo"] < -0.1 < f["slope_per_decade"]["hi"]
    many = {f"r{i}": (x := rng.uniform(1, 5, 200), i * 0.5 - 0.1 * x + 0.02 * rng.normal(size=200)) for i in range(3)}
    g = control.collapse(many, B=100)
    assert g["r2_separate_lines"] - g["r2_pooled"] > 0.5


def test_exposure_counts_documents_containing_each_symbol(tmp_path, monkeypatch):
    from wu.gt import data as D

    monkeypatch.setattr(D, "GT", tmp_path)
    pool = np.array([[1, 1, 2], [2, 3, 3], [1, 4, 4]], dtype=np.int16)
    (tmp_path / "pool").mkdir()
    np.save(tmp_path / "pool" / "train.npy", pool)
    (tmp_path / "runs" / "r").mkdir(parents=True)
    (tmp_path / "runs" / "r" / "config.json").write_text(
        json.dumps({"data": str(tmp_path / "pool"), "steps": 3, "batch_size": 2})
    )
    E = control.exposure("r")
    # documents containing: 1 -> 2 of 3, 2 -> 2, 3 -> 1, 4 -> 1; 6 documents drawn in training
    assert np.allclose(E[[1, 2, 3, 4]], np.array([2, 2, 1, 1]) / 3 * 6) and E[0] == 0


def test_single_block_compression_leaves_other_blocks_untouched():
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama

    from wu.compress import quant
    from wu.gt.train import SHAPES, initialize

    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(SHAPES["gt-1m-q"].config()))
    initialize(model, 4)
    before = [dict(tree_flatten(b.parameters())) for b in model.layers]
    calib = np.random.default_rng(0).integers(0, P.NBINS, (8, 33)).tolist()
    with quant_group(quant, 64):
        quant.gptq(model, None, 2, progress=None, calibration=calib, blocks={1})
    after = [dict(tree_flatten(b.parameters())) for b in model.layers]
    for i in (0, 2, 3):
        assert after[i].keys() == before[i].keys() and all(mx.array_equal(after[i][k], before[i][k]) for k in after[i])
    assert any("scales" in k for k in after[1])


def quant_group(quant, g):
    from wu.gt.compress import group_size

    return group_size(g)


def test_report_queries_and_exact_target():
    from wu.gt import report as R

    x, _ = P.worlds()["zipf"].sample(3, np.random.default_rng(5))
    after = np.array([[0, 10, 200], [5, 6, 254], [100, 101, 102]])
    b = R.with_queries(x, after)
    toks = b["tokens"]
    for i in range(3):
        assert (toks[i] == R.QUERY).sum() == 3 and toks[i, 0] == P.BOS
        assert np.array_equal(toks[i][toks[i] < P.NBINS], x[i])  # symbols in order
        for t in range(1, P.T):  # the input predicting x_t is x_{t-1}
            assert toks[i, b["sym_pos"][i, t]] == x[i, t - 1]
        assert b["sym_pos"][i, 0] == 0
        assert np.all(toks[i, b["query_pos"][i]] == R.QUERY) and np.all(b["query_pos"][i] == b["guess_pos"][i] + 1)
        assert np.array_equal(toks[i, b["guess_pos"][i]], x[i, after[i]])
    exact = P.worlds()["zipf"].predict(x)[0]  # predictive of x_t given x_<t
    assert np.allclose(R.bayes_after(x, after), np.stack([exact[i, after[i] + 1] for i in range(3)]), atol=1e-10)
