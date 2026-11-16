# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""The ground-truth worlds: is the Bayes predictive really the posterior predictive of the prior?

Chain-rule checks: the product of one-step predictives must equal the joint density of the sequence, which
is known in closed form (Gaussian, multivariate t, Dirichlet process) or by enumeration (changepoints)."""

import itertools
import math

import numpy as np
import pytest
from scipy import special, stats

from wu.gt import priors as P

RNG = np.random.default_rng(0)


def test_bins_partition_the_line():
    assert len(P.EDGES) == P.NBINS - 1 and P.quantize(np.array([-100, -6, 0, 5.99, 100])).tolist() == [
        0,
        0,
        128,
        255,
        255,
    ]
    p = P.gauss_bins(np.array([0.3, -7.0]), np.array([0.2, 1.0]))
    assert np.allclose(p.sum(-1), 1) and (p >= 0).all()


def test_gaussian_chain_rule_equals_joint_density():
    n = 40
    K = P.kernel("rbf", n=n, ell=5) + P.kernel("periodic", n=n, period=7, ell=1.0)
    y = RNG.multivariate_normal(np.zeros(n), K + 0.09 * np.eye(n), size=3)
    m, v, _ = P.gauss_sequential(y, K, 0.09)
    chain = (-0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)).sum(1)
    joint = stats.multivariate_normal(np.zeros(n), K + 0.09 * np.eye(n)).logpdf(y)
    assert np.allclose(chain, joint, atol=1e-8)
    # per-sequence factor (heteroscedastic noise) gives the same answer
    m2, v2, _ = P.gauss_sequential(y, K, np.full(y.shape, 0.09))
    assert np.allclose(m, m2) and np.allclose(v, v2)


def test_student_t_chain_rule_equals_multivariate_t():
    w = P.worlds()["tp"]
    nu = w.params["nu"]
    C = w._C()
    y = np.stack([stats.multivariate_t(np.zeros(P.T), C * (nu - 2) / nu, df=nu).rvs(random_state=s) for s in range(2)])
    m, scale, dof, _, _ = w.sequential(y, extra_noise=0.0)
    chain = stats.t.logpdf(y, dof, loc=m, scale=scale).sum(1)
    joint = stats.multivariate_t(np.zeros(P.T), C * (nu - 2) / nu, df=nu).logpdf(y)
    assert np.allclose(chain, joint, atol=1e-6)


def _segments_logpdf(y, sigma2):
    n = len(y)
    return stats.multivariate_normal(np.zeros(n), np.ones((n, n)) + sigma2 * np.eye(n)).logpdf(y)


def test_changepoint_filter_equals_enumeration():
    w = P.Regimes("r", "", {"hazard": 0.2, "sigma": 0.3}, K=64)
    n = 7
    y = RNG.standard_normal((3, n))
    chain = w.run(y, density=True).sum(1)
    h = 0.2
    for i in range(3):
        total = -np.inf
        for starts in itertools.product([0, 1], repeat=n - 1):  # 1 = a new segment starts before value t
            lp = sum(np.log(h) if s else np.log1p(-h) for s in starts)
            cuts = [0] + [t + 1 for t, s in enumerate(starts) if s] + [n]
            lp += sum(_segments_logpdf(y[i, a:b], 0.09) for a, b in zip(cuts, cuts[1:]))
            total = np.logaddexp(total, lp)
        assert math.isclose(chain[i], total, abs_tol=1e-8)


def test_dirichlet_process_chain_rule():
    z = P.Zipf("z", "", {"s": 1.1, "alphas": [3.0], "order_seed": 7})
    x, _ = z.sample(3, np.random.default_rng(2))
    Pr, _ = z.predict(x)
    chain = np.log(P.take(Pr, x)).sum(1)
    p0, a = z.p0(), 3.0
    for i in range(3):
        # Dirichlet-multinomial with a discrete base: a fresh draw may repeat a seen symbol
        sym, counts = np.unique(x[i], return_counts=True)
        joint = (
            (special.gammaln(counts + a * p0[sym]) - special.gammaln(a * p0[sym])).sum()
            + special.gammaln(a)
            - special.gammaln(a + P.T)
        )
        assert math.isclose(chain[i], joint, abs_tol=1e-8)


@pytest.mark.parametrize("name", P.LADDER)
def test_every_world_samples_and_predicts(name):
    w = P.worlds()[name]
    x, h = w.sample(6, np.random.default_rng(3))
    assert x.shape == (6, P.T) and x.dtype == np.int16 and x.min() >= 0 and x.max() < P.NBINS
    Pr, ex = w.predict(x)
    assert Pr.shape == (6, P.T, P.NBINS) and np.allclose(Pr.sum(-1), 1) and (Pr >= 0).all()
    lp = np.log(P.take(Pr, x))
    assert np.isfinite(lp).all()
    assert np.isfinite(ex["entropy"]).all()
    # later positions are better predicted than the first, in every world: the prefix teaches something
    assert lp[:, 64:].mean() > lp[:, 0].mean()


def test_bayes_beats_a_misspecified_predictor():
    """Expected excess log loss over Bayes is KL >= 0: the 'level' predictor on smooth-function data loses."""
    ws = P.worlds()
    x, _ = ws["gp"].sample(16, np.random.default_rng(4))
    right = np.log(P.take(ws["gp"].predict(x)[0], x)).mean()
    wrong = np.log(P.take(ws["level"].predict(x)[0], x)).mean()
    assert right > wrong


def test_structural_uncertainty_resolves():
    w = P.worlds()["gp-hyper"]
    x, h = w.sample(24, np.random.default_rng(5))
    _, ex = w.predict(x)
    assert ex["hyp_entropy"][:, 0].mean() > 2.0 and ex["hyp_entropy"][:, -1].mean() < 0.5
    assert (ex["epi_var"] > -1e-9).all()


def test_values_rarely_reach_the_open_end_bins():
    """Bayes conditions on end bins as if exact, so every continuous world must keep values inside [-6, 6]."""
    rng = np.random.default_rng(0)
    for name, w in P.worlds().items():
        if name.startswith("zipf") or name == "mix":
            continue
        x, _ = w.sample(4000, rng)
        assert ((x == 0) | (x == P.NBINS - 1)).any(1).mean() < 2e-3, name


def test_zipf_bayes_averages_alpha_exactly():
    """The chain of the alpha-averaged predictive = log mean over alpha of the Dirichlet-multinomial joint."""
    from scipy.special import gammaln, logsumexp

    w = P.worlds()["zipf"]
    x, _ = w.sample(3, np.random.default_rng(1))
    p0, alphas = w.p0(), np.array(w.params["alphas"])
    chain = np.log(P.take(w.predict(x)[0], x)).sum(1)
    for i in range(3):
        c = np.bincount(x[i].astype(int), minlength=P.NBINS)
        joint = [gammaln(a) - gammaln(a + P.T) + (gammaln(c + a * p0) - gammaln(a * p0)).sum() for a in alphas]
        assert np.isclose(chain[i], logsumexp(joint) - np.log(len(alphas)), atol=1e-8)
