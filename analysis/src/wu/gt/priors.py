# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""The worlds: a sampler and an exact Bayes predictive for each.

Tokens. A real value y becomes one of 256 equal bins on [-6, 6] (width 3/64); the two end bins are open to
infinity. A sequence is 256 tokens. BOS = 256 is prepended for the model only.

Bayes predictive. For every prefix x_<t, p(x_t | x_<t) over the 256 bins: the predictive density
integrated over each bin. Validation data are drawn from the same prior, so the mean of
log p_Bayes(x_t) - log p_model(x_t) over tokens is an unbiased estimate of E KL(p_Bayes || p_model).

Approximations (identical for every model compared):
    - Gaussian worlds condition on bin centres, with the rounding variance width^2/12 added to the noise
      variance (Sheppard's correction: W. F. Sheppard, "On the calculation of the most probable values of
      frequency-constants, for data arranged according to equidistant division of a scale", Proc. London
      Math. Soc. 29:353-380, 1898). That is +7.3% at the smallest noise, 0.05^2. Checked against exact
      integration over the level for `level` with noise 0.05: KL(exact || this) <= 1e-6 nats per token.
      Hypothesis weights use exact bin probabilities of each hypothesis's (centre-conditioned) predictive,
      so every mixture is an exact Bayesian model average of those sequential predictors.
    - Every bin probability is floored at 1e-30 and renormalized (`_pad_cdf`).
    - The open end bins are conditioned on as if the value were the end bin's centre (censoring ignored).
      World parameters keep this rare: at most 0.015% of sequences reach an end bin (`tp`; tested). Before
      14 Nov (evening), `warped` had 7% and `tp` 0.4%, and there the predictive was measurably not Bayes.
    - With these approximations, mean log p_Bayes - log p_model estimates E KL(p_true || p_model) minus
      E KL(p_true || p_Bayes), the latter small except as above.

Uncertainty, per prefix:
    entropy     of the Bayes predictive over bins (nats): everything not knowable from the prefix
    alea_var    noise variance: what no amount of data would remove
    epi_var     posterior variance of the noise-free value f_t, including uncertainty over hypotheses:
                what more data would remove
    hyp_entropy entropy of the posterior over discrete hypotheses (kernels, regimes, concentrations, worlds)
    novel       (Zipf world) probability that the next symbol has not appeared yet

Worlds (parameters in `worlds()`):
    level      constant level plus Gaussian noise
    gp         GP, RBF kernel, known hyperparameters (predictive variance depends on position only)
    gp-hyper   GP with one of 6 kernels x 2 noise levels, unknown: Bayesian model averaging
               (Hoeting, Madigan, Raftery & Volinsky 1999, "Bayesian model averaging: a tutorial", Stat. Sci.)
    tp         Student-t process (Shah, Wilson & Ghahramani 2014, AISTATS, PMLR 33:877-885)
    warped     GP passed through (exp(0.4u) - 1)/0.4 (warped GP: Snelson, Rasmussen & Ghahramani,
               "Warped Gaussian processes", NIPS 16, 2004; here the warp is fixed and known)
    regimes    piecewise-constant level with changepoints; exact BOCPD (Adams & MacKay 2007,
               arXiv:0710.3742), every run length kept
    zipf       per document G ~ DP(alpha, G0), tokens ~ G; G0 Zipf(s) over 256 shuffled symbols;
               alpha unknown, from a finite set: Bayesian model averaging over alpha
    mix        each sequence from one of the 7 worlds above, unannounced
    zipf-*     zipf with other s or alpha sets (`VARIANTS`)
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import special
from scipy.linalg import solve_triangular

NBINS, LO, HI, T = 256, -6.0, 6.0, 256
DELTA = (HI - LO) / NBINS
BOS = NBINS
VOCAB = NBINS + 1
EDGES = LO + DELTA * np.arange(1, NBINS)  # 255 interior edges
CENTERS = LO + DELTA * (np.arange(NBINS) + 0.5)
Q = DELTA**2 / 12  # variance of rounding to a bin
TINY = 1e-300


def quantize(y: np.ndarray) -> np.ndarray:
    return np.clip(np.floor((y - LO) / DELTA), 0, NBINS - 1).astype(np.int16)


def _pad_cdf(c: np.ndarray) -> np.ndarray:
    """CDF at the 255 interior edges -> 256 bin probabilities."""
    z, o = np.zeros(c.shape[:-1] + (1,)), np.ones(c.shape[:-1] + (1,))
    p = np.maximum(np.diff(np.concatenate([z, c, o], -1), axis=-1), 1e-30)  # no bin is impossible: logs stay finite
    return p / p.sum(-1, keepdims=True)


def gauss_bins(m, s, edges=EDGES):
    return _pad_cdf(special.ndtr((edges - m[..., None]) / s[..., None]))


def student_bins(m, scale, dof, edges=EDGES):
    return _pad_cdf(special.stdtr(dof[..., None], (edges - m[..., None]) / scale[..., None]))


def take(P: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.take_along_axis(P, x[..., None].astype(np.int64), -1)[..., 0]


def seen_before(x: np.ndarray) -> np.ndarray:
    """(n, T, 256) bool: symbol k occurred at a position < t of the same sequence."""
    onehot = np.eye(NBINS, dtype=bool)[x.astype(np.int64)]
    return np.concatenate([np.zeros((len(x), 1, NBINS), bool), np.cumsum(onehot, 1)[:, :-1] > 0], 1)


def entropy(P: np.ndarray) -> np.ndarray:
    return -(P * np.log(np.maximum(P, TINY))).sum(-1)


# ── kernels on the integer grid 0..T-1 ────────────────────────────────────────────────────────────


def kernel(kind: str, n: int = T, ell: float = 1.0, amp: float = 1.0, period: float = 1.0) -> np.ndarray:
    d = np.subtract.outer(np.arange(n), np.arange(n)).astype(float)
    if kind == "const":
        return amp**2 * np.ones((n, n))
    if kind == "rbf":
        return amp**2 * np.exp(-(d**2) / (2 * ell**2))
    if kind == "matern12":
        return amp**2 * np.exp(-np.abs(d) / ell)
    if kind == "periodic":
        return amp**2 * np.exp(-2 * np.sin(np.pi * d / period) ** 2 / ell**2)
    raise ValueError(kind)


def gauss_sequential(Y: np.ndarray, K: np.ndarray, noise) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-step-ahead Gaussian predictive for every position of every sequence, from one Cholesky factor.

    Y (n, T) observed; covariance C = K + diag(noise). With C = L L^T and z = L^-1 y, the conditional of y_t
    given y_<t has mean sum_{j<t} L_tj z_j = y_t - L_tt z_t and variance L_tt^2. Returns (mean, var, z).
    (Cholesky form of GP prediction: Rasmussen & Williams 2006, Gaussian Processes for Machine Learning,
    Algorithm 2.1, applied to every prefix.) Row t of L below the diagonal does not involve noise_t, so a
    data-dependent noise_t (Warped) does not leak y_t into its own predictive.
    `noise` is a scalar (one factor shared by all sequences) or an (n, T) array (one factor per sequence)."""
    n = Y.shape[1]
    if np.ndim(noise) == 0:
        L = np.linalg.cholesky(K + noise * np.eye(n))
        z = solve_triangular(L, Y.T, lower=True).T
        d = np.diag(L)[None, :]
    else:
        L = np.linalg.cholesky(K[None] + noise[:, :, None] * np.eye(n)[None])
        z = np.stack([solve_triangular(L[i], Y[i], lower=True) for i in range(len(Y))])
        d = np.diagonal(L, axis1=1, axis2=2)
    return Y - z * d, d**2, z


def mixture(P: np.ndarray, x: np.ndarray, log_prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian model averaging. P (H, n, T, B) per-hypothesis predictives, x (n, T) the tokens.
    The weight of hypothesis h at position t is its prior times the probability it gave tokens 0..t-1.
    Returns (mixture predictive (n, T, B), posterior weights (H, n, T))."""
    ll = np.log(np.maximum(take(P, x[None]), TINY))  # (H, n, T)
    cum = np.concatenate([np.zeros(ll.shape[:-1] + (1,)), np.cumsum(ll, -1)[..., :-1]], -1)
    lw = log_prior[:, None, None] + cum
    w = np.exp(lw - special.logsumexp(lw, axis=0, keepdims=True))
    return np.einsum("hnt,hntb->ntb", w, P), w


def _weighted_moments(w, means, variances):
    """Mixture mean and variance over the leading axis."""
    mu = (w * means).sum(0)
    return mu, (w * (variances + means**2)).sum(0) - mu**2


# ── worlds ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class World:
    name: str
    blurb: str
    params: dict = field(default_factory=dict)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(tokens (n, T) int16, latent hypothesis id per sequence (n,))."""
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Bayes predictive over bins for every position, (n, T, 256), and uncertainty extras (n, T)."""
        raise NotImplementedError


@dataclass
class Gaussian(World):
    """A GP with a fixed kernel: y = f + noise, f ~ GP(0, K). (`level` is the GP with a constant kernel.)"""

    def _K(self):
        p = self.params
        return kernel(p["kernel"], ell=p.get("ell", 1.0), amp=p.get("amp", 1.0), period=p.get("period", 1.0))

    def sample(self, n, rng):
        K, s = self._K(), self.params["sigma"]
        L = np.linalg.cholesky(K + 1e-6 * np.eye(T))
        y = rng.standard_normal((n, T)) @ L.T + s * rng.standard_normal((n, T))
        return quantize(y), np.zeros(n, dtype=np.int16)

    def gaussian_parts(self, x):
        """Per position: predictive mean and variance of the next value, and the noise variance."""
        s2 = self.params["sigma"] ** 2
        m, v, _ = gauss_sequential(CENTERS[x], self._K(), s2 + Q)
        return m, v - Q, np.full_like(m, s2)

    def predict(self, x):
        m, v, s2 = self.gaussian_parts(x)
        P = gauss_bins(m, np.sqrt(v))
        return P, {"entropy": entropy(P), "alea_var": s2, "epi_var": v - s2, "hyp_entropy": np.zeros_like(m)}


@dataclass
class GaussianMixture(World):
    """Several Gaussian worlds, one drawn per sequence with equal prior; Bayes averages over them."""

    parts: list = field(default_factory=list)

    def sample(self, n, rng):
        h = rng.integers(0, len(self.parts), n)
        x = np.zeros((n, T), dtype=np.int16)
        for i, part in enumerate(self.parts):
            if (h == i).any():
                x[h == i] = part.sample(int((h == i).sum()), rng)[0]
        return x, h.astype(np.int16)

    def predict(self, x):
        parts = [p.gaussian_parts(x) for p in self.parts]
        M, V, S2 = (np.stack(a) for a in zip(*parts))
        P = gauss_bins(M, np.sqrt(V))
        mix, w = mixture(P, x, np.full(len(self.parts), -np.log(len(self.parts))))
        mu, var = _weighted_moments(w, M, V)
        alea = (w * S2).sum(0)
        return mix, {
            "entropy": entropy(mix),
            "alea_var": alea,
            "epi_var": var - alea,
            "hyp_entropy": entropy(np.moveaxis(w, 0, -1)),
        }


@dataclass
class StudentT(World):
    """Student-t process (Shah, Wilson & Ghahramani 2014, "Student-t processes as alternatives to Gaussian
    processes", AISTATS), covariance parametrization: y ~ MVT(nu, 0, C), C = K + s^2 I. Equivalent to a GP
    whose overall scale is drawn per sequence, so the prefix teaches the scale. Conditional of y_t given
    y_<t: MVT(nu + t, m, (nu + beta - 2)/(nu + t - 2) * v), beta = y_<t^T C_<t^-1 y_<t (covariance form).
    The rounding variance Q is added to the predictive variance unscaled: var = kappa (v - Q) + Q. It still
    enters the conditioning through C (approximation, below 2e-5 nats per token)."""

    def _C(self):
        p = self.params
        return kernel("rbf", ell=p["ell"], amp=p["amp"]) + p["sigma"] ** 2 * np.eye(T)

    def sample(self, n, rng):
        nu = self.params["nu"]
        L = np.linalg.cholesky(self._C())
        g = rng.chisquare(nu, n)
        y = (rng.standard_normal((n, T)) @ L.T) * np.sqrt((nu - 2) / g)[:, None]
        return quantize(y), np.zeros(n, dtype=np.int16)

    def sequential(self, Y, extra_noise=Q):
        """Per position: Student-t mean, scale and dof of the next value, the Gaussian conditional variance v
        (of C plus `extra_noise`), and kappa, the posterior factor on the scale learned from the prefix."""
        nu = self.params["nu"]
        m, v, z = gauss_sequential(Y, self._C(), extra_noise)
        beta = np.concatenate([np.zeros((len(Y), 1)), np.cumsum(z**2, -1)[:, :-1]], -1)
        t = np.arange(Y.shape[1])[None, :]
        dof = (nu + t) + 0.0 * m
        kappa = (nu + beta - 2) / (nu + t - 2)
        var = kappa * (v - extra_noise) + extra_noise  # the rounding variance is not scaled by kappa
        return m, np.sqrt(var * (dof - 2) / dof), dof, v, kappa

    def predict(self, x):
        m, scale, dof, v, kappa = self.sequential(CENTERS[x])
        P = student_bins(m, scale, dof)
        s2 = self.params["sigma"] ** 2
        return P, {
            "entropy": entropy(P),
            "alea_var": kappa * s2,
            "epi_var": kappa * (v - s2 - Q),
            "hyp_entropy": np.zeros_like(m),
        }


@dataclass
class Warped(World):
    """A GP seen through a monotone warp: u = f + noise, y = (exp(lam u) - 1) / lam. Skewed, bounded below
    by -1/lam. Bayes works in u: bin edges are mapped back through the inverse warp (change of variables:
    P(y in [a, b)) = P(u in [g^-1(a), g^-1(b))), g increasing). Rounding variance is mapped by the slope of
    g^-1 (delta method); it exceeds 9% of the latent noise variance only for u < -3. Tokens in the open top
    bin (y > 6) are conditioned on as its centre; with amp 0.6 this happens in 0.005% of sequences."""

    def g_inv(self, y):
        lam = self.params["lam"]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(1 + lam * y > 0, np.log(np.maximum(1 + lam * y, TINY)) / lam, -np.inf)

    def sample(self, n, rng):
        p = self.params
        L = np.linalg.cholesky(kernel("rbf", ell=p["ell"], amp=p["amp"]) + 1e-6 * np.eye(T))
        u = rng.standard_normal((n, T)) @ L.T + p["sigma"] * rng.standard_normal((n, T))
        return quantize((np.exp(p["lam"] * u) - 1) / p["lam"]), np.zeros(n, dtype=np.int16)

    def predict(self, x):
        p = self.params
        lam, s2 = p["lam"], p["sigma"] ** 2
        c = np.maximum(CENTERS[x], -1 / lam + 1e-3)
        u = self.g_inv(c)
        q = (DELTA / (1 + lam * c)) ** 2 / 12  # rounding error, mapped into u by the warp's slope
        m, v, _ = gauss_sequential(u, kernel("rbf", ell=p["ell"], amp=p["amp"]), s2 + q)
        v = v - q
        P = gauss_bins(m, np.sqrt(v), edges=self.g_inv(EDGES))
        return P, {
            "entropy": entropy(P),
            "alea_var": np.full_like(m, s2),
            "epi_var": v - s2,
            "hyp_entropy": np.zeros_like(m),
        }


@dataclass
class Regimes(World):
    """Piecewise-constant levels. Before each value after the first, a new segment starts with probability h;
    each segment's level ~ N(0, 1); y = level + noise. Bayes: online changepoint detection (Adams & MacKay
    2007, "Bayesian online changepoint detection", arXiv:0710.3742) with constant hazard and a conjugate
    Normal level, exact: every run length is kept (K = T + 1). Keeping only the 48 likeliest would leave a mean KL
    of 1.5e-6 nats but up to 0.035 nats in single log-probabilities, so we don't prune."""

    K: int = T + 1

    def sample(self, n, rng):
        p = self.params
        new = rng.random((n, T)) < p["hazard"]
        new[:, 0] = True
        seg = np.cumsum(new, 1) - 1
        levels = rng.standard_normal((n, T))
        y = np.take_along_axis(levels, seg, 1) + p["sigma"] * rng.standard_normal((n, T))
        return quantize(y), np.zeros(n, dtype=np.int16)

    def run(self, Y, x=None, density=False):
        """Run-length filtering. With `density`, likelihoods are densities of the values Y (for tests) and
        the return is the log density of each value given the past; otherwise bins x are scored and the
        predictive over bins is returned with the extras."""
        p = self.params
        h, s2 = p["hazard"], p["sigma"] ** 2
        s2c = s2 if density else s2 + Q
        n, nT = Y.shape
        K = self.K
        # components: log weight, count, sum of values; slot 0 starts as the first segment
        lw = np.full((n, K), -np.inf)
        lw[:, 0] = 0.0
        cnt = np.zeros((n, K))
        sm = np.zeros((n, K))
        out_lp = np.zeros((n, nT))
        P_all = None if density else np.zeros((n, nT, NBINS))
        ex = {k: np.zeros((n, nT)) for k in ("alea_var", "epi_var", "hyp_entropy")}
        for t in range(nT):
            if t > 0:  # transition: every run grows, or a new one starts with mass h
                total = special.logsumexp(lw, axis=1)
                lw = np.concatenate([(total + np.log(h))[:, None], lw + np.log1p(-h)], 1)
                cnt = np.concatenate([np.zeros((n, 1)), cnt], 1)
                sm = np.concatenate([np.zeros((n, 1)), sm], 1)
                keep = np.argsort(-lw, axis=1)[:, :K]
                lw, cnt, sm = (np.take_along_axis(a, keep, 1) for a in (lw, cnt, sm))
            w = np.exp(lw - special.logsumexp(lw, axis=1, keepdims=True))
            prec = 1 + cnt / s2c
            mean = sm / s2c / prec
            var_mu = 1 / prec
            pv = var_mu + s2
            if density:
                lik = -0.5 * (np.log(2 * np.pi * pv) + (Y[:, t : t + 1] - mean) ** 2 / pv)
                out_lp[:, t] = special.logsumexp(lw + lik, axis=1) - special.logsumexp(lw, axis=1)
            else:
                Pk = gauss_bins(mean, np.sqrt(pv))  # (n, K, B)
                P = np.einsum("nk,nkb->nb", w, Pk)
                P_all[:, t] = P / P.sum(-1, keepdims=True)
                lik = np.log(np.maximum(take(Pk, np.repeat(x[:, t : t + 1], K, 1)), TINY))
                mu, var = _weighted_moments(w.T, mean.T, var_mu.T)
                ex["alea_var"][:, t], ex["epi_var"][:, t] = s2, var
                # uncertainty over *when* the current regime began, in nats of run length
                ex["hyp_entropy"][:, t] = entropy(w)
            lw = np.where(np.isfinite(lw), lw + lik, -np.inf)
            cnt, sm = cnt + 1, sm + Y[:, t : t + 1]
        return out_lp if density else (P_all, ex)

    def predict(self, x):
        P, ex = self.run(CENTERS[x], x)
        return P, {"entropy": entropy(P), **ex}


@dataclass
class Zipf(World):
    """Per document G ~ DP(alpha, G0), tokens i.i.d. from G (sampled as a Chinese restaurant process).
    G0(symbol of rank r) proportional to r^-s, ranks assigned to the 256 symbols by a fixed permutation.
    alpha is drawn uniformly from params["alphas"]. Predictive given alpha: (counts + alpha G0) / (t + alpha),
    the Polya urn of Blackwell & MacQueen (1973, "Ferguson distributions via Polya urn schemes", Ann. Statist.
    1:353-355). With a discrete G0 on 256 symbols, G's weights on them are Dirichlet(alpha G0) (Ferguson 1973,
    "A Bayesian analysis of some nonparametric problems", Ann. Statist. 1:209-230), so the joint is
    Dirichlet-multinomial and a fresh draw may repeat a seen symbol; the CRP sampler draws from the same urn."""

    def p0(self):
        p = self.params
        r = np.arange(1, NBINS + 1, dtype=float) ** -p["s"]
        order = np.random.default_rng(p["order_seed"]).permutation(NBINS)
        out = np.zeros(NBINS)
        out[order] = r / r.sum()
        return out

    def sample(self, n, rng):
        alphas = np.array(self.params["alphas"], dtype=float)
        h = rng.integers(0, len(alphas), n)
        a = alphas[h]
        x = np.zeros((n, T), dtype=np.int16)
        base = rng.choice(NBINS, size=(n, T), p=self.p0())
        for t in range(T):  # Chinese restaurant: a new draw from G0, or a copy of an earlier token
            fresh = rng.random(n) < a / (a + t)
            copy = x[np.arange(n), rng.integers(0, max(t, 1), n)]
            x[:, t] = np.where(fresh | (t == 0), base[:, t], copy)
        return x, h.astype(np.int16)

    def predict(self, x):
        alphas = np.array(self.params["alphas"], dtype=float)
        p0 = self.p0()
        onehot = np.eye(NBINS)[x.astype(np.int64)]
        counts = np.concatenate([np.zeros((len(x), 1, NBINS)), np.cumsum(onehot, 1)[:, :-1]], 1)
        t = np.arange(T)[None, :, None]
        P = np.stack([(counts + a * p0) / (t + a) for a in alphas])
        mix, w = mixture(P, x, np.full(len(alphas), -np.log(len(alphas))))
        seen = counts > 0
        novel = (mix * ~seen).sum(-1)
        nan = np.full(x.shape, np.nan)
        return mix, {
            "entropy": entropy(mix),
            "alea_var": nan,
            "epi_var": nan,
            "hyp_entropy": entropy(np.moveaxis(w, 0, -1)),
            "novel": novel,
        }


@dataclass
class Mix(World):
    """A corpus of genres: each sequence comes from one of the worlds, unannounced."""

    worlds: list = field(default_factory=list)

    def sample(self, n, rng):
        h = rng.integers(0, len(self.worlds), n)
        x = np.zeros((n, T), dtype=np.int16)
        for i, w in enumerate(self.worlds):
            if (h == i).any():
                x[h == i] = w.sample(int((h == i).sum()), rng)[0]
        return x, h.astype(np.int16)

    def predict(self, x):
        outs = [w.predict(x) for w in self.worlds]
        P = np.stack([o[0] for o in outs])
        mix, w = mixture(P, x, np.full(len(self.worlds), -np.log(len(self.worlds))))
        nan = np.full(x.shape, np.nan)
        return mix, {
            "entropy": entropy(mix),
            "alea_var": nan,
            "epi_var": nan,
            "hyp_entropy": entropy(np.moveaxis(w, 0, -1)),
        }


# ── the ladder ───────────────────────────────────────────────────────────────────────────────────


def _gp(kind, sigma, **kw):
    return Gaussian(f"{kind}", "", {"kernel": kind, "sigma": sigma, **kw})


def worlds() -> dict[str, World]:
    hyper_kernels = [
        dict(kind="rbf", ell=3),
        dict(kind="rbf", ell=10),
        dict(kind="rbf", ell=30),
        dict(kind="matern12", ell=10),
        dict(kind="periodic", period=12, ell=1.0),
        dict(kind="periodic", period=30, ell=1.0),
    ]
    parts = [
        _gp(k["kind"], s, **{a: b for a, b in k.items() if a != "kind"}) for k in hyper_kernels for s in (0.05, 0.3)
    ]
    ladder = {
        "level": Gaussian("level", "a constant level ~ N(0,1) plus noise 0.3", {"kernel": "const", "sigma": 0.3}),
        "gp": Gaussian(
            "gp", "smooth function: RBF GP, lengthscale 8, noise 0.1", {"kernel": "rbf", "ell": 8, "sigma": 0.1}
        ),
        "gp-hyper": GaussianMixture(
            "gp-hyper",
            "one of 6 kernels (RBF 3/10/30, Matern-1/2, periodic 12/30) x noise 0.05/0.3",
            {"n": len(parts)},
            parts,
        ),
        # amplitudes keep values inside [-6, 6]: sequences reaching an end bin, tp 0.015%, warped 0.005%
        "tp": StudentT(
            "tp",
            "Student-t process, nu 5, RBF lengthscale 8, amplitude 0.4, noise 0.1",
            {"nu": 5.0, "ell": 8, "amp": 0.4, "sigma": 0.1},
        ),
        "warped": Warped(
            "warped",
            "RBF GP (lengthscale 8, amplitude 0.6) seen through (exp(0.4u)-1)/0.4, latent noise 0.15",
            {"lam": 0.4, "ell": 8, "amp": 0.6, "sigma": 0.15},
        ),
        "regimes": Regimes(
            "regimes",
            "levels ~ N(0,1) that jump with probability 1/40 per step, noise 0.25",
            {"hazard": 1 / 40, "sigma": 0.25},
        ),
        "zipf": Zipf(
            "zipf",
            "DP(alpha, Zipf(1.1) over 256 shuffled symbols), alpha in {1, 5, 25}",
            {"s": 1.1, "alphas": [1.0, 5.0, 25.0], "order_seed": 7},
        ),
    }
    ladder["mix"] = Mix("mix", "each sequence from one of the 7 worlds above, unannounced", {}, list(ladder.values()))
    # Zipf variants. s: exponent of G0. alphas: DP concentrations (small = many repeats; 1e9 = i.i.d. from G0).
    z = lambda name, blurb, s_, alphas: Zipf(name, blurb, {"s": s_, "alphas": alphas, "order_seed": 7})
    ladder["zipf-s0.7"] = z("zipf-s0.7", "as zipf, flatter base: Zipf(0.7)", 0.7, [1.0, 5.0, 25.0])
    ladder["zipf-s1.5"] = z("zipf-s1.5", "as zipf, steeper base: Zipf(1.5)", 1.5, [1.0, 5.0, 25.0])
    ladder["zipf-bursty"] = z(
        "zipf-bursty", "as zipf, alpha in {0.25, 0.5, 1}: documents repeat a few symbols", 1.1, [0.25, 0.5, 1.0]
    )
    ladder["zipf-fresh"] = z(
        "zipf-fresh", "as zipf, alpha in {50, 100, 200}: most tokens new to the document", 1.1, [50.0, 100.0, 200.0]
    )
    ladder["zipf-iid"] = z(
        "zipf-iid",
        "i.i.d. draws from Zipf(1.1): nothing to learn from context, all knowledge in the weights",
        1.1,
        [1e9],
    )
    return ladder


LADDER = ["level", "gp", "gp-hyper", "tp", "warped", "regimes", "zipf", "mix"]
VARIANTS = ["zipf-s0.7", "zipf-s1.5", "zipf-bursty", "zipf-fresh", "zipf-iid"]
ALL = LADDER + VARIANTS  # a world's place here fixes its data seeds
