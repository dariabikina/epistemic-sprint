# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Measures of a compressed Zipf-world model against exact Bayes, paired with the base model.

Inputs: per-position arrays from `compress.evaluate` for the base and the compressed model on the same
validation sequences (n, T). All quantities are in nats. Means carry a cluster-bootstrap interval (stats.py);
differences are taken per position before averaging.

Token kinds. A token is *new* if its symbol has not occurred earlier in the sequence, else a *repeat*.
Rank = the symbol's rank in G0 (0 = most probable).

    kl, dkl                  KL(p_Bayes || p_comp) per position, and KL(comp) - KL(base). Exact sums over the
                             256 symbols; no sampling of tokens.
    excess_new/_repeat       log p_Bayes(x_t) - log p_comp(x_t), mean over new / repeat tokens.
    d_excess                 change in excess, compressed minus base, over all positions; d_excess_new/_repeat
                             the same over new / repeat tokens.
    d_excess_kind            change in excess on the binary event new/repeat: log p(kind) with
                             p(new) = mass on unseen symbols. Mean over all positions.
    d_excess_which           change in excess on log p(symbol | kind), mean over all positions;
                             d_excess = d_excess_kind + d_excess_which exactly. d_excess_which_new/_repeat:
                             the same over new / repeat tokens only. In expectation this is the chain rule for
                             relative entropy (Cover & Thomas 2006, Elements of Information Theory, 2nd ed.,
                             Thm 2.5.3) applied to (kind, symbol).
    pnew_bias                mean of p_comp(new) - p_Bayes(new).
    d_which_new_by_rank      change in log p(symbol | new) on new tokens, compressed minus base, by G0 rank
                             band (RANK_BANDS). Sign: + = higher log p (opposite to the d_excess_* columns).
    prior_kl                 KL(G0 || q1), q1 = the model's predictive at the BOS position. No interval.
    prior_log_ratio_by_rank  mean log(q1 / G0) per rank band. No interval.
    entropy_gap(_new/_repeat)  H(p_comp) - H(p_Bayes); d_entropy_gap = compressed minus base.
    tracking_auc             AUC separating the 10% of positions with the largest dkl from the 50% with the
                             smallest, scored by H(p_comp) residualized on H(p_base) (least squares, one line
                             over all positions). As scripts/tracking.py for the LLMs.
    corr_dH_dKL              Pearson correlation over positions of H(comp) - H(base) with dkl.
"""

import numpy as np

from . import priors as P
from .stats import B_AUC, auc_over, interval, mean_over, resample_sequences

RANK_BANDS = [(0, 8), (8, 32), (32, 96), (96, 256)]


def token_kinds(x: np.ndarray, p0: np.ndarray) -> dict:
    """Per position: whether the token is new to its document, and its symbol's rank in G0 (0 = commonest)."""
    seen = P.seen_before(x)
    rank_of = np.empty(P.NBINS, int)
    rank_of[np.argsort(-p0)] = np.arange(P.NBINS)
    return {"new": ~P.take(seen, x), "rank": rank_of[x.astype(np.int64)]}


def split_logp(e: dict, new: np.ndarray, model: bool) -> tuple[np.ndarray, np.ndarray]:
    """log p(actual) = log p(kind) + log p(symbol | kind), for the model's or the Bayes predictive."""
    logp = e["logp"] if model else e["bayes_logp"]
    p_new = (e["novel_model"] if model else e["novel_bayes"]).astype(np.float64)
    log_kind = np.log(np.clip(np.where(new, p_new, 1 - p_new), 1e-30, None))
    return log_kind, logp - log_kind


def _corr_over(a, b, B=B_AUC, seed=0) -> dict:
    est = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    draws = [np.corrcoef(a[i].ravel(), b[i].ravel())[0, 1] for i in resample_sequences(len(a), B, seed)]
    return interval(est, np.array(draws))


def summarize(base: dict, comp: dict, x: np.ndarray, p0: np.ndarray) -> dict:
    k = token_kinds(x, p0)
    new, rep = k["new"], ~k["new"]
    excess = {name: -e["logp"] + e["bayes_logp"] for name, e in (("base", base), ("comp", comp))}
    kind_b, which_b = split_logp(base, new, model=True)
    kind_c, which_c = split_logp(comp, new, model=True)
    kind_bayes, which_bayes = split_logp(comp, new, model=False)
    dkl = (comp["kl"] - base["kl"]).astype(np.float64)
    gap = {name: e["entropy"] - e["bayes_entropy"] for name, e in (("base", base), ("comp", comp))}
    q1 = np.maximum(comp["q1"], 1e-30)
    order = np.argsort(-p0)

    out = {
        "kl": mean_over(comp["kl"]),
        "dkl": mean_over(dkl),
        "excess_new": mean_over(excess["comp"], new),
        "excess_repeat": mean_over(excess["comp"], rep),
        "d_excess": mean_over(excess["comp"] - excess["base"]),
        "d_excess_new": mean_over(excess["comp"] - excess["base"], new),
        "d_excess_repeat": mean_over(excess["comp"] - excess["base"], rep),
        "d_excess_kind": mean_over((kind_bayes - kind_c) - (kind_bayes - kind_b)),
        "d_excess_which": mean_over(which_b - which_c),
        "d_excess_which_new": mean_over(which_b - which_c, new),
        "d_excess_which_repeat": mean_over(which_b - which_c, rep),
        "pnew_bias": mean_over(comp["novel_model"] - comp["novel_bayes"]),
        "d_which_new_by_rank": {
            f"{a + 1}-{b}": mean_over(which_c - which_b, new & (k["rank"] >= a) & (k["rank"] < b))
            for a, b in RANK_BANDS
        },
        "prior_kl": float((p0 * np.log(p0 / q1)).sum()),
        "prior_log_ratio_by_rank": {
            f"{a + 1}-{b}": float(np.log(q1[order[a:b]] / p0[order[a:b]]).mean()) for a, b in RANK_BANDS
        },
        "entropy_gap": mean_over(gap["comp"]),
        "entropy_gap_new": mean_over(gap["comp"], new),
        "entropy_gap_repeat": mean_over(gap["comp"], rep),
        "d_entropy_gap": mean_over(gap["comp"] - gap["base"]),
        "share_new": float(new.mean()),
    }
    if dkl.std() > 0:
        # residualize the compressed entropy on the base entropy (one line through all positions)
        A = np.c_[base["entropy"].ravel(), np.ones(dkl.size)]
        coef = np.linalg.lstsq(A, comp["entropy"].ravel(), rcond=None)[0]
        resid = comp["entropy"] - (coef[0] * base["entropy"] + coef[1])
        hi, lo = np.quantile(dkl, 0.9), np.quantile(dkl, 0.5)
        out["tracking_auc"] = auc_over(resid, dkl >= hi, (dkl >= hi) | (dkl <= lo))
        out["corr_dH_dKL"] = _corr_over(comp["entropy"] - base["entropy"], dkl)
    else:
        nan = {"est": float("nan"), "lo": float("nan"), "hi": float("nan")}
        out["tracking_auc"] = out["corr_dH_dKL"] = nan
    return out
