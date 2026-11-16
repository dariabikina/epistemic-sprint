# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Intervals: 95% percentile cluster bootstrap over sequences.

Sequences are resampled with replacement and all positions of a resampled sequence are kept. Paired
comparisons pass the per-position difference. B = 1,000 resamples, generator seed 0.
Training-seed variation is not included here; `control.across_seeds` reports it separately.

References. Percentile bootstrap: Efron, B., Tibshirani, R. J. (1993) An Introduction to the Bootstrap,
Chapman & Hall, ch. 13; Davison, A. C., Hinkley, D. V. (1997) Bootstrap Methods and their Application, CUP,
ch. 5 (BCa, a second-order alternative: Efron, B. (1987) Better bootstrap confidence intervals, JASA 82,
171-185). Resampling whole clusters ("cluster bootstrap"): Davison & Hinkley (1997) sec. 3.8; Field, C. A.,
Welsh, A. H. (2007) Bootstrapping clustered data, JRSS B 69, 369-390. AUC as the Mann-Whitney statistic:
Hanley, J. A., McNeil, B. J. (1982) The meaning and use of the area under a ROC curve, Radiology 143, 29-36.
`r2_over` keeps the cross-fitted predictions fixed, so its interval omits refitting variability; see Bates,
S., Hastie, T., Tibshirani, R. (2024) Cross-validation: what does it estimate and how well does it do it?,
JASA 119, 1434-1445.
"""

import numpy as np
from scipy.stats import rankdata

B_DEFAULT, B_AUC, LEVEL = 1000, 1000, 0.95


def resample_sequences(n_seq: int, B: int, seed: int) -> np.ndarray:
    """(B, n_seq) sequence indices, drawn with replacement."""
    return np.random.default_rng(seed).integers(0, n_seq, (B, n_seq))


def interval(estimate: float, draws: np.ndarray) -> dict:
    lo, hi = np.nanquantile(draws, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
    return {"est": float(estimate), "lo": float(lo), "hi": float(hi)}


def mean_over(values: np.ndarray, mask: np.ndarray | None = None, B: int = B_DEFAULT, seed: int = 0) -> dict:
    """Mean of `values` (n_seq, T) over the positions in `mask`, with a cluster-bootstrap interval.
    The estimator is a ratio (sum over selected positions / their count), resampled by sequence."""
    if mask is None:
        mask = np.ones(values.shape, bool)
    num = np.where(mask, values, 0.0).sum(1)  # per sequence
    den = mask.sum(1).astype(float)
    if den.sum() == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan")}
    idx = resample_sequences(len(num), B, seed)
    with np.errstate(invalid="ignore"):
        draws = num[idx].sum(1) / den[idx].sum(1)
    return interval(num.sum() / den.sum(), draws)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties counted half (Mann-Whitney)."""
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_over(score: np.ndarray, label: np.ndarray, mask: np.ndarray, B: int = B_AUC, seed: int = 0) -> dict:
    """AUC over positions in `mask` (arrays (n_seq, T)), with a cluster-bootstrap interval by sequence."""
    est = auc(score[mask], label[mask])
    draws = np.array([auc(score[i][mask[i]], label[i][mask[i]]) for i in resample_sequences(len(score), B, seed)])
    return interval(est, draws)


def r2_over(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, B: int = B_DEFAULT, seed: int = 0) -> dict:
    """R^2 of cross-fitted predictions, with an interval from resampling the groups (sequences)."""

    def r2(yy, pp):
        return 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()

    uniq = np.unique(groups)
    members = [np.nonzero(groups == g)[0] for g in uniq]
    draws = []
    for idx in resample_sequences(len(uniq), B, seed):
        rows = np.concatenate([members[j] for j in idx])
        draws.append(r2(y[rows], pred[rows]))
    return interval(r2(y, pred), np.array(draws))


def fmt(ci: dict, digits: int = 3, sign: bool = False) -> str:
    """'0.035 [0.031, 0.040]'."""
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"{f.format(ci['est'])} [{f.format(ci['lo'])}, {f.format(ci['hi'])}]"
