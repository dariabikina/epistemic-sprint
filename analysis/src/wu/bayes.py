# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Uncertainty for the RQ1/RQ2 numbers: how much of each number is the model, and how much is the sample of questions.

Each question has a *paired* outcome under the base model and the compressed model: both right (11), lost (10),
gained (01), both wrong (00). Within a popularity group those four counts are multinomial, and a uniform
Dirichlet(1, 1, 1, 1) prior gives a Dirichlet posterior over the four cell probabilities. Every metric is
a function of the cells, so posterior draws of the cells give posterior draws of the metric:

    Acc_0 = p11 + p10,  Acc_c = p11 + p01,  r_g = Acc_c / Acc_0,  LR_g = p10 / (p11 + p10)
    r     = the same over all groups, weighted by group size (fixed by design)
    RS_g  = (r_g - r) x 100

Pairing matters: the two models answer the same questions, so the uncertainty of a *difference* between
them is much smaller than the uncertainty of either accuracy alone.

The "value" of an eval for a metric is how much it narrows that metric's posterior. `width_by_n` measures
how the 95% interval narrows as questions are added, so we can see what more data would buy.
"""

import numpy as np

from .data import BUCKETS

DRAWS, SEED = 20000, 2026


def cells(base: list[dict], comp: list[dict], idx) -> np.ndarray:
    b = np.array([base[i]["correct"] for i in idx], dtype=bool)
    c = np.array([comp[i]["correct"] for i in idx], dtype=bool)
    return np.array([(b & c).sum(), (b & ~c).sum(), (~b & c).sum(), (~b & ~c).sum()], dtype=float)


def posterior(
    base: list[dict], comp: list[dict], draws: int = DRAWS, seed: int = SEED, subsample: int | None = None
) -> dict:
    """Posterior draws of r, r_g, RS_g, LR_g and RS_tail - RS_head. `subsample`: use only that many questions per group."""
    rng = np.random.default_rng(seed)
    groups = {g: [i for i, r in enumerate(base) if r["bucket"] == g] for g in BUCKETS}
    if subsample:
        groups = {g: list(rng.choice(ix, size=min(subsample, len(ix)), replace=False)) for g, ix in groups.items()}
    n = {g: len(ix) for g, ix in groups.items()}
    p = {g: rng.dirichlet(cells(base, comp, ix) + 1, size=draws) for g, ix in groups.items()}  # columns: 11, 10, 01, 00
    out = {}
    w = {g: n[g] / sum(n.values()) for g in BUCKETS}
    acc0 = sum(w[g] * (p[g][:, 0] + p[g][:, 1]) for g in BUCKETS)
    accc = sum(w[g] * (p[g][:, 0] + p[g][:, 2]) for g in BUCKETS)
    out["r"] = accc / acc0
    for g in BUCKETS:
        out[f"r_{g}"] = (p[g][:, 0] + p[g][:, 2]) / (p[g][:, 0] + p[g][:, 1])
        out[f"RS_{g}"] = (out[f"r_{g}"] - out["r"]) * 100
        out[f"LR_{g}"] = p[g][:, 1] / (p[g][:, 0] + p[g][:, 1])
    out["RS_tail-head"] = out["RS_tail"] - out["RS_head"]
    out["LR_head-tail"] = out["LR_head"] - out["LR_tail"]
    return out


def summarize(draws: np.ndarray) -> dict:
    lo, med, hi = np.quantile(draws, [0.025, 0.5, 0.975])
    return {"median": float(med), "lo": float(lo), "hi": float(hi), "p_positive": float((draws > 0).mean())}


def width_by_n(base: list[dict], comp: list[dict], key: str, ns=(125, 250, 500, 1000), repeats: int = 5) -> dict:
    """Mean width of the 95% interval of `key` when only n questions per group are used."""
    out = {}
    for n in ns:
        widths = []
        for k in range(repeats):
            s = summarize(posterior(base, comp, draws=4000, seed=SEED + k, subsample=n)[key])
            widths.append(s["hi"] - s["lo"])
        out[n] = float(np.mean(widths))
    return out


def perplexity_spread(
    base_windows: list[dict], comp_windows: list[dict] | None, k: int = 4, draws: int = 5000, seed: int = SEED
) -> dict:
    """How much the paper's protocol (k = 4 windows of 2,048 tokens) depends on *which* text is read: perplexity
    on random k-window subsets of the windows we scored, and, for a compressed model, its ratio to the base
    on the same subset (paired: both models read the same tokens)."""
    rng = np.random.default_rng(seed)
    nb = np.array([w["nll"] for w in base_windows])
    tb = np.array([w["tokens"] for w in base_windows])
    picks = np.array([rng.choice(len(nb), size=k, replace=False) for _ in range(draws)])
    ppl_b = np.exp(nb[picks].sum(1) / tb[picks].sum(1))
    out = {
        "windows": len(nb),
        "base_all": float(np.exp(nb.sum() / tb.sum())),
        "base_first_k": float(np.exp(nb[:k].sum() / tb[:k].sum())),
        "base_subsets": np.quantile(ppl_b, [0.025, 0.5, 0.975]).tolist(),
    }
    if comp_windows is not None:
        nc = np.array([w["nll"] for w in comp_windows])
        ratio = np.exp((nc[picks].sum(1) - nb[picks].sum(1)) / tb[picks].sum(1))
        ppl_c = np.exp(nc[picks].sum(1) / tb[picks].sum(1))
        out |= {
            "ratio_all": float(np.exp((nc.sum() - nb.sum()) / tb.sum())),
            "ratio_first_k": float(np.exp((nc[:k].sum() - nb[:k].sum()) / tb[:k].sum())),
            "ratio_subsets": np.quantile(ratio, [0.025, 0.5, 0.975]).tolist(),
            "comp_subsets": np.quantile(ppl_c, [0.025, 0.5, 0.975]).tolist(),
        }
    return out
