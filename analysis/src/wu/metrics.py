# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Wu's RQ1 and RQ2 measures, from a base model's records and a compressed model's records on the same questions.

RQ1  Acc_g (Eq. 1); r = Acc_c / Acc_0 and r_g likewise; RS_g = (r_g - r) x 100 (Eq. 2).
RQ2  lost knowledge: correct under the base, wrong under the compressed model.
     LR_g: Eq. 3 divides |lost_g| by N_g, while Figure 3's caption describes a share of what the base got right.
     We report both: `loss_rate` (of base-correct, the caption) and `loss_rate_of_all` (Eq. 3 as printed).
     Confidence (Eq. 4) is calibrated by isotonic regression fitted on a popularity-stratified 20% split
     (raw conf -> correct), and all confidence and ECE (Eq. 5, 10 equal-width bins) numbers use the other 80%.
"""

import random

import numpy as np

from .data import BUCKETS

CAL_FRACTION, ECE_BINS, SPLIT_SEED = 0.2, 10, 0


def calibration_split(records: list[dict]) -> set[str]:
    """Ids in the calibration 20%, stratified by bucket. Depends only on the ids, so it is shared by every setting."""
    rng, cal = random.Random(SPLIT_SEED), set()
    for b in BUCKETS:
        ids = sorted(r["id"] for r in records if r["bucket"] == b)
        cal |= set(rng.sample(ids, round(CAL_FRACTION * len(ids))))
    return cal


def isotonic_fit(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators: the non-decreasing step function of x closest to y; returns a predictor."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(float)
    blocks = []  # [sum, count, x_max]
    for xi, yi in zip(xs, ys):
        blocks.append([yi, 1, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            s, c, xm = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][2] = xm
    edges = np.array([b[2] for b in blocks])
    values = np.array([b[0] / b[1] for b in blocks])
    return lambda q: values[np.minimum(np.searchsorted(edges, q, side="left"), len(values) - 1)]


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    if len(conf) == 0:
        return float("nan")
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(
        sum(
            abs(correct[idx == k].mean() - conf[idx == k].mean()) * (idx == k).sum()
            for k in range(bins)
            if (idx == k).any()
        )
        / len(conf)
    )


def _quantiles(values) -> dict:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(v)),
        "p05": q[0],
        "p25": q[1],
        "median": q[2],
        "p75": q[3],
        "p95": q[4],
        "mean": float(v.mean()),
    }


def calibrated(records: list[dict], cal: set[str]) -> np.ndarray:
    """Calibrated confidence for every record (NaN raw confidence counts as 0)."""
    raw = np.nan_to_num(np.array([r["conf"] for r in records], dtype=float))
    correct = np.array([r["correct"] for r in records], dtype=float)
    fit = np.array([r["id"] in cal for r in records])
    return isotonic_fit(raw[fit], correct[fit])(raw)


def accuracy_table(records: list[dict]) -> dict:
    acc = {"all": float(np.mean([r["correct"] for r in records]))}
    for b in BUCKETS:
        acc[b] = float(np.mean([r["correct"] for r in records if r["bucket"] == b]))
    return acc


def compare(base: list[dict], comp: list[dict]) -> dict:
    """RQ1 and RQ2 for one compressed setting against its base."""
    assert [r["id"] for r in base] == [r["id"] for r in comp], "records must cover the same questions in the same order"
    cal = calibration_split(base)
    conf_c = calibrated(comp, cal)
    acc0, accc = accuracy_table(base), accuracy_table(comp)
    r_all = accc["all"] / acc0["all"] if acc0["all"] else float("nan")
    out = {
        "acc_base": acc0,
        "acc": accc,
        "retention": {"all": r_all},
        "rs_pp": {},
        "loss_rate": {},
        "loss_rate_of_all": {},
        "conf_lost": {},
        "conf_lost_raw": {},
        "ece": {},
        "ece_base": {},
    }
    conf_0 = calibrated(base, cal)
    for g in (*BUCKETS, "all"):
        in_g = [i for i, r in enumerate(base) if g == "all" or r["bucket"] == g]
        test = [i for i in in_g if base[i]["id"] not in cal]
        if g != "all":
            r_g = accc[g] / acc0[g] if acc0[g] else float("nan")
            out["retention"][g] = r_g
            out["rs_pp"][g] = (r_g - r_all) * 100
        known = [i for i in in_g if base[i]["correct"]]
        lost = [i for i in known if not comp[i]["correct"]]
        out["loss_rate"][g] = len(lost) / len(known) if known else float("nan")
        out["loss_rate_of_all"][g] = len(lost) / len(in_g)
        lost_test = [i for i in lost if base[i]["id"] not in cal]
        out["conf_lost"][g] = _quantiles(conf_c[lost_test])
        out["conf_lost_raw"][g] = _quantiles([comp[i]["conf"] for i in lost_test])
        correct_c = np.array([comp[i]["correct"] for i in test], dtype=float)
        correct_0 = np.array([base[i]["correct"] for i in test], dtype=float)
        out["ece"][g] = ece(conf_c[test], correct_c)
        out["ece_base"][g] = ece(conf_0[test], correct_0)
    return out
