# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Tracking index: does the model's sense of knowing move when its knowledge moves?

M = a meta-knowledge signal (probe score from a probe trained on the uncompressed model's states; or self-report P(Yes)).
Tracking AUC = P(dM on a lost fact < dM on a kept fact), dM = M(compressed) - M(uncompressed), same question.
0.5: M ignores the loss (detached). 1.0: M drops exactly on the lost facts (attached).
Familiarity check: how well log popularity alone predicts correctness, and whether M still predicts it within a popularity bucket.
"""

import sys

import numpy as np

from wu.cli import read_jsonl
from wu.data import BUCKETS, popqa
from wu.models import setting_dir
from wu.represent import Probe, auc, outcome

rng = np.random.default_rng(0)


def boot(score, label, n=500):
    v = [auc(score[b], label[b]) for b in (rng.integers(0, len(score), len(score)) for _ in range(n))]
    return np.nanquantile(v, [0.025, 0.975])


rows = popqa(3000)
logpop = np.log10(np.array([r["s_pop"] for r in rows]) + 1)
bucket = np.array([r["bucket"] for r in rows])
for spec in sys.argv[1:]:
    model, setting = spec.split(":")
    A, B = (
        np.load(setting_dir(model, "base") / "states-3000.npz"),
        np.load(setting_dir(model, setting) / "states-3000.npz"),
    )
    base = read_jsonl(setting_dir(model, "base") / "eval-short-3000.jsonl")
    comp = read_jsonl(setting_dir(model, setting) / "eval-short-3000.jsonl")
    y0 = np.array([r["correct"] for r in base], float)
    cat = outcome(base, comp)
    sel = (cat == 1) | (cat == 2)
    lab = (cat[sel] == 1).astype(float)  # kept = 1, lost = 0
    j = len(A["layers"]) - 2
    fold = np.random.default_rng(0).integers(0, 5, len(base))
    s0, sc = np.zeros(len(base)), np.zeros(len(base))
    for f in range(5):
        p = Probe(ridge=100.0).fit(A["H"][fold != f, j], y0[fold != f])
        te = fold == f
        s0[te], sc[te] = p.score(A["H"][te, j]), p.score(B["H"][te, j])

    def resid(after, before):  # change beyond what the starting value predicts (removes regression to the mean)
        X = np.c_[np.ones(sel.sum()), before[sel]]
        coef = np.linalg.lstsq(X, after[sel], rcond=None)[0]
        out = np.full(len(after), np.nan)
        out[sel] = after[sel] - X @ coef
        return out

    dM, dP = resid(sc, s0), resid(B["p_yes"], A["p_yes"])
    props = np.array([r["prop"] for r in rows])
    within_rel = np.nanmean(
        [auc(s0[props == q], y0[props == q]) for q in set(props) if 0 < y0[props == q].sum() < (props == q).sum()]
    )
    print(f"\n{model} {setting} (layer {A['layers'][j]}): kept {int(lab.sum())}, lost {int(len(lab) - lab.sum())}")
    print(
        f"  raw change AUC (confounded by regression to the mean): probe {auc((sc - s0)[sel], lab):.2f}, self-report {auc((B['p_yes'] - A['p_yes'])[sel], lab):.2f}"
    )
    print(f"  tracking AUC, probe:       {auc(dM[sel], lab):.2f} {np.round(boot(dM[sel], lab), 2)}")
    print(f"  tracking AUC, self-report: {auc(dP[sel], lab):.2f} {np.round(boot(dP[sel], lab), 2)}")
    c0 = np.nan_to_num(np.array([r["conf"] for r in base]))
    cc = np.nan_to_num(np.array([r["conf"] for r in comp]))
    dC = resid(cc, c0)
    print(
        f"  tracking AUC, answer confidence (Wu's Conf, on the generated answer): {auc(dC[sel], lab):.2f} {np.round(boot(dC[sel], lab), 2)}"
    )
    print(
        f"  familiarity: log popularity -> base correct AUC {auc(logpop, y0):.2f}; probe -> base correct AUC {auc(s0, y0):.2f}; "
        f"probe within popularity bucket "
        + ", ".join(f"{b} {auc(s0[bucket == b], y0[bucket == b]):.2f}" for b in BUCKETS)
        + f"; probe within relation (mean) {within_rel:.2f}"
    )
