# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Does the correctness probe read more than how well known the subject is? CPU only.

    pixi run python scripts/probe_controls.py [--model qwen3-8b]  ->  <RUNS>/<model>/probe-controls.json

The probe of `wu.confidence_report` (ridge on 256 principal components of the residual stream at the last
token of "is the proposed answer correct?", cross-fitted, refitted per setting) predicts whether the model's
own answer is right. Three controls, per setting and recorded layer:

    within    AUROC of the probe score among questions of the same relation and popularity third
              (16 relations x head/middle/tail), averaged over cells with both outcomes, weighted by the
              number of right-wrong pairs. Popularity and relation cannot separate questions within a cell.
    known     AUROC of a ridge model on what is known without the hidden state: log10 subject popularity,
              relation, the log-probability and length of the model's reply, whether it was cut off.
    added     AUROC of the same model with the 256 components added. added - known = what the hidden state
              holds beyond those covariates.
Intervals: 95% percentile bootstrap over questions (B = 300, seed 0); the fits are not repeated.
"""

import argparse
import json

import numpy as np

from wu import data
from wu.confidence_report import LABELS, SETTINGS, load_facts, pca_scores, probe_scores
from wu.gt.stats import auc
from wu.models import RUNS, setting_dir

B = 300


def within_cells(score, y, cell) -> float:
    total, pairs = 0.0, 0
    for c in np.unique(cell):
        m = cell == c
        n1 = int(y[m].sum())
        n0 = int(m.sum()) - n1
        if n1 and n0:
            total += auc(score[m], y[m]) * n1 * n0
            pairs += n1 * n0
    return total / pairs


def interval(fn, n, idx) -> dict:
    draws = [fn(j) for j in idx]
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return {"est": float(fn(np.arange(n))), "lo": float(lo), "hi": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    args = ap.parse_args()
    rows = data.popqa(3000)
    relations = sorted({r["prop"] for r in rows})
    cell = np.array([relations.index(r["prop"]) * 3 + ("head", "middle", "tail").index(r["bucket"]) for r in rows])
    onehot = np.eye(len(relations))[[relations.index(r["prop"]) for r in rows]]
    log_pop = np.log10([max(r["s_pop"], 1) for r in rows])
    n, q = len(rows), np.arange(len(rows))
    idx = np.random.default_rng(0).integers(0, n, (B, n))
    out = {"model": args.model, "definitions": __doc__, "settings": {}}
    for s in ["base", *SETTINGS]:
        f = load_facts(args.model, s)
        if f is None or f["c"].mean() < 0.01:
            continue
        rec = [
            json.loads(x) for x in (setting_dir(args.model, s) / "eval-short-3000.jsonl").read_text().split("\n") if x
        ]
        y = f["c"].astype(float)
        known_X = np.c_[
            log_pop,
            onehot,
            [r["logprob_sum"] for r in rec],
            [r["n_tokens"] for r in rec],
            [not r["finished"] for r in rec],
        ]
        known = probe_scores(known_X, y, known_X, q)
        layers = []
        for j, layer in enumerate(f["layers"]):
            Z = pca_scores(f["H"][:, j])[0]
            probe = probe_scores(Z, y, Z, q)
            both_X = np.c_[known_X, Z]
            added = probe_scores(both_X, y, both_X, q)
            layers.append(
                {
                    "layer": int(layer),
                    "probe": interval(lambda k: auc(probe[k], y[k]), n, idx),
                    "within": interval(lambda k: within_cells(probe[k], y[k], cell[k]), n, idx),
                    "added": interval(lambda k: auc(added[k], y[k]), n, idx),
                    "added_minus_known": interval(lambda k: auc(added[k], y[k]) - auc(known[k], y[k]), n, idx),
                }
            )
        best = max(layers, key=lambda r: r["within"]["est"])
        out["settings"][s] = {
            "known": interval(lambda k: auc(known[k], y[k]), n, idx),
            "known_within": interval(lambda k: within_cells(known[k], y[k], cell[k]), n, idx),
            "popularity_only": float(auc(log_pop, y)),
            "layers": layers,
        }
        f3 = lambda e: f"{e['est']:.2f} [{e['lo']:.2f}, {e['hi']:.2f}]"
        print(
            f"{LABELS[s]:>18}: popularity alone {auc(log_pop, y):.2f} | known {f3(out['settings'][s]['known'])} | "
            f"layer {best['layer']}: probe {f3(best['probe'])}, within cells {f3(best['within'])}, "
            f"known + state {f3(best['added'])} (gain {best['added_minus_known']['est']:+.3f} "
            f"[{best['added_minus_known']['lo']:+.3f}, {best['added_minus_known']['hi']:+.3f}]) | "
            f"first layer within {f3(layers[0]['within'])}",
            flush=True,
        )
    dest = RUNS / args.model / "probe-controls.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
