# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Bridge 1: does a compressed LLM still know when its own answer is wrong? Qwen PopQA records, no GPU.

    pixi run python scripts/bridge_popqa.py [--model qwen3-8b]  ->  <RUNS>/<model>/confidence-bridge.{json,png}

Per question, from each setting's eval-short-3000.jsonl: correct (Wu's matching) and conf, the model's implicit
confidence in its greedy answer (geometric-mean token probability). Facts are *kept* (right before and after
compression), *lost* (right before, wrong after) or *always wrong*.

    auroc2              P(conf of a right answer > conf of a wrong answer), each model on its own answers
                        (type-2 AUROC; Fleming & Lau 2014). d_auroc2 = compressed - base, paired.
    conf_kept/lost/always_wrong  mean conf of the compressed model in each group
    lost_vs_kept        P(conf on a lost fact > conf on a kept fact): low = the model is less sure of what it lost
    lost_vs_wrong       P(conf on a lost fact > conf on an always-wrong fact): 0.5 = lost facts look like ordinary
                        wrong answers to the model; 1 = it is as sure of them as of nothing it gets wrong
Intervals: 95% percentile bootstrap over questions (B = 1,000, seed 0), paired across settings.
"""

import argparse
import json

import numpy as np

from wu.gt.stats import auc
from wu.models import RUNS

SETTINGS = [
    "gptq-4",
    "awq-4",
    "gptq-3",
    "awq-3",
    "magnitude-30",
    "wanda-30",
    "wanda-50",
    "wanda-2of4",
    "shortgpt-5",
    "shortgpt-10",
]


def load(model, setting):
    path = RUNS / model / setting / "eval-short-3000.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def stats(cb, cc, fb, fc) -> dict:
    lost, kept, wrong = cb & ~cc, cb & cc, ~cb & ~cc

    def a(x, y):
        return auc(np.r_[fc[x], fc[y]], np.r_[np.ones(x.sum()), np.zeros(y.sum())])

    return {
        "accuracy_lost_pp": 100 * (cb.mean() - cc.mean()),
        "auroc2": auc(fc, cc),
        "auroc2_base": auc(fb, cb),
        "d_auroc2": auc(fc, cc) - auc(fb, cb),
        "conf_kept": fc[kept].mean(),
        "conf_lost": fc[lost].mean(),
        "conf_always_wrong": fc[wrong].mean(),
        "lost_vs_kept": a(lost, kept),
        "lost_vs_wrong": a(lost, wrong),
        "n_lost": int(lost.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    base = load(args.model, "base")
    ids = sorted(base)
    cb = np.array([base[i]["correct"] for i in ids])
    fb = np.array([base[i]["conf"] for i in ids])
    boot = np.random.default_rng(0).integers(0, len(ids), (args.B, len(ids)))
    out = {}
    for s in SETTINGS:
        if not (RUNS / args.model / s / "eval-short-3000.jsonl").exists():
            continue
        c = load(args.model, s)
        cc = np.array([c[i]["correct"] for i in ids])
        fc = np.array([c[i]["conf"] for i in ids])
        est = stats(cb, cc, fb, fc)
        draws = [stats(cb[j], cc[j], fb[j], fc[j]) for j in boot]
        out[s] = {
            k: {
                "est": float(v),
                "lo": float(np.quantile([d[k] for d in draws], 0.025)),
                "hi": float(np.quantile([d[k] for d in draws], 0.975)),
            }
            for k, v in est.items()
        }
        e = out[s]
        print(
            f"{s:>12}: accuracy lost {e['accuracy_lost_pp']['est']:4.1f} pp  AUROC2 {e['auroc2']['est']:.3f} "
            f"(base {e['auroc2_base']['est']:.3f}; change {e['d_auroc2']['est']:+.3f} [{e['d_auroc2']['lo']:+.3f}, {e['d_auroc2']['hi']:+.3f}])  "
            f"lost vs always-wrong {e['lost_vs_wrong']['est']:.2f} [{e['lost_vs_wrong']['lo']:.2f}, {e['lost_vs_wrong']['hi']:.2f}]"
        )
    dest = RUNS / args.model / "confidence-bridge.json"
    dest.write_text(
        json.dumps({"model": args.model, "n_questions": len(ids), "settings": out, "definitions": __doc__}, indent=1)
    )
    plot(out, RUNS / args.model / "confidence-bridge.png", args.model)
    print(f"wrote {dest}")


LABELS = {
    "gptq-4": "GPTQ 4-bit",
    "awq-4": "AWQ 4-bit",
    "gptq-3": "GPTQ 3-bit",
    "awq-3": "AWQ 3-bit",
    "magnitude-30": "magnitude 30%",
    "wanda-30": "WANDA 30%",
    "wanda-50": "WANDA 50%",
    "wanda-2of4": "WANDA 2:4",
    "shortgpt-5": "5 blocks removed",
    "shortgpt-10": "10 blocks removed",
}


def plot(out, path, model):
    """Poster figure, in the style of wu.gt.poster: one row per setting, ordered by accuracy lost; the
    paired change in type-2 AUROC with its interval."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wu.gt.poster import family_colour, frame, save, theme

    theme()
    order = sorted(out, key=lambda s: out[s]["accuracy_lost_pp"]["est"])
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.axvspan(-0.1, 0, color="#efece6", zorder=0, lw=0)
    ax.axvline(0, color="0.3", lw=1.2)
    side = ax.get_yaxis_transform()
    for i, s in enumerate(order):
        y, e, colour, focus = len(order) - 1 - i, out[s], family_colour(s), s == "wanda-2of4"
        d = e["d_auroc2"]
        ax.plot([d["lo"], d["hi"]], [y, y], color=colour, lw=3, zorder=2)
        ax.plot(d["est"], y, "o", ms=19 if focus else 13, color=colour, mec="white", mew=1.5, zorder=3)
        weight = "bold" if focus else "normal"
        ax.text(
            -0.02, y, LABELS[s], ha="right", va="center", fontsize=18, color=colour, fontweight=weight, transform=side
        )
        ax.text(
            1.02,
            y,
            f"{e['accuracy_lost_pp']['est']:.0f}",
            ha="left",
            va="center",
            fontsize=18,
            color="0.3",
            transform=side,
        )
    top = len(order) - 0.3
    ax.text(1.02, top, "accuracy lost\n(points)", ha="left", va="bottom", fontsize=15, color="0.3", transform=side)
    ax.text(-0.004, top, "worse than uncompressed", ha="right", va="bottom", fontsize=16, color="0.3")
    ax.text(0.004, top, "better", ha="left", va="bottom", fontsize=16, color="0.3")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(order) + 0.5)
    ax.set_xlim(-0.1, 0.04)
    ax.set_xticks([-0.08, -0.06, -0.04, -0.02, 0, 0.02])
    ax.set_xlabel("change in how well confidence separates right answers from wrong ones")
    rect = frame(
        fig,
        "In a real language model, confidence survives most compression; WANDA 2:4 breaks it",
        "Qwen3-8B answers 3,000 factual questions. For each way of compressing it: does the model's confidence "
        "in its own answer still tell its right answers from its wrong ones? Most settings are "
        "indistinguishable from the uncompressed model, even after losing 5 to 11 points of accuracy.",
        "PopQA, 3,000 questions. Confidence: geometric-mean token probability of the model's greedy answer. "
        "Axis: area under the ROC curve for right vs wrong answers, compressed minus uncompressed "
        "(uncompressed: 0.87), paired by question. Bars: 95% bootstrap intervals over questions. Blue: "
        "quantization; orange: WANDA pruning; grey: magnitude pruning and block removal.",
    )
    save(fig, path, (0.02, rect[1], 0.9, rect[3]))


if __name__ == "__main__":
    main()
