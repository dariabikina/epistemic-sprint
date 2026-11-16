# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""A poster of the Zipf compression results: one claim per figure, large type, and a canvas.

    gt poster  ->  <GT>/poster/{0-example, 1-skill-vs-confidence, 2-training, 3-inputs, 4-blocks, 5-umap,
                   6-report}.png, poster.canvas

Every figure stands alone: `frame` gives it a headline (the claim), two or three sentences on how to read
it, and a footnote with the model, the seeds and what the bars are. Lines are labelled directly. Colour means
one thing: blue = quantization, orange = WANDA pruning, grey = other pruning. A shaded zone marks where a
compressed model is surer exactly where it got worse.

Units.
    skill       % of the learnable information the model uses: 100 (1 - excess / (ln 256 - floor)), with
                excess = mean log p_Bayes - log p_model over validation tokens and floor = the Bayes loss.
                Uniform guessing = 0; the Bayes predictive = 100.
    skill lost  base skill - compressed skill, in percentage points.
    change in uncertainty on the tokens it now predicts worse
                mean change in the model's entropy (nats), compressed - base, on the 10% of positions whose
                KL to Bayes grew most. Below 0: the model became surer exactly where it got worse.
Seeds: MAIN has four; dots are their mean, bars their range. Other runs: seed 0 (E1: seeds 0 and 1), bars are
95% bootstrap intervals over the 1,024 validation sequences.
"""

import json
from pathlib import Path

import numpy as np

from . import data as D
from .figures import boot, ci, tokens_per_param
from .stats import resample_sequences

ACCENT, QUANT, GREY = "#D55E00", "#0072B2", "0.55"
MAIN_RUNS = [f"zipf-gt-6m-q-s{s}-long" for s in range(4)]


def _arrays(run, setting):
    z = np.load(D.GT / "compress" / run / f"{setting.replace(':', 'of')}.npz")
    return {k: z[k].astype(np.float64) for k in ("logp", "bayes_logp", "kl", "entropy")}


def _learnable(base) -> float:
    return np.log(256) - (-base["bayes_logp"].mean())


def skill_per_sequence(arr, learnable) -> np.ndarray:
    return 100 * (1 - (arr["bayes_logp"] - arr["logp"]).mean(1) / learnable)


def main_points() -> dict:
    """Every setting on MAIN, every seed: skill lost (pp) and Δ uncertainty where worst."""
    out = {}
    for run in MAIN_RUNS:
        if not (D.GT / "compress" / run / "base.npz").exists():
            continue
        base = _arrays(run, "base")
        tot = _learnable(base)
        settings = [f.stem.replace("of", ":") for f in (D.GT / "compress" / run).glob("*.npz")]
        for st in settings:
            if st == "base" or "@" in st:
                continue
            c = _arrays(run, st)
            dk = c["kl"] - base["kl"]
            worst = dk >= np.quantile(dk, 0.9)
            lost = 100 * ((c["bayes_logp"] - c["logp"]).mean() - (base["bayes_logp"] - base["logp"]).mean()) / tot
            out.setdefault(st, []).append((lost, float((c["entropy"] - base["entropy"])[worst].mean())))
    return {st: np.array(v) for st, v in out.items()}


INK, NOTE = "0.15", "0.42"
SHADE = "#efece6"  # the zone where a model grew surer where it got worse
DPI = 200
DH_LABEL = "change in uncertainty on the tokens\nit now predicts worse (nats)"
NAMES = {
    "gptq-2": "GPTQ 2-bit",
    "gptq-3": "GPTQ 3-bit",
    "rtn-2": "RTN 2-bit",
    "rtn-3": "RTN 3-bit",
    "rtn-3+emb": "RTN 3-bit + embedding",
    "magnitude-50": "magnitude 50%",
    "magnitude-70": "magnitude 70%",
    "wanda-50": "WANDA 50%",
    "wanda-70": "WANDA 70%",
    "wanda-2:4": "WANDA 2:4",
}


def family_colour(setting: str) -> str:
    """Colour has one job on the poster: blue = quantization, orange = WANDA pruning, grey = other pruning."""
    if setting.startswith("wanda"):
        return ACCENT
    return GREY if setting.startswith(("magnitude", "shortgpt")) else QUANT


def theme():
    import seaborn as sns

    sns.set_theme(
        context="poster",
        style="ticks",
        font_scale=0.8,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
            "axes.edgecolor": "0.3",
            "xtick.color": "0.3",
            "ytick.color": "0.3",
            "axes.labelcolor": INK,
            "text.color": INK,
        },
    )


def _wrap(text: str, fig, size: float) -> list[str]:
    import textwrap

    return textwrap.wrap(text, int(fig.get_figwidth() * 0.96 * 72 / (0.47 * size)))


def frame(fig, headline: str, read, note: str) -> tuple:
    """The text every poster figure carries: the claim, how to read the figure, and a footnote with the
    sample and the intervals. Returns the rectangle left for the axes (for tight_layout)."""
    h = fig.get_figheight()
    paragraphs = [read] if isinstance(read, str) else [p for p in read if p]  # one line break between paragraphs
    read_lines = [line for p in paragraphs for line in _wrap(p, fig, 19)]
    head_lines, note_lines = _wrap(headline, fig, 32), _wrap(note, fig, 14)
    below_head = 0.3 + 0.52 * len(head_lines) + 0.18
    fig.text(0.02, 1 - 0.3 / h, "\n".join(head_lines), fontsize=30, va="top", linespacing=1.2)
    fig.text(0.02, 1 - below_head / h, "\n".join(read_lines), fontsize=19, va="top", color="0.25", linespacing=1.35)
    fig.text(0.02, 0.18 / h, "\n".join(note_lines), fontsize=14, va="bottom", color=NOTE, linespacing=1.3)
    top = below_head + 0.37 * len(read_lines) + 0.45
    bottom = 0.18 + 0.27 * len(note_lines) + 0.25
    return (0, bottom / h, 1, 1 - top / h)


def save(fig, path, rect=None):
    import matplotlib.pyplot as plt

    if rect:
        fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _surer_zone(ax, low: float, words: bool = True):
    """Shade below zero: there the compressed model is surer exactly where it got worse."""
    ax.axhspan(low, 0, color=SHADE, zorder=0, lw=0)
    ax.axhline(0, color="0.3", lw=1.2, zorder=1)
    if words:
        kw = dict(transform=ax.transAxes, fontsize=17, color="0.3")
        ax.text(0.02, 0.97, "↑ less sure where it got worse", va="top", **kw)
        ax.text(0.02, 0.03, "↓ surer where it got worse: false confidence", va="bottom", **kw)


def _plain_log_axis(axis, ticks):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(lambda v, _: f"{v:g}")
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def fig_skill_vs_confidence(plt, path):
    pts = main_points()
    fig, ax = plt.subplots(figsize=(13, 10))
    _surer_zone(ax, -0.95)
    focus = {"gptq-2": (14, 12, "left", "bottom"), "wanda-50": (-24, -2, "right", "center")}
    named = {  # label offset in points, alignment
        "wanda-2:4": (26, 0, "left", "center"),
        "rtn-2": (16, 10, "left", "bottom"),
        "wanda-70": (-14, 10, "right", "bottom"),
        "magnitude-70": (14, 10, "left", "bottom"),
        "magnitude-50": (-12, 14, "right", "bottom"),
        "rtn-3+emb": (-12, -10, "right", "top"),
    }
    mild = []
    for st, v in pts.items():
        if v[:, 0].mean() < 0.05:  # 8-bit and GPTQ 4-bit lose almost nothing: off the left edge
            continue
        colour, m = family_colour(st), v.mean(0)
        size = 19 if st in focus else (11 if st in named else 8)
        ax.plot([v[:, 0].min(), v[:, 0].max()], [m[1], m[1]], color=colour, lw=2, alpha=0.55, zorder=2)
        ax.plot([m[0], m[0]], [v[:, 1].min(), v[:, 1].max()], color=colour, lw=2, alpha=0.55, zorder=2)
        ax.plot(*m, "o", ms=size, color=colour, mec="white", mew=1.5, zorder=3)
        if st in focus or st in named:
            dx, dy, ha, va = (focus | named)[st]
            ax.annotate(
                NAMES[st],
                m,
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=22 if st in focus else 16,
                fontweight="bold" if st in focus else "normal",
                color=colour,
                ha=ha,
                va=va,
            )
        else:
            mild.append(m)
    mild = np.array(mild)
    ax.text(
        np.exp(np.log(mild[:, 0]).mean()),
        0.06,
        "3- and 4-bit quantization, 30% pruning:\nlittle skill lost, little change",
        ha="center",
        fontsize=15,
        color=NOTE,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.05, 13)
    _plain_log_axis(ax.xaxis, [0.1, 0.3, 1, 3, 10])
    ax.set_ylim(-0.95, 0.55)
    ax.set_xlabel("prediction skill lost to compression (percentage points, log scale)")
    ax.set_ylabel(DH_LABEL)
    rect = frame(
        fig,
        "Equal loss of knowledge, opposite self-knowledge",
        "Each dot is one way of compressing the same model. Across: how much prediction skill it lost. "
        "Up or down: whether it became less or more certain on the tokens it now predicts worse. At about "
        "1 point lost, GPTQ 2-bit grew less sure there; WANDA 50% grew surer.",
        "Transformer with 5.2M parameters trained on Zipf-distributed symbols, where the best possible prediction "
        "is known exactly. Skill: % of the predictable information captured (uncompressed: 98.6%). Dots: mean of "
        "4 training seeds; bars: range across seeds. Tokens it predicts worse: the 10% whose error grew most. "
        "Blue: quantization; orange: WANDA pruning; grey: magnitude pruning.",
    )
    save(fig, path, rect)


def fig_training(plt, path):
    series = [("base", "black", "uncompressed"), ("wanda-50", ACCENT, "WANDA 50%"), ("gptq-2", QUANT, "GPTQ 2-bit")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    for ax, (size, title) in zip(axes, (("gt-6m-q", "5.2M parameters"), ("gt-1m-q", "0.9M parameters"))):
        runs = [f"zipf-{size}-s0-{n}" for n in ("d8k", "long", "d128k")]
        runs = [r for r in runs if (D.GT / "compress" / r / "base.npz").exists()]
        tot = {r: _learnable(_arrays(r, "base")) for r in runs}
        xs = [tokens_per_param(r) for r in runs]
        ends = {}
        for st, colour, label in series:
            ys = []
            for r, x in zip(runs, xs):
                est, dr = boot(skill_per_sequence(_arrays(r, st), tot[r]))
                lo, hi = ci(dr)
                ax.plot([x, x], [lo, hi], color=colour, lw=2.5, zorder=2)
                ys.append(est)
            ax.plot(xs, ys, "o-", color=colour, lw=3, ms=11, mec="white", mew=1.5, zorder=3)
            ends[label] = (ys, colour)
        # direct labels, where the lines are apart
        ys, colour = ends["uncompressed"]
        ax.annotate(
            "uncompressed", (xs[1], ys[1]), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=17
        )
        if size == "gt-6m-q":
            for label, dy in (("WANDA 50%", 12), ("GPTQ 2-bit", -12)):
                ys, colour = ends[label]
                ax.annotate(
                    label,
                    (xs[2], ys[2]),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if dy > 0 else "top",
                    fontsize=17,
                    color=colour,
                )
        else:
            for label, dy in (("WANDA 50%", 14), ("GPTQ 2-bit", -14)):
                ys, colour = ends[label]
                ax.annotate(
                    label,
                    (xs[1], ys[1]),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if dy > 0 else "top",
                    fontsize=17,
                    color=colour,
                )
        ax.set_xscale("log")
        ax.set_xlim(xs[0] / 1.6, xs[-1] * 1.6)
        _plain_log_axis(ax.xaxis, [round(x) for x in xs])
        ax.set_title(title, fontsize=20, pad=14)
        ax.set_xlabel("training tokens per parameter")
    axes[0].set_ylabel("prediction skill\n(% of predictable information)")
    rect = frame(
        fig,
        "More training helps the uncompressed model, not the compressed ones",
        "Each model is trained on 4 and 16 times more data, then compressed. The uncompressed model keeps "
        "improving. The compressed ones stall or get worse: most of what extra training adds does not "
        "survive compression.",
        "Transformers trained on Zipf-distributed symbols, one training seed per point. Bars: 95% bootstrap "
        "intervals over 1,024 validation documents. Skill: 0 = guessing, 100 = the best possible prediction.",
    )
    save(fig, path, rect)


def fig_inputs(plt, path):
    worlds = [("zipf-s0.7", 0.7), ("zipf", 1.1), ("zipf-s1.5", 1.5)]
    fig, ax = plt.subplots(figsize=(13, 9.5))
    for st, colour, ls in (("gptq-2", QUANT, "-"), ("wanda-50", ACCENT, "-"), ("rtn-3+emb", QUANT, "--")):
        xs, ys = [], []
        for w, s in worlds:
            r = f"{w}-gt-6m-q-s0-long"
            base = _arrays(r, "base")
            tot = _learnable(base)
            est, dr = boot(skill_per_sequence(base, tot) - skill_per_sequence(_arrays(r, st), tot))
            lo, hi = ci(dr)
            ax.plot([s, s], [lo, hi], color=colour, lw=2.5, zorder=2)
            xs.append(s)
            ys.append(est)
        ax.plot(xs, ys, "o", ls=ls, color=colour, lw=3, ms=12, mec="white", mew=1.5, zorder=3)
        ax.annotate(
            NAMES[st],
            (xs[0], ys[0]),
            xytext=(-16, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=18,
            color=colour,
        )
    ax.set_yscale("log")
    _plain_log_axis(ax.yaxis, [0.2, 0.5, 1, 2, 4])
    ax.set_xlim(0.38, 1.58)
    ax.set_xticks(
        [0.7, 1.1, 1.5],
        ["0.7\nflat: many symbols\nabout equally common", "1.1", "1.5\nsteep: a few symbols\ndominate"],
    )
    ax.set_xlabel("how unequal the symbol frequencies are (Zipf exponent)")
    ax.set_ylabel("prediction skill lost to compression\n(percentage points, log scale)")
    rect = frame(
        fig,
        "Compression costs more when many symbols are about equally common",
        "Three worlds that differ only in how unequal their symbol frequencies are. One model is trained on "
        "each, then compressed. From the steep world to the flat one, the skill lost grows 4 to 19 times.",
        "Transformers with 5.2M parameters, 51 training tokens per parameter, one training seed. Bars: 95% "
        "bootstrap intervals over 1,024 validation documents. In a world with no context to use (independent "
        "symbols), the same weight compression loses nothing measurable.",
    )
    save(fig, path, rect)


def _worst_dH(c, base) -> float:
    dk = c["kl"] - base["kl"]
    return float((c["entropy"] - base["entropy"])[dk >= np.quantile(dk, 0.9)].mean())


def _worst_dH_interval(c, base, B: int = 300) -> tuple[float, float]:
    """95% interval of _worst_dH, resampling documents (the worst 10% is recomputed in each draw)."""
    dk, dh = c["kl"] - base["kl"], c["entropy"] - base["entropy"]
    draws = []
    for idx in resample_sequences(len(dk), B, 0):
        k, h = dk[idx], dh[idx]
        draws.append(h[k >= np.quantile(k, 0.9)].mean())
    return ci(np.array(draws))


def fig_blocks(plt, path):
    run = MAIN_RUNS[0]
    base = _arrays(run, "base")
    tot = _learnable(base)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 9.5))
    _surer_zone(right, -0.22, words=False)
    for st, colour, shift in (("gptq-2", QUANT, -0.06), ("wanda-50", ACCENT, 0.06)):
        lost, dH = [], []
        for b in range(6):
            c = _arrays(run, f"{st}@{b}")
            est, dr = boot(skill_per_sequence(base, tot) - skill_per_sequence(c, tot))
            lost.append((est, *ci(dr)))
            dH.append((_worst_dH(c, base), *_worst_dH_interval(c, base)))
        x = np.arange(6) + shift
        for ax, vals in ((left, lost), (right, dH)):
            for xi, (_, lo, hi) in zip(x, vals):
                ax.plot([xi, xi], [lo, hi], color=colour, lw=2.5, zorder=2)
            ax.plot(x, [v[0] for v in vals], "o-", color=colour, lw=3, ms=12, mec="white", mew=1.5, zorder=3)
        up = st == "wanda-50"
        left.annotate(
            NAMES[st],
            (x[4], lost[4][0]),
            xytext=(-6, 22 if up else -18),
            textcoords="offset points",
            fontsize=18,
            color=colour,
            ha="center",
            va="bottom" if up else "top",
        )
    for ax in (left, right):
        ax.set_xticks(range(6), ["0\nfirst", "1", "2", "3", "4", "5\nlast"])
        ax.set_xlabel("the one block that was compressed")
    left.set_ylim(0, None)
    left.set_title("Knowledge lost", fontsize=20, pad=14)
    left.set_ylabel("prediction skill lost (percentage points)")
    right.set_title("Self-knowledge", fontsize=20, pad=14)
    right.set_ylabel(DH_LABEL)
    right.set_ylim(-0.22, 0.2)
    kw = dict(transform=right.transAxes, fontsize=16, color="0.3", ha="right")
    right.text(0.98, 0.96, "↑ less sure where it got worse", va="top", **kw)
    right.text(0.04, 0.03, "↓ surer where it got worse:\nfalse confidence", va="bottom", **(kw | {"ha": "left"}))
    rect = frame(
        fig,
        "Knowledge is lost in early blocks; false confidence comes from late ones",
        "The model has six blocks in a row. Here one block at a time is compressed and the rest left "
        "untouched. Left: compressing block 0 or 1 costs the most skill. Right: compressing block 4 or 5 "
        "with WANDA makes the model surer where it got worse.",
        "Transformer with 5.2M parameters on Zipf-distributed symbols, one training seed. Bars: 95% bootstrap "
        "intervals over 1,024 validation documents. Tokens it predicts worse: the 10% whose error grew most.",
    )
    save(fig, path, rect)


REPORT_RUNS = [f"zipf-report-gt-6m-q-s{s}-long" for s in range(2)]
REPORT_ROWS = ["rtn-2", "rtn-3+emb", "gptq-2", "rtn-3", "gptq-3", "magnitude-50", "wanda-50", "wanda-2:4"]


def _overconfidence(run: str, setting: str) -> dict:
    """Confidence minus the exact chance that the guess is right, for the model's bet (m - c) and its
    answer when asked (r - c). Mean over all steps, and over the steps where compression switched the
    guess to a symbol whose exact chance is lower by more than 0.1. Each (est, lo, hi), 95% bootstrap
    over documents. `setting` = "base" gives the uncompressed model (all steps only)."""
    d = D.GT / "report" / run
    base, comp = np.load(d / "base.npz"), np.load(d / f"{setting.replace(':', 'of')}.npz")
    idx = resample_sequences(len(base["c"]), 1000, 0)
    groups = {"all": np.ones_like(base["c"], bool)}
    if setting != "base":
        groups["worse"] = (comp["c"] - base["c"]) < -0.1  # same context: c changes only if the guess does
    out = {}
    for g, mask in groups.items():
        den = mask.sum(1)
        for k in ("m", "r"):
            num = ((comp[k] - comp["c"]) * mask).sum(1)
            out[g, k] = (num.sum() / den.sum(), *ci(num[idx].sum(1) / den[idx].sum(1)))
    return out


def _report_example(fig, spec):
    """Three small panels with made-up numbers: what prediction confidence, stated confidence and
    overconfidence are, for one guess."""
    steps = [
        (
            "1. The model predicts",
            ["A", "B", "C", "others"],
            [40, 25, 10, 25],
            0,
            "○  prediction confidence: 40%\n    (the probability on its top choice, A)",
        ),
        (
            "2. We ask: is your guess right?",
            ["yes", "no"],
            [55, 45],
            0,
            "●  stated confidence: 55%\n    (the probability it gives to yes)",
        ),
        (
            "3. The real chance of A",
            ["A"],
            [40],
            None,
            "overconfidence = confidence − real chance:\n○  40 − 40 = 0\n●  55 − 40 = +15 points",
        ),
    ]
    for j, (title, names, values, pick, note) in enumerate(steps):
        ax = fig.add_subplot(spec[j])
        colours = ["black" if pick is None else (QUANT if i == pick else "0.75") for i in range(len(names))]
        ax.barh(range(len(names)), values, color=colours, height=0.7)
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v}%", va="center", fontsize=15, color="0.25")
        ax.set_yticks(range(len(names)), names, fontsize=16)
        ax.set_ylim(3.6, -0.6)
        ax.set_xlim(0, 75)
        ax.set_xticks([])
        ax.tick_params(left=False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=17, loc="left", pad=8)
        ax.text(0, -0.08, note, transform=ax.transAxes, va="top", fontsize=15, color="0.2", linespacing=1.35)


def fig_report(plt, path):
    """E1: how far each kind of confidence sits above the exact chance of being right, after compression.
    The exact chance is the zero line. One row per setting, two training seeds per row."""
    if not all((D.GT / "report" / r / "base.npz").exists() for r in REPORT_RUNS):
        return
    rows = ["base", *REPORT_ROWS]
    fig = plt.figure(figsize=(13, 17))
    rect = frame(
        fig,
        "Asked whether its guess is right, a model after RTN quantization is overconfident",
        "A model's confidence in its own guess can be read in two ways, shown in the example. Below it, for "
        "each way of compressing the model: how many points each confidence sits above the real chance that "
        "the guess is right.",
        "Transformer with 5.2M parameters on Zipf-distributed symbols, where the real chance of every symbol is "
        "known exactly and never changes. The stated confidence was trained to equal the real chance. Worse "
        "guess: compression changed the model's top choice to a symbol whose real chance is lower by more than "
        "10 points (2–4% of 15,360 guesses). Choosing those guesses by itself favours overconfidence, so compare "
        "rows and markers within that panel; it has no uncompressed row because the guesses differ per setting. "
        "Per row: two training seeds; bars: 95% bootstrap intervals over validation documents. Blue: "
        "quantization; orange: WANDA pruning; grey: magnitude pruning.",
    )
    h = fig.get_figheight()
    fig.text(0.02, rect[3] - 0.1 / h, "Example, with made-up numbers", fontsize=19, va="top", fontweight="bold")
    top = rect[3] - 1.05 / h
    example = fig.add_gridspec(1, 3, left=0.09, right=0.82, top=top, bottom=top - 1.9 / h, wspace=0.6)
    _report_example(fig, example)
    grid = fig.add_gridspec(1, 2, left=0.28, right=0.97, top=top - 4.1 / h, bottom=rect[1] + 1.5 / h, wspace=0.08)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    panels = [("all", "All guesses"), ("worse", "Only where compression\nswitched it to a worse guess")]
    n = len(rows)
    for ax, (group, title) in zip(axes, panels):
        ax.axvspan(0, 45, color=SHADE, zorder=0, lw=0)
        ax.axvline(0, color="0.3", lw=1.2)
        for i, st in enumerate(rows):
            colour = "black" if st == "base" else family_colour(st)
            for seed, run in enumerate(REPORT_RUNS):
                o = _overconfidence(run, st)
                if (group, "m") not in o:
                    continue
                y = n - 1 - i + (0.15 if seed == 0 else -0.15)
                for k, filled in (("m", False), ("r", True)):
                    est, lo, hi = (100 * v for v in o[group, k])
                    ax.plot([lo, hi], [y, y], color=colour, lw=2.5, alpha=1 if filled else 0.45, zorder=2)
                    ax.plot(est, y, "o", ms=13, mew=2.5, color=colour if filled else "white", mec=colour, zorder=3)
        ax.set_title(title, fontsize=20, pad=14)
        ax.set_xlim(-8, 38)
        ax.set_xticks([0, 10, 20, 30], ["0\nconfidence =\nreal chance", "+10", "+20", "+30"])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, n + 0.2)
        ax.set_xlabel("overconfidence (percentage points)")
    side = axes[0].get_yaxis_transform()
    for i, st in enumerate(rows):
        colour = "black" if st == "base" else family_colour(st)
        name = "uncompressed" if st == "base" else NAMES[st]
        axes[0].text(-0.04, n - 1 - i, name, ha="right", va="center", fontsize=18, color=colour, transform=side)
    first = _overconfidence(REPORT_RUNS[0], rows[1])
    kw = dict(textcoords="offset points", ha="center", va="bottom", fontsize=16, color=QUANT, linespacing=1.1)
    axes[1].annotate("○ prediction\nconfidence", (100 * first["worse", "m"][0], n - 2 + 0.15), xytext=(0, 14), **kw)
    axes[1].annotate("● stated\nconfidence", (100 * first["worse", "r"][0], n - 2 + 0.15), xytext=(0, 14), **kw)
    save(fig, path)


EXAMPLE = (129, 224)  # validation sequence, position: chosen as a clear case (see fig_example)


def fig_example(plt, path):
    """One token of MAIN: the document so far, and each model's prediction for the next symbol.

    Chosen among positions where WANDA-50's entropy fell by > 0.2 nats, GPTQ-2's rose, and WANDA-50 put less
    probability on the true symbol than the base model; this one maximizes those differences."""
    import mlx.core as mx

    from .compress import calibration, compressed
    from .train import with_bos

    i, t = EXAMPLE
    x = D.load("zipf")[1][i : i + 1]
    calib = calibration("zipf")
    probs = {"The exact answer": P_bayes(x)[0, t]}
    for st, name in (("base", "Uncompressed model"), ("gptq-2", "GPTQ 2-bit"), ("wanda-50", "WANDA 50%")):
        model = compressed(D.GT / "runs" / MAIN_RUNS[0], st, calib)
        lg = model(mx.array(with_bos(x))[:, :-1]).astype(mx.float32)[0, t, :256]
        probs[name] = np.array(mx.softmax(lg))
    prefix, truth = x[0, :t], int(x[0, t])
    counts = np.bincount(prefix, minlength=256)
    shown = list(np.argsort(-counts)[:6])
    if truth not in shown:
        shown[-1] = truth
    letters = {s: chr(65 + j) for j, s in enumerate(shown)}
    k, seen = shown.index(truth), counts > 0
    fig, axes = plt.subplots(1, 5, figsize=(22, 8.6), gridspec_kw={"width_ratios": [1.15, 1, 1, 1, 1]}, sharey=True)
    ys = np.arange(len(shown) + 1)
    axes[0].barh(ys[:-1], counts[shown], color="0.7")
    for y, c in zip(ys[:-1], counts[shown]):
        axes[0].text(c + 3, y, str(c), va="center", fontsize=16, color="0.3")
    axes[0].set_title("The document so far", fontsize=21, pad=34)
    axes[0].text(
        0.5,
        1.03,
        f"{t} symbols, {seen.sum()} different ones",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=16,
        color=NOTE,
    )
    axes[0].set_xlabel("times seen")
    axes[0].set_xlim(0, counts.max() * 1.2)
    axes[0].set_xticks([0, 50, 100, 150])
    for ax, (name, p), colour in zip(axes[1:], probs.items(), ["black", "0.4", QUANT, ACCENT]):
        v = np.r_[p[shown], p[~seen].sum()]
        ax.barh(ys, v, color=colour)
        H = -(p * np.log(np.maximum(p, 1e-12))).sum()
        ax.set_title(name, fontsize=21, color=colour, pad=34)
        ax.text(0.5, 1.03, f"uncertainty {H:.2f} nats", transform=ax.transAxes, ha="center", fontsize=16, color=NOTE)
        ax.text(v[0] + 0.03, 0, f"{v[0]:.2f}", va="center", fontsize=18, color=colour)
        ax.text(
            v[k] + 0.03,
            k,
            f"{v[k]:.2f}  ← the true next symbol" if ax is axes[1] else f"{v[k]:.2f}  ←",
            va="center",
            fontsize=18,
            color=colour,
        )
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("probability")
    for ax in axes:
        ax.tick_params(left=False)
    axes[0].set_yticks(ys, [letters[s] for s in shown] + ["any symbol\nnot yet seen"])
    axes[0].invert_yaxis()
    w = probs["WANDA 50%"]
    rect = frame(
        fig,
        f"One prediction: after pruning, the model bets {w[shown[0]]:.2f} on the commonest symbol; "
        f"the exact answer is {probs['The exact answer'][shown[0]]:.2f}",
        f"A document has shown symbol A {counts[shown[0]]} times out of {t}. Each panel shows the probability one "
        f"predictor gives to each possible next symbol. The true next symbol was {letters[truth]}. WANDA 50% is far "
        f"more certain than it should be and gives the truth a third of the probability; GPTQ 2-bit, which loses as "
        f"much skill overall, stays close to the uncompressed model.",
        f"Transformer with 5.2M parameters on Zipf-distributed symbols (seed 0); validation document {i}, "
        f"position {t}, chosen as a clear case of the pattern in figure 1. Uncertainty: entropy of the prediction "
        f"over all 256 symbols.",
    )
    save(fig, path, rect)


def P_bayes(x):
    from . import priors as P

    return P.worlds()["zipf"].predict(x)[0]


NEW_BINS = [0.01, 0.1]  # exact P(next symbol is new): below 1%, 1-10%, above 10%


def _change_by_new(run: str, setting: str) -> list[tuple[float, float, float]]:
    """Mean change in entropy (compressed - base) over all tokens, by the exact P(new); 95% bootstrap."""
    d = D.GT / "compress" / run
    base = np.load(d / "base.npz")
    dh = np.load(d / f"{setting}.npz")["entropy"].astype(np.float64) - base["entropy"]
    which = np.digitize(base["novel_bayes"], NEW_BINS)
    idx = resample_sequences(len(dh), 1000, 0)
    out = []
    for k in range(len(NEW_BINS) + 1):
        num, den = (dh * (which == k)).sum(1), (which == k).sum(1)
        draws = num[idx].sum(1) / den[idx].sum(1)
        out.append((num.sum() / den.sum(), *ci(draws)))
    return out


def fig_umap(plt, path):
    """Final-layer states of MAIN at 2,000 random positions (from gt inside), one UMAP of the base states,
    coloured three ways; and the same change in uncertainty averaged over all tokens."""
    import umap

    run = MAIN_RUNS[0]
    z = np.load(D.GT / "inside" / run / "layer_points.npz")
    pts = z["points"]
    H = z["H_base"][-1].astype(np.float32)
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    Y = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=0).fit_transform(H)
    d = D.GT / "compress" / run
    base = np.load(d / "base.npz")["entropy"][pts[:, 0], pts[:, 1]]
    dH = {st: np.load(d / f"{st}.npz")["entropy"][pts[:, 0], pts[:, 1]] - base for st in ("gptq-2", "wanda-50")}
    fig = plt.figure(figsize=(22, 10))
    rect = frame(
        fig,
        "WANDA grows surer where a repeat is near-certain; GPTQ grows less sure everywhere",
        "Maps: 2,000 tokens placed by the uncompressed model's internal state, so that tokens the model treats "
        "alike sit together. The first map shows the exact chance that the next symbol is new to the document; "
        "the next two show how each compressed model's uncertainty changed. Right: that change, averaged over "
        "all tokens.",
        "Transformer with 5.2M parameters on Zipf-distributed symbols, seed 0. Maps: UMAP (cosine) of final-layer "
        "states, central 94% shown; positions on a map have no units. Right: all 262,144 validation tokens; bars: "
        "95% bootstrap intervals over documents.",
    )
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 1.15],
        wspace=0.08,
        left=0.02,
        right=0.98,
        top=rect[3] - 0.07,
        bottom=rect[1] + 0.16,
    )
    axes = [fig.add_subplot(gs[j]) for j in range(3)]
    lo, hi = np.percentile(Y, [3, 97], axis=0)
    sc = axes[0].scatter(*Y.T, c=z["colour"], cmap="viridis", vmin=-3, vmax=0, s=26, lw=0)
    cb = fig.colorbar(
        sc, cax=axes[0].inset_axes([0.15, -0.07, 0.7, 0.035]), orientation="horizontal", ticks=[-3, -2, -1, 0]
    )
    cb.ax.set_xticklabels(["0.1%", "1%", "10%", "100%"], fontsize=15)
    cb.outline.set_visible(False)
    axes[0].set_title("Exact chance the next\nsymbol is new", fontsize=20)
    for ax, st in zip(axes[1:], ("gptq-2", "wanda-50")):
        o = np.argsort(np.abs(dH[st]))
        sc = ax.scatter(*Y[o].T, c=dH[st][o], cmap="RdBu", vmin=-0.4, vmax=0.4, s=26, lw=0)
        ax.set_title(f"{NAMES[st]}:\nchange in uncertainty", fontsize=20, color=family_colour(st))
        cb = fig.colorbar(
            sc, cax=ax.inset_axes([0.15, -0.07, 0.7, 0.035]), orientation="horizontal", ticks=[-0.4, 0, 0.4]
        )
        cb.ax.set_xticklabels(["surer", "no change", "less sure"], fontsize=15)
        cb.outline.set_visible(False)
    for ax in axes:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = fig.add_subplot(gs[3])
    box = ax.get_position()
    ax.set_position([box.x0 + 0.055, box.y0, box.width - 0.06, box.height])
    _surer_zone(ax, -0.13, words=False)
    for st, shift in (("gptq-2", -0.05), ("wanda-50", 0.05)):
        vals, colour = _change_by_new(run, st), family_colour(st)
        x = np.arange(3) + shift
        for xi, (_, a, b) in zip(x, vals):
            ax.plot([xi, xi], [a, b], color=colour, lw=2.5, zorder=2)
        ax.plot(x, [v[0] for v in vals], "o-", color=colour, lw=3, ms=12, mec="white", mew=1.5, zorder=3)
        up = st == "gptq-2"
        ax.annotate(
            NAMES[st],
            (x[1], vals[1][0]),
            xytext=(0 if up else 12, 14 if up else -12),
            textcoords="offset points",
            ha="center" if up else "left",
            va="bottom" if up else "top",
            fontsize=18,
            color=colour,
        )
    ax.set_ylim(-0.13, 0.11)
    ax.set_xlim(-0.35, 2.35)
    ax.set_xticks(range(3), ["below 1%\nrepeat\nnear-certain", "1–10%", "above 10%\nnew symbol\nplausible"])
    ax.set_xlabel("exact chance the next symbol is new")
    ax.set_ylabel("change in uncertainty (nats)")
    ax.set_title("Averaged over all tokens", fontsize=20)
    ax.text(0.98, 0.03, "↓ surer than before compression", transform=ax.transAxes, ha="right", fontsize=15, color="0.3")
    save(fig, path)


def method_diagram(x0: int, y0: int) -> tuple[list, list, int]:
    """JSON Canvas cards and arrows: from the world to each number on the poster. Returns nodes, edges, height."""
    w, h, gx, gy = 600, 230, 90, 110
    cards = {
        "world": (
            0,
            0,
            "**1. The world**\nEach document picks how repetitive it is (α ∈ {1, 5, 25}), then draws "
            "256 symbols from a Dirichlet process over 256 symbols with Zipf(1.1) frequencies: common symbols are "
            "common everywhere, and each document repeats its own.",
        ),
        "data": (1, 0, "**2. Data**\n200,000 training documents and 1,024 validation documents, from fixed seeds."),
        "bayes": (
            1,
            1,
            "**3. The exact answer**\nFor every validation token, the Bayes probability of each of the "
            "256 possible next symbols, p_Bayes, computed in closed form from the world's definition.",
        ),
        "train": (
            2,
            0,
            "**4. Train**\nA small transformer (6 blocks, 5.2M parameters) learns to predict the next "
            "symbol from 262M training tokens. It never sees p_Bayes.",
        ),
        "compress": (
            3,
            0,
            "**5. Compress**\nThe same code as for the LLMs: quantization (RTN, GPTQ, AWQ; 2–8 bits) "
            "and pruning (magnitude, WANDA; 30–70%), calibrated on 128 training documents.",
        ),
        "score": (
            3,
            1,
            "**6. Score every token**\nUncompressed (p_base) and compressed (p_comp) predictions over the "
            "256 symbols at all 262,144 validation tokens, next to p_Bayes.",
        ),
        "skill": (
            1,
            2,
            "**Skill (figures 1–4)**\nexcess = mean of log p_Bayes(true) − log p_model(true)\n"
            "skill = 100 · (1 − excess / (ln 256 − Bayes loss))\n0 = uniform guessing, 100 = exact Bayes. "
            "Skill lost = base − compressed.",
        ),
        "damage": (
            2,
            2,
            "**Damage per token**\nΔKL_t = KL(p_Bayes ‖ p_comp) − KL(p_Bayes ‖ p_base), summed exactly "
            "over the 256 symbols.\nWorst tokens = the 10% with the largest ΔKL.",
        ),
        "conf": (
            3,
            2,
            "**Δ uncertainty where worst (figures 1, 4, 5)**\nH(p) = −Σ p log p, the model's own "
            "uncertainty. Mean of H(p_comp) − H(p_base) over the worst tokens.\n< 0: surer where it got worse.",
        ),
        "ci": (
            0,
            2,
            "**Error bars**\nResample the 1,024 validation documents with replacement (1,000 times). "
            "Figure 1 also repeats training with four seeds; the bars there are the range across seeds.",
        ),
    }
    nodes = [
        {
            "id": "m-group",
            "type": "group",
            "label": "How the numbers are made",
            "x": x0,
            "y": y0,
            "width": 4 * w + 3 * gx + 80,
            "height": 3 * h + 2 * gy + 120,
        }
    ]
    for k, (c, r, text) in cards.items():
        nodes.append(
            {
                "id": f"m-{k}",
                "type": "text",
                "text": text,
                "x": x0 + 40 + c * (w + gx),
                "y": y0 + 60 + r * (h + gy),
                "width": w,
                "height": h,
                "color": "4" if r == 2 and k != "ci" else None,
            }
        )
    nodes = [{k: v for k, v in n.items() if v is not None} for n in nodes]
    arrows = [
        ("world", "data"),
        ("data", "bayes"),
        ("data", "train"),
        ("train", "compress"),
        ("compress", "score"),
        ("bayes", "score"),
        ("score", "conf"),
        ("score", "damage"),
        ("damage", "conf"),
        ("score", "skill"),
    ]
    edges = [{"id": f"e-{a}-{b}", "fromNode": f"m-{a}", "toNode": f"m-{b}"} for a, b in arrows]
    return nodes, edges, 3 * h + 2 * gy + 120


def write_canvas(out):
    def size(p):
        h = open(p, "rb").read(24)
        return int.from_bytes(h[16:20], "big"), int.from_bytes(h[20:24], "big")

    W = 1300
    cards = [
        (
            "text",
            "# Compression costs little knowledge, and can invert self-knowledge\n\n"
            "Small transformers trained on Zipf-distributed symbols (documents repeat symbols; which ones are "
            "common is stored in the weights). The best possible prediction is known exactly, so every loss is "
            "measured against truth.",
            2 * W + 80,
            300,
        ),
        (
            "text",
            "## How to read\n\n"
            "- **Skill** = % of the learnable information the model uses (guessing 0, perfect 100). "
            "The uncompressed model reaches 98.6%.\n"
            "- **Good:** little skill lost, and the model becomes *less* sure where it got worse.\n"
            "- **Bad:** the model becomes *surer* where it got worse: it cannot tell what it lost.",
            2 * W + 80,
            300,
        ),
        (
            "text",
            "## Terms\n\n"
            "- **Guess:** the symbol the model would output (its most probable one).\n"
            "- **Correctness:** the exact probability that the guess is the next symbol (known here; 0/1 in an LLM).\n"
            "- **Implicit confidence:** the probability the model itself gives its guess. Uncertainty: the entropy "
            "of its whole prediction. Nothing is asked.\n"
            "- **Explicit report:** after a query token ?, the model answers Y or N; its P(Y) is trained to equal "
            'the correctness. Like asking an LLM "is your answer right?".\n'
            "- **Quantization** (RTN, GPTQ, AWQ): every weight rounded to few levels. **Pruning** (magnitude, "
            "WANDA): a share of weights set to zero, the rest kept. **WANDA** removes weights with small "
            "|weight| x input size; 2:4 keeps 2 of every 4, the pattern GPUs accelerate.\n"
            '- Definitions: `GT.md`, "Terms".',
            2 * W + 80,
            440,
        ),
    ]
    from .. import probs_poster  # here, because probs_poster draws with this module

    report = D.GT.parent / "qwen3-8b" / "probs-v1-report.json"
    EXAMPLE_CARD = probs_poster.example_card(json.loads(report.read_text())) if report.exists() else None
    # Each figure carries its own headline, reading guide and footnote, so the canvas adds no captions.
    figs = [
        "0-example.png",
        "1-skill-vs-confidence.png",
        "4-blocks.png",
        "5-umap.png",
        "6-report.png",
        "2-training.png",
        "3-inputs.png",
        "## The same questions in a language model: Qwen3-8B\n\nQwen3-8B and 12 compressed versions of it "
        "answer 3,000 factual questions (PopQA) and 900 marble problems with exact probabilities. For each model we "
        "ask two things: how much it still gets right, and whether its confidence still tracks what it gets right. "
        "Confidence is read in two ways: the probability the model assigned to its own answer, and its reply "
        "when asked whether that answer is correct.",
        "../../qwen3-8b/confidence-ranking.png",
        "../../qwen3-8b/confidence-facts.png",
        "../../qwen3-8b/confidence-inside.png",
        "../../qwen3-8b/confidence-marbles.png",
        "## Interim (15 Nov): does a compressed model know which facts it has lost?\n\n"
        "Qwen3-8B and 12 compressed versions of it get 400 factual questions that each have one correct answer, "
        "and 100 questions about invented subjects that have none, in English and in Russian. "
        "For every question we record three things: the answer the model writes, the probability it "
        "gives the correct answer and each wrong answer on a fixed list, and the probability that it replies Yes "
        "to 'Do you know the answer to this question?'. The card below takes one question through these steps.\n\n"
        "- **Figure 1: facts are lost.** A model knows a fact if it gives the correct answer the highest "
        "probability on the list. Compression makes it lose 3–40% of the facts it knew in English and 7–50% in "
        "Russian.\n"
        "- **Figure 2: the answer probabilities still show what the model gets right.** How far the correct "
        "answer is ahead of the wrong ones separates correctly answered questions from wrongly answered ones "
        "equally well before and after compression (84–88 pairs of 100 in English). P(Yes) separates them less "
        "well (68–75).\n"
        "- **Figure 3: P(Yes) does not pick out the lost facts.** It separates the facts a compressed model kept "
        "from those it lost only as well as the uncompressed model's P(Yes) already did, before anything was "
        "lost. The lost facts are ones the model was less sure of from the start.\n"
        "- **Figure 4: how often the model replies Yes shifts as a whole.** Each compression method moves P(Yes) "
        "up or down for real and invented subjects alike. The direction depends on the model: pruning half the "
        "weights (WANDA 50%) moves Qwen3-8B towards Yes and Qwen3-4B towards No. So more or fewer Yes replies "
        "do not mean the model knows more or less.\n\n"
        "Interim because the list of wrong answers is a first version, 'Do you know the answer?' is asked in one "
        "wording only, and there is one trained model per size. Numbers and limits: `PLAN.md`, section P1.",
        EXAMPLE_CARD,
        "../../qwen3-8b/probs-v1-plain-1-forgetting.png",
        "../../qwen3-8b/probs-v1-plain-2-shows-vs-says.png",
        "../../qwen3-8b/probs-v1-plain-3-kept-vs-lost.png",
        "../../qwen3-8b/probs-v1-plain-4-yes.png",
        "## Details, Qwen3-8B: the same results per language, with every definition\n\n"
        "English on the left, Russian on the right. Top: which signal separates correct from incorrect "
        "answers, including a probe on the hidden state. Middle: facts lost, and the probability of Yes on "
        "lost and kept facts. Bottom: 'I know' separating kept from lost facts, before and after. Separation "
        "is given as AUROC (0.5 = coin flip, 1 = perfect), the plain figures' pairs out of 100 divided by 100.",
        "../../qwen3-8b/probs-v1-coupling-en.png",
        "../../qwen3-8b/probs-v1-coupling-ru.png",
        "../../qwen3-8b/probs-v1-lost-en.png",
        "../../qwen3-8b/probs-v1-lost-ru.png",
        "../../qwen3-8b/probs-v1-tracking-en.png",
        "../../qwen3-8b/probs-v1-tracking-ru.png",
        "## The same on Qwen3-4B (6 compressed versions)\n\n"
        "The smaller model gives the same results. It loses 6–36% of the facts it knew in English and 10–44% in "
        "Russian. The answer margin orders 87–90 pairs of 100 correctly in English, P(Yes) 57–65. P(Yes) "
        "separates kept from lost facts no better after compression than before, in all 12 cases.",
        "../../qwen3-4b/probs-v1-plain-1-forgetting.png",
        "../../qwen3-4b/probs-v1-plain-2-shows-vs-says.png",
        "../../qwen3-4b/probs-v1-plain-3-kept-vs-lost.png",
        "../../qwen3-4b/probs-v1-plain-4-yes.png",
        "## Details, Qwen3-4B: the same results per language, with every definition\n\n"
        "English on the left, Russian on the right. Top: which signal separates correct from incorrect "
        "answers, including a probe on the hidden state. Middle: facts lost, and the probability of Yes on "
        "lost and kept facts. Bottom: 'I know' separating kept from lost facts, before and after. Separation "
        "is given as AUROC (0.5 = coin flip, 1 = perfect), the plain figures' pairs out of 100 divided by 100.",
        "../../qwen3-4b/probs-v1-coupling-en.png",
        "../../qwen3-4b/probs-v1-coupling-ru.png",
        "../../qwen3-4b/probs-v1-lost-en.png",
        "../../qwen3-4b/probs-v1-lost-ru.png",
        "../../qwen3-4b/probs-v1-tracking-en.png",
        "../../qwen3-4b/probs-v1-tracking-ru.png",
    ]
    nodes, y = [], 0
    for i, (_, text, w, h) in enumerate(cards):
        nodes.append({"id": f"c{i}", "type": "text", "text": text, "x": 0, "y": y, "width": w, "height": h})
        y += h + 40
    diagram, edges, dh = method_diagram(0, y)
    nodes += diagram
    y += dh + 80
    col, row_height = 0, 0
    for i, f in enumerate(f for f in figs if f and (f.startswith("## ") or (out / f).exists())):
        if f.startswith("## "):  # a section heading across both columns
            y = y + row_height + 80 if col else y
            col, row_height = 0, 0
            # room for the text: about 190 characters per line at this width, 36 px per line
            lines = sum(1 + len(part) // 190 for part in f.split("\n") if part)
            lines += 1.2 * sum(
                part.startswith((">", "|")) for part in f.split("\n")
            )  # quotes and table rows are taller
            hh = max(240, int(130 + 36 * lines))
            nodes.append({"id": f"h{i}", "type": "text", "text": f, "x": 0, "y": y, "width": 2 * W + 80, "height": hh})
            y += hh + 40
            continue
        pw, ph = size(out / f)
        wide = pw / ph > 2  # the five- and four-panel figures take a full row
        if wide and col:
            y, col, row_height = y + row_height + 80, 0, 0
        width = 2 * W + 80 if wide else W
        height = int(width * ph / pw)
        nodes.append(
            {"id": f"f{i}", "type": "file", "file": f, "x": col * (W + 80), "y": y, "width": width, "height": height}
        )
        row_height = max(row_height, height)
        if wide or col:
            y, col, row_height = y + row_height + 80, 0, 0
        else:
            col = 1
    y = max(n["y"] + n["height"] for n in nodes) + 60
    notes = Path(__file__).resolve().parents[3] / "KNOWING.md"
    nodes.append(
        {
            "id": "notes",
            "type": "text",
            "x": 0,
            "y": y,
            "width": 2 * W + 80,
            "height": 260,
            "text": "## Notes: what 'knowing what it knows' means\n\nThree things under one name (calibrated "
            "prediction, information about its own correctness, reporting it), where calibration comes from in "
            "training, where each is encoded from what we measured, what the literature suggests, and the "
            f"popularity control for the probes.\n\n[KNOWING.md](file://{notes})\n\n`{notes}`",
        }
    )
    y += 260 + 40
    nodes.append(
        {
            "id": "caveats",
            "type": "text",
            "x": 0,
            "y": y,
            "width": 2 * W + 80,
            "height": 220,
            "text": "## Caveats\n\nToy models (0.9–5.2M parameters) on a synthetic world; one real model on one "
            "question set. Seeds and intervals are stated under each figure. Methods and every number: `GT.md`, "
            "`GT-RUNS.md`, `PLAN.md`; detailed figures: `../figures/`.",
        }
    )
    (out / "poster.canvas").write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=1))


def run(say=print):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme()
    out = D.GT / "poster"
    out.mkdir(parents=True, exist_ok=True)
    fig_skill_vs_confidence(plt, out / "1-skill-vs-confidence.png")
    fig_training(plt, out / "2-training.png")
    fig_inputs(plt, out / "3-inputs.png")
    fig_blocks(plt, out / "4-blocks.png")
    fig_example(plt, out / "0-example.png")
    fig_umap(plt, out / "5-umap.png")
    fig_report(plt, out / "6-report.png")
    write_canvas(out)
    say(f"[gt poster] wrote {out}")
