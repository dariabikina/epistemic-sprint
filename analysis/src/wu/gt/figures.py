# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Summary figures of the Zipf compression experiments, from cached evaluations and reports (no GPU).

    gt figures  ->  <GT>/figures/{scaling, seeds, inputs, blocks, learning}.png, fits.json, figures.canvas

Intervals: 95% cluster bootstrap over the 1,024 validation sequences (B = 1,000, seed 0), drawn per run
from the cached per-position arrays; tracking-AUC intervals are taken from each run's report.

Scaling fit. For each setting and model size, log10 ΔKL = a + γ log10(tokens per parameter), least squares
over that size's three seed-0 runs. The interval on γ refits the line to each bootstrap draw (runs' draws
independent): it covers sampling only. `max_log10_residual` reports how far the points lie from the line;
with three points, γ describes these runs, not an extrapolation.
"""

import json

import numpy as np

from . import data as D
from . import priors as P
from .stats import B_DEFAULT, resample_sequences

SIZES = {"gt-1m-q": ("0.9M parameters", "o", "white"), "gt-6m-q": ("5.2M parameters", "o", None)}
LENGTHS = ["d8k", "long", "d128k"]
FOCUS = ["gptq-2", "wanda-50", "rtn-3+emb"]
COLOURS = {"gptq-2": "#0072B2", "wanda-50": "#D55E00", "rtn-3+emb": "#009E73", "base": "black"}
VARIANTS = ["zipf-s0.7", "zipf", "zipf-s1.5", "zipf-bursty", "zipf-fresh", "zipf-iid"]
MAIN = "zipf-gt-6m-q-s0-long"


def _cfg(run):
    return json.loads((D.GT / "runs" / run / "config.json").read_text())


def tokens_per_param(run) -> float:
    c = _cfg(run)
    return c["steps"] * c["batch_size"] * P.T / c["parameters"]


def per_sequence(run: str, setting: str, what: str) -> np.ndarray:
    """Per validation sequence: mean over positions of ΔKL (what='dkl') or of excess over Bayes ('excess')."""
    d = D.GT / "compress" / run
    comp = np.load(d / f"{setting.replace(':', 'of')}.npz")
    if what == "excess":
        return (comp["bayes_logp"].astype(np.float64) - comp["logp"]).mean(1)
    return (comp["kl"].astype(np.float64) - np.load(d / "base.npz")["kl"]).mean(1)


def boot(per_seq: np.ndarray, B: int = B_DEFAULT, seed: int = 0) -> tuple[float, np.ndarray]:
    """Mean over sequences and its bootstrap draws (sequences have equal length)."""
    return float(per_seq.mean()), per_seq[resample_sequences(len(per_seq), B, seed)].mean(1)


def ci(draws) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def report(run):
    return json.loads((D.GT / "compress" / run / "report.json").read_text())["settings"]


def fit_slope(points: dict, draws: dict) -> dict:
    """points: x -> y > 0; draws: x -> (B,) draws. log10 y = a + γ log10 x."""
    xs = sorted(points)
    X = np.c_[np.ones(len(xs)), np.log10(xs)]

    def solve(y):
        return np.linalg.lstsq(X, np.log10(np.maximum(y, 1e-6)), rcond=None)[0]

    y = np.array([points[x] for x in xs])
    a, g = solve(y)
    B = len(next(iter(draws.values())))
    gs = np.array([solve(np.array([draws[x][b] for x in xs]))[1] for b in range(B)])
    resid = np.log10(y) - (a + g * np.log10(xs))
    return {
        "gamma": float(g),
        "gamma_ci": ci(gs),
        "intercept": float(a),
        "x": xs,
        "max_log10_residual": float(np.abs(resid).max()),
    }


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def _err(ax, x, est, lo, hi, colour, face, **kw):
    ax.errorbar(
        x,
        est,
        yerr=[[est - lo], [hi - est]],
        fmt="o",
        ms=4,
        color=colour,
        mfc=face or colour,
        mew=1,
        elinewidth=1,
        capsize=0,
        zorder=3,
        **kw,
    )


def scaling(plt, out) -> dict:
    """Damage, excess after compression and tracking AUC against tokens per parameter."""
    fits = {}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for st in FOCUS:
        for size, (label, _, face) in SIZES.items():
            pts, drw, xs = {}, {}, []
            for length in LENGTHS:
                run = f"zipf-{size}-s0-{length}"
                if not (D.GT / "compress" / run / f"{st}.npz").exists():
                    continue
                x = tokens_per_param(run)
                est, dr = boot(per_sequence(run, st, "dkl"))
                pts[x], drw[x] = est, dr
                _err(axes[0], x, est, *ci(dr), COLOURS[st], face)
                e_est, e_dr = boot(per_sequence(run, st, "excess"))
                _err(axes[1], x, e_est, *ci(e_dr), COLOURS[st], face)
                a = report(run)[st]["tracking_auc"]
                _err(axes[2], x, a["est"], a["lo"], a["hi"], COLOURS[st], face)
                xs.append((x, e_est))
            xs.sort()
            ls = "-" if face is None else "--"
            axes[1].plot(*zip(*xs), color=COLOURS[st], lw=0.8, ls=ls, zorder=2)
            if len(pts) == 3:
                f = fits.setdefault(st, {})[size] = fit_slope(pts, drw)
                xx = np.array(f["x"])
                axes[0].plot(
                    xx, 10 ** (f["intercept"] + f["gamma"] * np.log10(xx)), color=COLOURS[st], lw=0.8, ls=ls, zorder=2
                )
                lo, hi = f["gamma_ci"]
                axes[0].plot(
                    [],
                    [],
                    color=COLOURS[st],
                    ls=ls,
                    label=f"{st}, {label.split()[0]}: γ = {f['gamma']:.2f} [{lo:.2f}, {hi:.2f}]",
                )
    for size, (_, _, face) in SIZES.items():  # base model excess
        xs = []
        for length in LENGTHS:
            run = f"zipf-{size}-s0-{length}"
            if (D.GT / "compress" / run / "base.npz").exists():
                est, dr = boot(per_sequence(run, "base", "excess"))
                _err(axes[1], tokens_per_param(run), est, *ci(dr), "black", face)
                xs.append((tokens_per_param(run), est))
        if xs:
            axes[1].plot(*zip(*sorted(xs)), color="black", lw=0.8, ls="-" if face is None else "--", zorder=2)
    for label, _, face in SIZES.values():
        axes[2].plot([], [], "o", color="0.3", mfc=face or "0.3", ms=4, label=label)
    axes[1].plot([], [], color="black", label="base")
    for st in FOCUS:
        axes[1].plot([], [], color=COLOURS[st], label=st)
    axes[2].axhline(0.5, color="0.6", lw=0.6, zorder=0)
    titles = [
        "damage: ΔKL to Bayes (nats/token)",
        "excess over Bayes after compression (nats/token)",
        "tracking AUC (< 0.5: surer where damaged)",
    ]
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xlabel("training tokens per parameter", fontsize=7)
        ax.set_title(t, fontsize=8)
        _style(ax)
        ax.tick_params(which="minor", labelsize=6)
    for ax in axes:
        ax.legend(fontsize=6, frameon=False)
    fig.text(
        0.01,
        0.01,
        "Filled: 5.2M parameters (solid lines); open: 0.9M (dashed). Seed 0. Bars: 95% bootstrap "
        "over 1,024 sequences. Left: least-squares line per size through three points; γ interval covers sampling only.",
        fontsize=6,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "scaling.png", dpi=200)
    plt.close(fig)
    return fits


def seeds(plt, out):
    """MAIN: every setting, four training seeds."""
    runs = [f"zipf-gt-6m-q-s{s}-long" for s in range(4) if (D.GT / "compress" / f"zipf-gt-6m-q-s{s}-long").exists()]
    reps = [report(r) for r in runs]
    settings = [s for s in reps[0] if s != "base" and "@" not in s and all(s in r for r in reps)]
    settings.sort(key=lambda s: np.mean([r[s]["dkl"]["est"] for r in reps]))
    fig, axes = plt.subplots(1, 2, figsize=(8, 0.22 * len(settings) + 1.2), sharey=True)
    for i, st in enumerate(settings):
        for j, r in enumerate(reps):
            y = i + (j - (len(reps) - 1) / 2) * 0.15
            for ax, key in zip(axes, ("dkl", "tracking_auc")):
                v = r[st][key]
                ax.plot([v["lo"], v["hi"]], [y, y], color="0.55", lw=0.8, zorder=1)
                ax.plot(v["est"], y, "o", ms=2.5, color="black", zorder=2)
    axes[0].set_yticks(range(len(settings)), settings, fontsize=7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("damage ΔKL (nats/token)", fontsize=7)
    axes[1].axvline(0.5, color="0.7", lw=0.6)
    axes[1].set_xlabel("tracking AUC", fontsize=7)
    for ax in axes:
        _style(ax)
    fig.suptitle(
        f"{MAIN.replace('-s0', '')}: one row per training seed ({len(reps)}); bars 95% bootstrap over sequences",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out / "seeds.png", dpi=200)
    plt.close(fig)


def inputs(plt, out):
    """The input distribution: damage and tracking AUC per world variant, at 5.2M parameters, 32k steps."""
    worlds = [w for w in VARIANTS if (D.GT / "compress" / f"{w}-gt-6m-q-s0-long" / "report.json").exists()]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
    for i, w in enumerate(worlds):
        r = report(f"{w}-gt-6m-q-s0-long")
        for k, st in enumerate(FOCUS):
            y = i + (k - 1) * 0.2
            for ax, key in zip(axes, ("dkl", "tracking_auc")):
                v = r[st][key]
                if r[st]["dkl"]["est"] < 1e-4:  # no damage: nothing to draw, and the AUC is undefined in practice
                    continue
                ax.plot([v["lo"], v["hi"]], [y, y], color=COLOURS[st], lw=1, zorder=1)
                ax.plot(
                    v["est"], y, "o", ms=3, color=COLOURS[st], zorder=2, label=st if i == 0 and key == "dkl" else None
                )
    blurbs = {w: P.worlds()[w].blurb.split(":")[0][:60] for w in worlds}
    axes[0].set_yticks(range(len(worlds)), [f"{w}" for w in worlds], fontsize=7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("damage ΔKL (nats/token)", fontsize=7)
    axes[1].axvline(0.5, color="0.7", lw=0.6)
    axes[1].set_xlabel("tracking AUC", fontsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=6, frameon=False)
    for ax in axes:
        _style(ax)
    fig.suptitle(
        "Input distribution (5.2M parameters, 51 tokens/param, seed 0); bars 95% bootstrap. "
        "Settings with damage below 1e-4 (zipf-iid: GPTQ-2, WANDA-50) are not drawn.",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out / "inputs.png", dpi=200)
    plt.close(fig)
    return blurbs


def blocks(plt, out):
    """Single-block compression on MAIN: damage and entropy gap per block."""
    r = report(MAIN)
    if "gptq-2@0" not in r:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for k, st in enumerate(["gptq-2", "wanda-50"]):
        for b in range(6):
            x = b + (k - 0.5) * 0.2
            for ax, key in zip(axes, ("dkl", "entropy_gap")):
                v = r[f"{st}@{b}"][key]
                ax.plot([x, x], [v["lo"], v["hi"]], color=COLOURS[st], lw=1)
                ax.plot(x, v["est"], "o", ms=3, color=COLOURS[st], label=st if b == 0 and key == "dkl" else None)
        for ax, key in zip(axes, ("dkl", "entropy_gap")):
            ax.axhline(r[st][key]["est"], color=COLOURS[st], lw=0.6, ls=":")
    axes[1].axhline(r["base"]["entropy_gap"]["est"], color="black", lw=0.6, ls="--")
    axes[1].axhline(0, color="0.7", lw=0.5)
    axes[0].set_ylabel("damage ΔKL (nats/token)", fontsize=7)
    axes[1].set_ylabel("entropy gap (nats; < 0 overconfident)", fontsize=7)
    for ax in axes:
        ax.set_xticks(range(6))
        ax.set_xlabel("compressed block (all others as trained)", fontsize=7)
        _style(ax)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{MAIN}: one block compressed. Dotted: all blocks; dashed: base. Bars 95% bootstrap.", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "blocks.png", dpi=200)
    plt.close(fig)


def learning(plt, out):
    """Excess over Bayes during training (validation every eval_every steps; band ± 2 SE over sequences)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for size, (label, _, face) in SIZES.items():
        for length, lw in zip(LENGTHS, (0.8, 1.1, 1.5)):
            run = f"zipf-{size}-s0-{length}"
            f = D.GT / "runs" / run / "trace.jsonl"
            if not f.exists():
                continue
            v = [json.loads(line) for line in f.read_text().splitlines() if '"valid"' in line]
            v = [r for r in v if r["tokens"] > 0]
            t = np.array([r["tokens"] for r in v]) / _cfg(run)["parameters"]
            e = np.array([r["valid"]["excess"] for r in v])
            se = np.array([r["valid"]["excess_se"] for r in v])
            ls = "-" if face is None else "--"
            ax.fill_between(t, e - 2 * se, e + 2 * se, color="0.85", lw=0, zorder=0)
            ax.plot(t, e, color="black", lw=lw, ls=ls, zorder=1)
            ax.text(t[-1] * 1.05, e[-1], f"{label.split()[0]}, {length}", fontsize=5, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens seen per parameter", fontsize=7)
    ax.set_ylabel("excess over Bayes (nats/token)", fontsize=7)
    ax.set_title("Training: each run's cosine schedule ends at its last point", fontsize=7)
    _style(ax)
    ax.tick_params(which="minor", labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=200)
    plt.close(fig)


def run(say=print):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .train import provenance

    out = D.GT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fits = scaling(plt, out)
    seeds(plt, out)
    inputs(plt, out)
    blocks(plt, out)
    learning(plt, out)
    canvas(out)
    (out / "fits.json").write_text(json.dumps({"scaling": fits, "provenance": provenance()}, indent=1))
    for st, by_size in fits.items():
        for size, f in by_size.items():
            lo, hi = f["gamma_ci"]
            say(
                f"[gt figures] {st} {size}: ΔKL ∝ (tokens/param)^γ, γ = {f['gamma']:.2f} [{lo:.2f}, {hi:.2f}], "
                f"max |log10 residual| {f['max_log10_residual']:.2f}"
            )
    say(f"[gt figures] wrote {out}")


# ── canvas ────────────────────────────────────────────────────────────────────────────────────────


def _png_size(path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def canvas(out):
    """<GT>/figures/figures.canvas (JSON Canvas, https://jsoncanvas.org): every figure as a picture card,
    in labelled groups, with a one-line caption. File paths are relative to the canvas."""
    G = D.GT
    rows = [
        (
            "Scaling with training (seed 0)",
            [
                (
                    "figures/scaling.png",
                    "Damage, excess after compression and tracking AUC vs tokens per parameter; "
                    "per-size power-law fits (fits.json).",
                ),
                ("figures/learning.png", "Excess over Bayes during training, ± 2 SE."),
                ("control/tokens_per_param.png", "Damage vs tokens per parameter, every setting in gt control."),
            ],
        ),
        (
            "Settings and seeds (MAIN)",
            [
                ("figures/seeds.png", "All settings, four training seeds, 95% intervals."),
                ("figures/blocks.png", "One block compressed at a time."),
            ],
        ),
        (
            "Input distribution and exposure",
            [
                ("figures/inputs.png", "Damage and tracking AUC per world variant."),
                ("control/exposure.png", "Added prior error per symbol vs training exposure; pooled vs per-run lines."),
            ],
        ),
        (
            "Inside, layer by layer",
            [
                (f"inside/{r}/{name}", f"{r}: {what}")
                for r in sorted(p.name for p in (G / "inside").iterdir() if p.is_dir())
                for name, what in (("lens.png", "logit and Jacobian lens"), ("layers.png", "UMAP, models x layers"))
            ],
        ),
        ("All worlds", [("ladder.png", "Loss against the Bayes floor, per world.")]),
    ]
    nodes, y, width = [], 0, 900
    for g, (title, items) in enumerate(rows):
        items = [(f, c) for f, c in items if (G / f).exists()]
        if not items:
            continue
        x, top, tallest = 40, y + 60, 0
        for i, (f, cap) in enumerate(items):
            w, h = _png_size(G / f)
            hh = int(width * h / w)
            nodes.append(
                {"id": f"f{g}-{i}", "type": "file", "file": "../" + f, "x": x, "y": top, "width": width, "height": hh}
            )
            nodes.append(
                {
                    "id": f"t{g}-{i}",
                    "type": "text",
                    "text": cap,
                    "x": x,
                    "y": top + hh + 10,
                    "width": width,
                    "height": 60,
                }
            )
            x += width + 60
            tallest = max(tallest, hh + 70)
        nodes.insert(
            0, {"id": f"g{g}", "type": "group", "label": title, "x": 0, "y": y, "width": x, "height": tallest + 100}
        )
        y += tallest + 160
    text = (
        "# Zipf compression experiments\n\nGenerated by `pixi run gt-figures`. Bars and bands: 95% "
        "bootstrap over validation sequences; seeds shown separately. Methods: GT.md."
    )
    nodes.insert(0, {"id": "title", "type": "text", "text": text, "x": 0, "y": -200, "width": 900, "height": 140})
    (out / "figures.canvas").write_text(json.dumps({"nodes": nodes, "edges": []}, indent=1))
