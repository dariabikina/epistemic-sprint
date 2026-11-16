# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Damage against per-symbol training exposure and tokens per parameter; agreement across seeds.

    gt control RUN,RUN,... [--settings ...]  ->  <GT>/control/{report.json, exposure.png, tokens_per_param.png}
    gt seeds RUN,RUN,...                     ->  <GT>/seeds/NAME.json

exposure_k      expected number of training documents containing symbol k: (fraction of pool sequences
                containing k) x steps x batch size.
derr_k          |log q1_comp(k) - log G0(k)| - |log q1_base(k) - log G0(k)|, q1 = predictive at BOS.
collapse        least-squares fits of derr = a + b log10(exposure): one line over all runs pooled, and one
                line per run. Reports a, b, R^2 of the pooled line, R^2 with a separate line per run, and
                each run's slope. Intervals on a, b: bootstrap resampling runs and, jointly, one set of
                symbol indices shared by all runs (B = 1,000). Exposure below 1 is set to 1 (log10 = 0).
tokens/param    steps x batch x 256 / parameters, against each run's dkl from its compress report.
across_seeds    per setting and metric: each seed's estimate and interval, the mean, the between-seed SD.
"""

import json
from pathlib import Path

import numpy as np

from . import data as D
from . import priors as P
from .stats import fmt, interval
from .train import provenance


def exposure(run_name: str) -> np.ndarray:
    """Per symbol: (fraction of pool sequences containing it) x sequences drawn in training."""
    cfg = json.loads((D.GT / "runs" / run_name / "config.json").read_text())
    xt = np.load(Path(cfg["data"]) / "train.npy", mmap_mode="r")
    present = np.zeros(P.NBINS)
    for s in range(0, len(xt), 20000):
        part = np.asarray(xt[s : s + 20000]).astype(np.int64)
        present += np.eye(P.NBINS, dtype=bool)[part].any(1).sum(0)
    return present / len(xt) * cfg["steps"] * cfg["batch_size"]


def added_prior_error(run_name: str, setting: str) -> np.ndarray | None:
    cdir = D.GT / "compress" / run_name
    f = cdir / f"{setting.replace(':', 'of')}.npz"
    if not f.exists():
        return None
    world = json.loads((D.GT / "runs" / run_name / "config.json").read_text())["world"]
    p0 = P.worlds()[world].p0()
    err = lambda q1: np.abs(np.log(np.maximum(q1, 1e-30)) - np.log(p0))
    return err(np.load(f)["q1"]) - err(np.load(cdir / "base.npz")["q1"])


def _line(x, y):
    A = np.c_[np.ones_like(x), x]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, r2


def collapse(per_run: dict[str, tuple[np.ndarray, np.ndarray]], B: int = 1000, seed: int = 0) -> dict:
    """per_run: run -> (log10 exposure, derr). Pooled vs per-run linear fits, with bootstrap intervals."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([v[0] for v in per_run.values()])
    y = np.concatenate([v[1] for v in per_run.values()])
    (a, b), r2_pooled = _line(x, y)
    # per-run fits: residual variance with each run's own line
    ss_res = sum(((yy - np.c_[np.ones_like(xx), xx] @ _line(xx, yy)[0]) ** 2).sum() for xx, yy in per_run.values())
    r2_separate = 1 - ss_res / ((y - y.mean()) ** 2).sum()
    runs = list(per_run.values())
    n_sym = len(runs[0][0])  # every run has the same 256 symbols, in the same order
    draws = []
    for _ in range(B):
        r = rng.integers(0, len(runs), len(runs))
        i = rng.integers(0, n_sym, n_sym)  # one draw of symbols, shared by all runs
        draws.append(_line(np.concatenate([runs[j][0][i] for j in r]), np.concatenate([runs[j][1][i] for j in r]))[0])
    draws = np.array(draws)
    return {
        "intercept": interval(a, draws[:, 0]),
        "slope_per_decade": interval(b, draws[:, 1]),
        "r2_pooled": float(r2_pooled),
        "r2_separate_lines": float(r2_separate),
        "per_run_slope": {r: float(_line(*v)[0][1]) for r, v in per_run.items()},
    }


def run(run_names: list[str], settings: list[str], say=print) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = D.GT / "control"
    out.mkdir(parents=True, exist_ok=True)
    rows, fits, curves = [], {}, {}
    for st in settings:
        per_run = {}
        for rn in run_names:
            d = added_prior_error(rn, st)
            rep_f = D.GT / "compress" / rn / "report.json"
            if d is None or not rep_f.exists():
                continue
            cfg = json.loads((D.GT / "runs" / rn / "config.json").read_text())
            E = exposure(rn)
            per_run[rn] = (np.log10(np.maximum(E, 1.0)), d)
            r = json.loads(rep_f.read_text())["settings"][st]
            rows.append(
                {
                    "run": rn,
                    "world": cfg["world"],
                    "shape": cfg["shape"],
                    "seed": cfg["seed"],
                    "parameters": cfg["parameters"],
                    "steps": cfg["steps"],
                    "tokens_per_param": cfg["steps"] * cfg["batch_size"] * P.T / cfg["parameters"],
                    "setting": st,
                    "dkl": r["dkl"],
                    "d_excess_new": r["d_excess_new"],
                    "d_excess_repeat": r["d_excess_repeat"],
                    "prior_kl": r["prior_kl"],
                    "tracking_auc": r["tracking_auc"],
                    "entropy_gap": r["entropy_gap"],
                }
            )
        if len(per_run) >= 2:
            fits[st] = collapse(per_run)
        curves[st] = per_run
    (out / "report.json").write_text(
        json.dumps(
            {"runs": rows, "collapse": fits, "provenance": provenance(), "settings": settings, "run_names": run_names},
            indent=1,
        )
    )
    _plot_exposure(curves, fits, out / "exposure.png", plt)
    _plot_tpp(rows, settings, out / "tokens_per_param.png", plt)
    say(
        "| run | setting | tokens/param | damage ΔKL | Δexcess new | Δexcess repeat | tracking AUC |\n|---|---|---|---|---|---|---|"
    )
    for r in rows:
        say(
            f"| {r['run']} | {r['setting']} | {r['tokens_per_param']:.0f} | {fmt(r['dkl'], 4, True)} | "
            f"{fmt(r['d_excess_new'], 3, True)} | {fmt(r['d_excess_repeat'], 3, True)} | {fmt(r['tracking_auc'], 2)} |"
        )
    for st, f in fits.items():
        say(
            f"[gt control] {st}: added prior error = {fmt(f['intercept'])} + {fmt(f['slope_per_decade'], 3, True)} x log10(exposure); "
            f"R² pooled {f['r2_pooled']:.2f} vs separate lines {f['r2_separate_lines']:.2f}"
        )
    return out


def _plot_exposure(curves, fits, path, plt):
    settings = [s for s in curves if curves[s]]
    if not settings:
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(3.4 * len(settings), 3.5), squeeze=False, sharey=True)
    names = sorted({r for s in settings for r in curves[s]})
    for ax, st in zip(axes[0], settings):
        for c, rn in enumerate(names):
            if rn not in curves[st]:
                continue
            x, y = curves[st][rn]
            ax.scatter(10**x, y, s=3, color=f"C{c % 10}", alpha=0.3, lw=0, zorder=1)
            edges = np.linspace(x.min(), x.max(), 9)
            idx = np.clip(np.digitize(x, edges) - 1, 0, 7)
            med = [(10 ** np.median(x[idx == b]), np.median(y[idx == b])) for b in range(8) if (idx == b).sum() >= 4]
            if med:
                ax.plot(
                    *zip(*med), color=f"C{c % 10}", lw=1.4, zorder=2, label=rn.replace("-gt-", " ").replace("-s0", "")
                )
        if st in fits:
            f = fits[st]
            xx = np.linspace(
                min(v[0].min() for v in curves[st].values()), max(v[0].max() for v in curves[st].values()), 50
            )
            ax.plot(
                10**xx,
                f["intercept"]["est"] + f["slope_per_decade"]["est"] * xx,
                color="black",
                lw=1,
                ls="--",
                zorder=3,
            )
            ax.text(
                0.02,
                0.98,
                f"pooled R² {f['r2_pooled']:.2f}\nseparate {f['r2_separate_lines']:.2f}",
                transform=ax.transAxes,
                fontsize=6,
                va="top",
            )
        ax.set_xscale("log")
        ax.axhline(0, color="0.7", lw=0.6, zorder=0)
        ax.set_title(st, fontsize=8)
        ax.set_xlabel("exposure: training documents containing the symbol", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("added error in the stored prior (nats)\n|log q1 − log G0|, compressed − base", fontsize=7)
    axes[0, -1].legend(fontsize=5, frameon=False, loc="upper right")
    fig.suptitle(
        "Is exposure the control variable? Dots: symbols; lines: binned medians per run; dashed: one line for all runs",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_tpp(rows, settings, path, plt):
    zipf = [r for r in rows if r["world"] == "zipf"]
    if not zipf:
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(3.2 * len(settings), 3.2), squeeze=False)
    for ax, st in zip(axes[0], settings):
        for c, shape in enumerate(sorted({r["shape"] for r in zipf})):
            pts = sorted(
                ((r["tokens_per_param"], r["dkl"]) for r in zipf if r["setting"] == st and r["shape"] == shape),
                key=lambda p: p[0],
            )  # seeds share an x; never compare the interval dicts
            if not pts:
                continue
            t = [p[0] for p in pts]
            ax.errorbar(
                t,
                [p[1]["est"] for p in pts],
                yerr=[[p[1]["est"] - p[1]["lo"] for p in pts], [p[1]["hi"] - p[1]["est"] for p in pts]],
                color=f"C{c}",
                marker="o",
                ms=3,
                lw=1.2,
                capsize=2,
                label=shape,
            )
        ax.set_xscale("log")
        ax.set_title(st, fontsize=8)
        ax.set_xlabel("training tokens per parameter", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("damage ΔKL (nats/token, 95% CI)", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── seeds ─────────────────────────────────────────────────────────────────────────────────────────

SEED_METRICS = ["dkl", "d_excess_new", "d_excess_repeat", "d_excess_kind", "entropy_gap", "tracking_auc"]


def across_seeds(run_names: list[str], settings: list[str] | None = None, say=print) -> dict:
    """Runs that differ only in training seed: per setting and metric, each seed's estimate, mean, SD (ddof 1)."""
    reports = {rn: json.loads((D.GT / "compress" / rn / "report.json").read_text())["settings"] for rn in run_names}
    common = [s for s in next(iter(reports.values())) if all(s in r for r in reports.values())]
    out = {}
    say("| setting | metric | " + " | ".join(run_names) + " | mean | between-seed SD |")
    say("|---|---|" + "---|" * (len(run_names) + 2))
    for st in settings or common:
        if st not in common or st == "base":
            continue
        out[st] = {}
        for m in SEED_METRICS:
            vals = [reports[rn][st][m] for rn in run_names]
            est = np.array([v["est"] for v in vals])
            sd = float(est.std(ddof=1)) if len(est) > 1 else float("nan")
            out[st][m] = {"per_seed": vals, "mean": float(est.mean()), "between_seed_sd": sd}
            say(f"| {st} | {m} | " + " | ".join(fmt(v, 3) for v in vals) + f" | {est.mean():.3f} | {sd:.3f} |")
    return out
