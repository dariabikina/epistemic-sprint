# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Measures, intervals and figures for the confidence experiments (L1-L3 in PLAN.md). CPU only.

    wu confidence-report --model qwen3-8b   ->  <RUNS>/<model>/confidence-report.json,
                                                confidence-{facts,marbles,inside}.png

Reads each setting's confidence-facts-3000.npz and confidence-marbles-900.npz (`wu confidence`) and its graded
replies. Every number is paired with the base model on the same questions or items.

Facts, per question (c = the reply is right, 0 or 1):
    pred     exp(logprob_sum): the probability the model gave its own reply
    stated   (P(Yes | "correct?") + P(No | "incorrect?")) / 2, for its own reply
    judged   the same for the base model's reply: every setting judges the same strings
    p_gold   the summed probability of the accepted replies
    lost     right before compression, wrong after
  Reported: mean(pred) - accuracy and mean(stated) - accuracy (overconfidence); both on lost facts, where the
  truth is 0; the area under the ROC curve of each for c; the Yes-lean (P(Yes | "correct?") +
  P(Yes | "incorrect?") - 1; 0 = none).

Marbles, per item (p = exact probabilities of its colours):
    q        the model's next-token probabilities of the item's colour words, renormalized over them
    kl       KL(p || q); damage = kl(compressed) - kl(base)
    overbet  q - p at the likeliest colour
    worst    the 10% of items whose kl grew most; d_entropy_worst = mean H(q_comp) - H(q_base) there
    stated   expected probability from the digit reply; ideal = the digit bin that contains p

Inside (per recorded layer):
    lens     the model's own readout through the logit lens: AUROC for c (facts), KL to p (marbles)
    probe    a ridge probe on the first 256 principal components of the residual stream (components fitted on
             all rows; the ridge fit is cross-fitted), predicting c
             (facts) or logit p of the asked colour (marbles). `same`: PCA and probe fitted on the base model's
             states, applied to the compressed model's. `refit`: fitted on the compressed model's own states.

Intervals: 95% percentile bootstrap over questions or items (B = 1,000, seed 0); probes are not refitted
within the bootstrap.
"""

import json
import warnings

import numpy as np

from . import data, urn
from .gt.stats import auc
from .models import RUNS, setting_dir

B, PCS, RIDGE, FOLDS = 1000, 256, 10.0, 5
SETTINGS = [
    "rtn-4",
    "rtn-3",
    "rtn-2",
    "gptq-4",
    "gptq-3",
    "awq-4",
    "awq-3",
    "magnitude-30",
    "wanda-30",
    "wanda-50",
    "wanda-2:4",
    "shortgpt-5",
    "shortgpt-10",
]
LABELS = {
    "base": "uncompressed",
    "rtn-4": "RTN 4-bit",
    "rtn-3": "RTN 3-bit",
    "rtn-2": "RTN 2-bit",
    "gptq-4": "GPTQ 4-bit",
    "gptq-3": "GPTQ 3-bit",
    "awq-4": "AWQ 4-bit",
    "awq-3": "AWQ 3-bit",
    "magnitude-30": "magnitude 30%",
    "wanda-30": "WANDA 30%",
    "wanda-50": "WANDA 50%",
    "wanda-2:4": "WANDA 2:4",
    "shortgpt-5": "5 blocks removed",
    "shortgpt-10": "10 blocks removed",
}


def _interval(fn, n: int, idx: np.ndarray) -> dict:
    """fn(indices) -> a number; estimate on all n and a percentile interval over the rows of idx. A group
    that is empty (the base model has no lost facts) gives NaN."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        draws = np.array([fn(j) for j in idx])
        if np.isnan(draws).all():
            return {"est": float("nan"), "lo": float("nan"), "hi": float("nan")}
        lo, hi = np.nanquantile(draws, [0.025, 0.975])
        return {"est": float(fn(np.arange(n))), "lo": float(lo), "hi": float(hi)}


def _entropy(q: np.ndarray) -> np.ndarray:
    return -(q * np.log(np.maximum(q, 1e-12))).sum(-1)


# ── facts ──────────────────────────────────────────────────────────────────────────────────────────


def load_facts(model: str, setting: str) -> dict | None:
    d = setting_dir(model, setting)
    if not (d / "confidence-facts-3000.npz").exists():
        return None
    z = dict(np.load(d / "confidence-facts-3000.npz"))
    rec = [json.loads(line) for line in (d / "eval-short-3000.jsonl").read_text().split("\n") if line]
    assert [r["id"] for r in rec] == z["ids"].tolist()
    two = lambda who: (z[f"{who}_correct_p_yes"] + 1 - z[f"{who}_wrong_p_yes"]) / 2
    return {
        "c": np.array([r["correct"] for r in rec]),
        "pred": np.exp([r["logprob_sum"] for r in rec]),
        "unfinished": np.array([not r["finished"] for r in rec]),
        "stated": two("own"),
        "judged": two("base"),
        "gold": two("gold"),
        "lean": z["own_correct_p_yes"] + z["own_wrong_p_yes"] - 1,
        "self": z["self_correct_p_yes"],
        "p_gold": np.exp(z["logp_gold"]),
        "layers": z["layers"],
        "lens": z["lens_yes_no"][..., 0] - z["lens_yes_no"][..., 1],  # log-odds of Yes at each recorded layer
        "H": z["H"],
    }


def facts_measures(base: dict, comp: dict, idx: np.ndarray) -> dict:
    n = len(base["c"])
    c, cb = comp["c"], base["c"]
    lost, kept = cb & ~c, cb & c
    mean = lambda x, mask=None: lambda j: x[j].mean() if mask is None else x[j][mask[j]].mean()
    out = {
        "accuracy": _interval(mean(c.astype(float)), n, idx),
        "accuracy_lost_pp": _interval(lambda j: 100 * (cb[j].mean() - c[j].mean()), n, idx),
        "p_gold": _interval(mean(comp["p_gold"]), n, idx),
        "unfinished": float(comp["unfinished"].mean()),
        "n_lost": int(lost.sum()),
        "yes_lean": _interval(mean(comp["lean"]), n, idx),
        "self_p_yes": float(comp["self"].mean()),
        "judged_auroc": _interval(lambda j: auc(comp["judged"][j], cb[j].astype(float)), n, idx),
        "gold_on_lost": _interval(mean(comp["gold"], lost), n, idx),
        "gold_on_kept": _interval(mean(comp["gold"], kept), n, idx),
    }
    for k in ("pred", "stated"):
        x, xb = comp[k], base[k]
        out[k] = {
            "over_all_pp": _interval(lambda j: 100 * (x[j].mean() - c[j].mean()), n, idx),
            "on_lost_pp": _interval(lambda j: 100 * x[j][lost[j]].mean(), n, idx),
            "on_kept_pp": _interval(lambda j: 100 * x[j][kept[j]].mean(), n, idx),
            "base_on_lost_pp": _interval(lambda j: 100 * xb[j][lost[j]].mean(), n, idx),
            "auroc": _interval(lambda j: auc(x[j], c[j].astype(float)), n, idx),
            "d_auroc": _interval(lambda j: auc(x[j], c[j].astype(float)) - auc(xb[j], cb[j].astype(float)), n, idx),
        }
    return out


# ── marbles ────────────────────────────────────────────────────────────────────────────────────────


def load_marbles(model: str, setting: str, items: list[dict]) -> dict | None:
    path = setting_dir(model, setting) / "confidence-marbles-900.npz"
    if not path.exists():
        return None
    z = dict(np.load(path))
    assert z["ids"].tolist() == [it["id"] for it in items]
    k_max = 4
    p = np.zeros((len(items), k_max))
    cols = np.zeros((len(items), k_max), int)
    for i, it in enumerate(items):
        p[i, : it["k"]] = it["p"]
        cols[i, : it["k"]] = [urn.COLOURS.index(c) for c in it["colours"]]
    present = p > 0
    rows = np.arange(len(items))[:, None]

    def renorm(logp):  # (items, ..., 8 colours) -> probabilities over the item's own colours
        raw = np.where(present, np.exp(logp[rows, cols]), 0.0)
        return raw / raw.sum(-1, keepdims=True), raw.sum(-1)

    q, mass = renorm(z["predict_logp"])
    q_lens = np.stack([renorm(z["predict_lens"][:, j])[0] for j in range(z["predict_lens"].shape[1])], 1)
    kl = lambda qq: np.where(present, p * np.log(np.maximum(p, 1e-12) / np.maximum(qq, 1e-12)), 0).sum(-1)
    digits = np.exp(z["state_logp"])
    truth = p[z["state_item"], z["state_colour"]]
    return {
        "kind": np.array([it["kind"] for it in items]),
        "p": p,
        "q": q,
        "mass": mass,
        "kl": kl(q),
        "kl_lens": np.stack([kl(q_lens[:, j]) for j in range(q_lens.shape[1])], 1),
        "entropy": _entropy(q),
        "overbet": q[rows[:, 0], p.argmax(1)] - p.max(1),
        "state_item": z["state_item"],
        "truth": truth,
        "stated": (digits * (np.arange(10) + 0.5) / 10).sum(1) / digits.sum(1),
        "ideal": (np.minimum(np.floor(truth * 10 + 1e-9), 9) + 0.5) / 10,
        "digit_mass": digits.sum(1),
        "layers": z["layers"],
        "state_H": z["state_H"],
    }


def marbles_measures(base: dict, comp: dict, idx: np.ndarray, kind: str) -> dict:
    """Items of one kind. The bootstrap resamples items; each item's stated questions go with it."""
    sel = np.flatnonzero(comp["kind"] == kind)
    n = len(sel)
    j_all = idx[:, :n] % n  # (B, n) resamples of the positions in sel
    asks = [np.flatnonzero(comp["state_item"] == i) for i in sel]
    dk = comp["kl"][sel] - base["kl"][sel]
    dh = comp["entropy"][sel] - base["entropy"][sel]

    def worst(j):
        return dh[j][dk[j] >= np.quantile(dk[j], 0.9)].mean()

    def over_asks(fn):  # fn(ask indices) -> number, on the asks of the resampled items
        return lambda j: fn(np.concatenate([asks[t] for t in j]))

    item = lambda x: lambda j: x[sel][j].mean()
    return {
        "kl": _interval(item(comp["kl"]), n, j_all),
        "damage": _interval(lambda j: dk[j].mean(), n, j_all),
        "colour_mass": float(comp["mass"][sel].mean()),
        "overbet": _interval(item(comp["overbet"]), n, j_all),
        "d_overbet": _interval(lambda j: (comp["overbet"][sel] - base["overbet"][sel])[j].mean(), n, j_all),
        "d_entropy": _interval(lambda j: dh[j].mean(), n, j_all),
        "d_entropy_worst": _interval(worst, n, j_all),
        "stated_minus_ideal": _interval(over_asks(lambda a: (comp["stated"] - comp["ideal"])[a].mean()), n, j_all),
        "stated_abs_error": _interval(over_asks(lambda a: np.abs(comp["stated"] - comp["ideal"])[a].mean()), n, j_all),
        "stated_slope": _interval(over_asks(lambda a: np.polyfit(comp["truth"][a], comp["stated"][a], 1)[0]), n, j_all),
        "digit_mass": float(comp["digit_mass"][np.concatenate(asks)].mean()),
    }


# ── inside: probes on the residual stream ──────────────────────────────────────────────────────────────


def pca_scores(X: np.ndarray, basis=None):
    """Scores on the first PCS principal components. `basis` = (mean, components) from another call; None
    fits it on X. The components are fitted on all rows (they use no labels); the probe on them is
    cross-fitted."""
    X = X.astype(np.float32)
    if basis is None:
        mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
        basis = (mu, vt[:PCS].T)
    return (X - basis[0]) @ basis[1], basis


def probe_scores(Z_fit: np.ndarray, y_fit: np.ndarray, Z_apply: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cross-fitted ridge predictions for Z_apply from a probe fitted on Z_fit, y_fit. Rows with the same
    group (question, or marble item) stay in one fold; row i of Z_apply is scored by a probe that did not
    see row i of Z_fit."""
    fold = np.random.default_rng(0).integers(0, FOLDS, groups.max() + 1)[groups]
    out = np.zeros(len(Z_apply))
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        sd = Z_fit[tr].std(0) + 1e-6
        A = np.c_[Z_fit[tr] / sd, np.ones(tr.sum())]
        reg = RIDGE * np.eye(A.shape[1])
        reg[-1, -1] = 0  # the intercept is not shrunk
        w = np.linalg.solve(A.T @ A + reg, A.T @ y_fit[tr])
        out[te] = np.c_[Z_apply[te] / sd, np.ones(te.sum())] @ w
    return out


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _both_probes(H_base, y_base, H_comp, y_comp, groups, cache: dict, key):
    """Predictions for the compressed model's states: probe fitted on the base model's (`same`) and on its
    own (`refit`). The base model's components are computed once per layer and kept in `cache`."""
    if key not in cache:
        cache[key] = pca_scores(H_base)
    Zb, basis = cache[key]
    same = probe_scores(Zb, y_base, pca_scores(H_comp, basis)[0], groups)
    Zc = Zb if H_comp is H_base else pca_scores(H_comp)[0]
    return same, probe_scores(Zc, y_comp, Zc, groups)


def inside_facts(base: dict, comp: dict, cache: dict) -> list[dict]:
    y, yb = comp["c"].astype(float), base["c"].astype(float)
    q = np.arange(len(y))
    rows = []
    for j, layer in enumerate(comp["layers"]):
        same, refit = _both_probes(base["H"][:, j], yb, comp["H"][:, j], y, q, cache, ("facts", j))
        rows.append(
            {
                "layer": int(layer),
                "lens_auroc": auc(comp["lens"][:, j], y),
                "probe_same_auroc": auc(same, y),
                "probe_refit_auroc": auc(refit, y),
            }
        )
    return rows


def inside_marbles(base: dict, comp: dict, cache: dict) -> list[dict]:
    y = np.clip(comp["truth"], 0.02, 0.98)
    y = np.log(y / (1 - y))  # the exact probability of the asked colour, as log-odds
    rows = []
    for j, layer in enumerate(comp["layers"]):
        same, refit = _both_probes(
            base["state_H"][:, j], y, comp["state_H"][:, j], y, comp["state_item"], cache, ("marbles", j)
        )
        rows.append(
            {
                "layer": int(layer),
                "predict_kl_lens": float(comp["kl_lens"][:, j].mean()),
                "probe_same_r2": _r2(same, y),
                "probe_refit_r2": _r2(refit, y),
            }
        )
    return rows


# ── report ─────────────────────────────────────────────────────────────────────────────────────────────


def redraw(model: str, say=print):
    """The figures again from an existing confidence-report.json."""
    out = RUNS / model
    figures(json.loads((out / "confidence-report.json").read_text()), out)
    say(f"[confidence-report] redrew the figures in {out}")


def run(model: str, inside: bool = True, say=print) -> dict:
    items = urn.marbles()
    idx_q = np.random.default_rng(0).integers(0, 3000, (B, 3000))
    idx_m = np.random.default_rng(0).integers(0, 10**6, (B, len(items)))
    base_f, base_m = load_facts(model, "base"), load_marbles(model, "base", items)
    report = {"model": model, "definitions": __doc__, "facts": {}, "marbles": {}, "inside": {}}
    cache = {}
    if base_f:
        assert [r["id"] for r in data.popqa(3000)] == np.load(setting_dir(model, "base") / "confidence-facts-3000.npz")[
            "ids"
        ].tolist()
    for s in ["base", *SETTINGS]:
        f, m = load_facts(model, s), load_marbles(model, s, items)
        if base_f and f:
            report["facts"][s] = facts_measures(base_f, f, idx_q)
        if base_m and m:
            report["marbles"][s] = {k: marbles_measures(base_m, m, idx_m, k) for k in ("bag", "draws")}
        if inside and (f or m):
            report["inside"][s] = {
                "facts": inside_facts(base_f, base_f if s == "base" else f, cache) if base_f and f else [],
                "marbles": inside_marbles(base_m, base_m if s == "base" else m, cache) if base_m and m else [],
            }
        if f or m:
            say(f"[confidence-report] {model} {s}: " + _line(report, s))
    out = RUNS / model
    (out / "confidence-report.json").write_text(json.dumps(report, indent=1) + "\n")
    figures(report, out)
    say(f"[confidence-report] wrote {out / 'confidence-report.json'} and figures")
    return report


def _line(report: dict, s: str) -> str:
    parts = []
    if s in report["facts"]:
        f = report["facts"][s]
        parts.append(
            f"accuracy {f['accuracy']['est']:.3f}; over-confidence pred {f['pred']['over_all_pp']['est']:+.0f} "
            f"stated {f['stated']['over_all_pp']['est']:+.0f} pts; on {f['n_lost']} lost: pred "
            f"{f['pred']['on_lost_pp']['est']:.0f} stated {f['stated']['on_lost_pp']['est']:.0f}; AUROC pred "
            f"{f['pred']['auroc']['est']:.2f} stated {f['stated']['auroc']['est']:.2f}; Yes-lean {f['yes_lean']['est']:+.2f}"
        )
    if s in report["marbles"]:
        d = report["marbles"][s]["draws"]
        parts.append(
            f"draws: KL {d['kl']['est']:.3f} overbet {d['overbet']['est']:+.2f} dH worst "
            f"{d['d_entropy_worst']['est']:+.2f} stated-ideal {d['stated_minus_ideal']['est']:+.2f}"
        )
    if report["inside"].get(s, {}).get("facts"):
        best = max(report["inside"][s]["facts"], key=lambda r: r["probe_refit_auroc"])
        parts.append(
            f"probe AUROC refit {best['probe_refit_auroc']:.2f} same {best['probe_same_auroc']:.2f} (layer {best['layer']})"
        )
    return " | ".join(parts)


# ── figures ────────────────────────────────────────────────────────────────────────────────────────────

HEADLINES = {  # claims checked against confidence-report.json; other models get the neutral wording
    ("qwen3-8b", "ranking"): "Compression weakens Qwen3-8B's yes/no self-assessment. Its answer probabilities "
    "separate correct from incorrect answers as well as before.",
    ("qwen3-8b", "facts"): "Qwen3-8B is overconfident before and after compression, including on the questions "
    "that compression makes it answer incorrectly.",
    ("qwen3-8b", "marbles"): "On a task with exact probabilities, compression does not consistently worsen "
    "Qwen3-8B's predictions. Its stated probabilities shift by up to 30 points.",
    ("qwen3-8b", "inside"): "Qwen3-8B's hidden state encodes whether its answer is correct, at every depth, "
    "before and after compression.",
}
NEUTRAL = {
    "ranking": "{name}: how well three signals separate correct from incorrect answers, per compression method",
    "facts": "{name}: confidence compared with accuracy, per compression method",
    "marbles": "{name}: predicted and stated probabilities compared with exact ones, per compression method",
    "inside": "{name}: whether the hidden state encodes answer correctness, by depth and compression method",
}
NO_SIGNAL = (0.4, 0.6)  # a self-assessment AUROC in this range: the reply does not depend on the answer
ANSWER_PROB = "Answer probability: the probability the model assigned to the answer it generated."
SELF_ASSESS = (
    "Self-assessment: the model is shown the question and its answer and asked 'Is the proposed answer correct?'. "
    "The signal is the probability it gives to 'Yes'."
)
PROBE = (
    "Hidden-state probe: a linear classifier trained on the model's internal activations to predict whether "
    "its answer is correct."
)
AUROC = (
    "AUROC: the probability that a randomly chosen correct answer has a higher signal than a randomly chosen "
    "incorrect one (0.5 = no separation, 1 = perfect)."
)
SETUP = "Setup: each model answers the same 3,000 factual questions (PopQA), and each answer is graded correct or incorrect."
ROWS = "Rows: the uncompressed model, then one row per compression method."
FAMILIES = "Blue: quantization. Orange: WANDA pruning. Grey: magnitude pruning and block removal."


def _colour(setting: str) -> str:
    from .gt.poster import family_colour

    return "black" if setting == "base" else family_colour(setting)


def _dots(ax, report_rows, rows, readers, scale=1.0):
    """One row per setting. readers: (function row -> {est, lo, hi}, marker style) drawn top to bottom
    within the row. Styles: "filled", "hollow", "diamond"."""
    n, k = len(rows), len(readers)
    for i, s in enumerate(rows):
        colour = _colour(s)
        for r, (reader, style) in enumerate(readers):
            e = reader(report_rows[s])
            if e is None or np.isnan(e["est"]):
                continue
            y = n - 1 - i + 0.14 * (k - 1) - 0.28 * r
            ax.plot(
                [scale * e["lo"], scale * e["hi"]],
                [y, y],
                color=colour,
                lw=2.5,
                alpha=0.45 if style == "hollow" else 1,
                zorder=2,
            )
            ax.plot(
                scale * e["est"],
                y,
                "D" if style == "diamond" else "o",
                ms=10 if style == "diamond" else 12,
                mew=2.5,
                color="white" if style == "hollow" else colour,
                mec=colour,
                zorder=3,
            )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, n - 0.3)


def _row_labels(ax, rows):
    side = ax.get_yaxis_transform()
    for i, s in enumerate(rows):
        ax.text(
            -0.04, len(rows) - 1 - i, LABELS[s], ha="right", va="center", fontsize=17, color=_colour(s), transform=side
        )


def _key(fig, y, entries) -> float:
    """Marker-and-definition keys under the reading text, each wrapped to the figure's width. Returns the
    height used, as a fraction of the figure. entries: (marker, text)."""
    import textwrap

    h, w = fig.get_figheight(), fig.get_figwidth()
    width = int((w - 0.9) * 72 / (0.47 * 17))
    used = 0.0
    for marker, text in entries:
        lines = textwrap.wrap(text, width)
        fig.text(0.03, y - used, marker, fontsize=19, va="top", color="0.2")
        fig.text(0.03 + 0.28 / w, y - used, "\n".join(lines), fontsize=17, va="top", color="0.2", linespacing=1.3)
        used += (0.33 * len(lines) + 0.1) / h
    return used


def _control_note(model: str) -> str:
    """What scripts/probe_controls.py found for this model, if it was run."""
    path = RUNS / model / "probe-controls.json"
    if not path.exists():
        return ""
    settings = json.loads(path.read_text())["settings"].values()
    gain = max(max(r["added_minus_known"]["est"] for r in s["layers"]) for s in settings)
    return (
        " Control: a classifier given only the question type, the subject's popularity and the answer probability "
        f"reaches nearly the same AUROC; adding the hidden state raises it by at most {max(gain, 0):.2f}."
    )


def _best_probe(layers: list[dict]) -> dict | None:
    if not layers:
        return None
    v = max(r["probe_refit_auroc"] for r in layers)
    return {"est": v, "lo": v, "hi": v}


def figures(report: dict, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .gt.poster import SHADE, frame, save, theme

    theme()
    model = report["model"]
    name = "Qwen3-" + model.split("-")[1].upper()
    title = lambda kind: HEADLINES.get((model, kind), NEUTRAL[kind].format(name=name))
    F, M, inside = report["facts"], report["marbles"], report["inside"]
    rows = [s for s in ["base", *SETTINGS] if s in F and F[s]["accuracy"]["est"] >= 0.01]
    dropped = [LABELS[s] for s in F if s not in rows]
    dropped_note = f" Not shown because accuracy is below 1%: {', '.join(dropped)}." if dropped else ""
    h_rows = 0.62 * len(rows)

    def panel_top(fig, rect, entries, gap):
        """Draw the key under the reading text; return the top of the plotting area."""
        y = rect[3] - 0.05 / fig.get_figheight()
        return y - _key(fig, y, entries) - gap / fig.get_figheight()

    if rows:  # 1. ranking: which signal still separates correct from incorrect answers
        fig = plt.figure(figsize=(13, 10.6 + h_rows))
        b, w = F["base"], min((F[s]["stated"]["auroc"]["est"], s) for s in rows)
        rect = frame(
            fig,
            title("ranking"),
            [
                SETUP
                + " We measure how well each of three signals separates the correct answers from the incorrect ones.",
                AUROC + " " + ROWS,
                f"Result: uncompressed, answer probability scores {b['pred']['auroc']['est']:.2f} and self-assessment "
                f"{b['stated']['auroc']['est']:.2f}. After compression, answer probability stays between "
                f"{min(F[s]['pred']['auroc']['est'] for s in rows):.2f} and {max(F[s]['pred']['auroc']['est'] for s in rows):.2f}; "
                f"self-assessment falls as low as {w[0]:.2f} ({LABELS[w[1]]}).",
            ],
            "Bars: 95% bootstrap intervals over questions; the probe has none (a single fit). Vertical grey lines: "
            "the uncompressed model's answer probability (dotted) and self-assessment (dashed). The probe is trained "
            f"separately on each model, 5-fold cross-fitted.{_control_note(model)} {FAMILIES}{dropped_note}",
        )
        h = fig.get_figheight()
        top = panel_top(fig, rect, [("○", ANSWER_PROB), ("●", SELF_ASSESS), ("♦", PROBE)], 0.25)
        ax = fig.add_axes([0.27, rect[1] + 0.95 / h, 0.68, top - rect[1] - 0.95 / h])
        ax.axvline(0.5, color="0.3", lw=1.2)
        ax.axvspan(0, 0.5, color=SHADE, zorder=0, lw=0)
        for key in ("pred", "stated"):
            ax.axvline(b[key]["auroc"]["est"], color="0.75", lw=1.2, ls=":" if key == "pred" else "--", zorder=1)
        _dots(
            ax,
            {s: {**F[s], "probe": _best_probe(inside.get(s, {}).get("facts", []))} for s in rows},
            rows,
            [
                (lambda r: r["pred"]["auroc"], "hollow"),
                (lambda r: r["stated"]["auroc"], "filled"),
                (lambda r: r["probe"], "diamond"),
            ],
        )
        lowest = min(F[s][k]["auroc"]["lo"] for s in rows for k in ("pred", "stated"))
        ax.set_xlim(min(0.44, lowest - 0.03), 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5\nno separation", "0.6", "0.7", "0.8", "0.9", "1\nperfect"])
        ax.set_xlabel("AUROC: separation of correct from incorrect answers")
        _row_labels(ax, rows)
        save(fig, out / "confidence-ranking.png")

    if rows:  # 2. facts: confidence against accuracy
        fig = plt.figure(figsize=(13, 11.2 + h_rows))
        silent = [LABELS[s] for s in rows if NO_SIGNAL[0] <= F[s]["stated"]["auroc"]["est"] <= NO_SIGNAL[1]]
        silent_note = (
            " In these rows the self-assessment does not depend on the answer (AUROC 0.4 to 0.6), so its mean "
            f"near 50% carries no information: {', '.join(silent)}."
            if silent
            else ""
        )
        comp = [s for s in rows if s != "base"]
        n_newly = [F[s]["n_lost"] for s in comp]
        rect = frame(
            fig,
            title("facts"),
            [
                SETUP + " Confidence is measured in two ways, defined below.",
                "Left panel: mean confidence minus accuracy, in percentage points. At 0 the model's average "
                "confidence equals the fraction of its answers that are correct; above 0 it is overconfident.",
                "Right panel: mean confidence on the newly incorrect questions, those the uncompressed model "
                "answered correctly and the compressed model answers incorrectly. A model that registered these "
                "errors would show confidence near 0 there.",
                f"Result: the uncompressed model is overconfident by {F['base']['pred']['over_all_pp']['est']:.0f} points "
                f"(answer probability) and {F['base']['stated']['over_all_pp']['est']:.0f} points (self-assessment). On newly "
                f"incorrect questions, compressed models still report "
                f"{min(F[s]['pred']['on_lost_pp']['est'] for s in comp):.0f}–{max(F[s]['pred']['on_lost_pp']['est'] for s in comp):.0f}% "
                f"and {min(F[s]['stated']['on_lost_pp']['est'] for s in comp):.0f}–{max(F[s]['stated']['on_lost_pp']['est'] for s in comp):.0f}%.",
            ],
            "Self-assessment is the mean of P(Yes) to 'Is the proposed answer correct?' and P(No) to 'Is the proposed "
            "answer incorrect?', which cancels a fixed lean towards Yes or No. Bars: 95% bootstrap intervals over "
            f"questions. Newly incorrect questions per compression method: {min(n_newly)} to {max(n_newly)} of 3,000; "
            f"the uncompressed model has none by definition.{silent_note} {FAMILIES}{dropped_note}",
        )
        h = fig.get_figheight()
        top = panel_top(fig, rect, [("○", ANSWER_PROB), ("●", SELF_ASSESS)], 1.05)
        grid = fig.add_gridspec(1, 2, left=0.27, right=0.96, top=top, bottom=rect[1] + 0.95 / h, wspace=0.12)
        axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
        _dots(
            axes[0],
            F,
            rows,
            [(lambda r: r["pred"]["over_all_pp"], "hollow"), (lambda r: r["stated"]["over_all_pp"], "filled")],
        )
        _dots(
            axes[1],
            F,
            rows,
            [
                (lambda r: None if r["n_lost"] == 0 else r["pred"]["on_lost_pp"], "hollow"),
                (lambda r: None if r["n_lost"] == 0 else r["stated"]["on_lost_pp"], "filled"),
            ],
        )
        axes[0].axvline(0, color="0.3", lw=1.2)
        axes[0].axvspan(0, 200, color=SHADE, zorder=0, lw=0)
        axes[0].set_xlim(-15, 75)
        axes[0].set_xticks([0, 25, 50, 75], ["0", "+25", "+50", "+75"])
        axes[1].set_xlim(-5, 100)
        axes[1].set_xticks([0, 25, 50, 75, 100])
        axes[0].set_title("All 3,000 questions", fontsize=19, pad=12)
        axes[1].set_title(
            "Newly incorrect questions:\ncorrect before compression, incorrect after", fontsize=19, pad=12
        )
        axes[0].set_xlabel("mean confidence − accuracy (points)")
        axes[1].set_xlabel("mean confidence (%)")
        _row_labels(axes[0], rows)
        save(fig, out / "confidence-facts.png")

    m_rows = [s for s in ["base", *SETTINGS] if s in M and (s not in F or s in rows)]
    if m_rows:  # 3. marbles
        fig = plt.figure(figsize=(13, 11.4 + 0.62 * len(m_rows)))
        mb = M["base"]
        comp = [s for s in m_rows if s != "base"]
        st = [100 * M[s]["draws"]["stated_minus_ideal"]["est"] for s in comp]
        kl = [M[s]["draws"]["kl"]["est"] for s in comp]
        rect = frame(
            fig,
            title("marbles"),
            [
                "Setup: 900 short texts describe marbles drawn from a bag. For each text, the probability that each "
                "colour is drawn next is known exactly. There are two kinds of text, defined below.",
                "Left panel: error of the model's next-word probabilities over the colour words, as KL divergence "
                "from the exact distribution in nats (0 = exact).",
                "Right panel: the model is asked 'How likely is it that the next marble is <colour>?' and replies "
                "with one digit. Shown is its stated probability minus the exact probability, in percentage points.",
                f"Result, counting texts: prediction error is {mb['draws']['kl']['est']:.2f} uncompressed and "
                f"{min(kl):.2f}–{max(kl):.2f} after compression, with no consistent direction. The stated probability "
                f"is {100 * mb['draws']['stated_minus_ideal']['est']:+.0f} points off uncompressed and "
                f"{min(st):+.0f} to {max(st):+.0f} points off after compression.",
            ],
            "300 texts state the contents; 600 require counting. Stated probability: the expected value of the "
            "one-digit reply (digit d means d0% to (d+1)0%), compared with the digit bin that contains the exact "
            "probability. Bars: 95% bootstrap intervals over texts. Vertical grey lines: the uncompressed model "
            f"(dashed: contents stated; dotted: counting required). {FAMILIES}",
        )
        h = fig.get_figheight()
        top = panel_top(
            fig,
            rect,
            [
                (
                    "●",
                    "Contents stated: the text gives the number of marbles of each colour. Nothing has to be inferred.",
                ),
                (
                    "○",
                    "Counting required: the text gives the rule of the urn and lists 4 to 32 past draws. The exact "
                    "probability depends on how often each colour was drawn, so the model has to count.",
                ),
            ],
            0.7,
        )
        grid = fig.add_gridspec(1, 2, left=0.27, right=0.96, top=top, bottom=rect[1] + 0.95 / h, wspace=0.12)
        axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
        _dots(axes[0], M, m_rows, [(lambda r: r["bag"]["kl"], "filled"), (lambda r: r["draws"]["kl"], "hollow")])
        _dots(
            axes[1],
            M,
            m_rows,
            [
                (lambda r: r["bag"]["stated_minus_ideal"], "filled"),
                (lambda r: r["draws"]["stated_minus_ideal"], "hollow"),
            ],
            scale=100,
        )
        for ax, key in zip(axes, ("kl", "stated_minus_ideal")):
            for kind, ls in (("bag", "--"), ("draws", ":")):
                ax.axvline((100 if key != "kl" else 1) * mb[kind][key]["est"], color="0.75", lw=1.2, ls=ls, zorder=1)
        axes[1].axvline(0, color="0.3", lw=1.2)
        axes[0].set_xlim(0, None)
        lo, hi = axes[1].get_xlim()
        axes[1].set_xlim(min(lo, -15), max(hi, 30) + 4)
        axes[0].set_title("Prediction error", fontsize=19, pad=12)
        axes[1].set_title("Stated-probability error", fontsize=19, pad=12)
        axes[0].set_xlabel("KL divergence from exact (nats)")
        axes[1].set_xlabel("stated − exact probability (points)")
        _row_labels(axes[0], m_rows)
        save(fig, out / "confidence-marbles.png")

    order = ("base", "rtn-3", "gptq-3", "awq-3", "wanda-50", "wanda-2:4", "shortgpt-10")
    show = [s for s in order if inside.get(s, {}).get("facts") and s in rows][:6]
    if show:  # 4. inside
        fig, axes = plt.subplots(1, len(show), figsize=(22, 12.2), sharey=True, squeeze=False)
        refit = [max(r["probe_refit_auroc"] for r in inside[s]["facts"]) for s in show]
        same = [max(r["probe_same_auroc"] for r in inside[s]["facts"]) for s in show[1:]]
        for ax, s in zip(axes[0], show):
            L, colour = inside[s]["facts"], _colour(s)
            x = [r["layer"] for r in L]
            ax.axhspan(0.1, 0.5, color=SHADE, zorder=0, lw=0)
            ax.plot(x, [r["lens_auroc"] for r in L], "o-", color="0.6", lw=2.5, ms=8, mec="white", zorder=2)
            if s != "base":
                ax.plot(
                    x,
                    [r["probe_same_auroc"] for r in L],
                    "o--",
                    color=colour,
                    lw=2,
                    ms=7,
                    mec="white",
                    alpha=0.6,
                    zorder=3,
                )
            ax.plot(x, [r["probe_refit_auroc"] for r in L], "o-", color=colour, lw=3.5, ms=9, mec="white", zorder=4)
            ax.set_title(LABELS[s], fontsize=20, color=colour, pad=12)
            ax.set_xlabel("depth (block)")
            ax.set_ylim(0.2, 1.0)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        axes[0][0].set_ylabel("AUROC: separation of correct from incorrect answers\n(0.5 = none, 1 = perfect)")
        rect = frame(
            fig,
            title("inside"),
            [
                "Setup: the model is shown a question and its own answer and asked 'Is the proposed answer correct?'. "
                "At the last token of that prompt we record the hidden state after every fourth block. Each panel is one "
                "model; depth runs left to right.",
                "Thick line: a linear probe trained on that model's hidden states to predict whether the answer is "
                "correct. Dashed line: the probe trained on the uncompressed model, applied unchanged to the compressed "
                "model. Grey line: the model's own Yes-versus-No output, computed from the hidden state at that depth.",
                f"Result: the thick line reaches {min(refit):.2f}–{max(refit):.2f} in every panel, so the hidden state "
                "still encodes correctness after compression. The dashed line reaches only "
                f"{min(same):.2f}–{max(same):.2f}, so compression changes the directions that encode it."
                if same
                else "",
            ],
            "PopQA, 3,000 questions. Probe: ridge regression on the first 256 principal components, 5-fold "
            "cross-fitted; no intervals (a single fit per block). " + AUROC + " The grey line uses the logit lens "
            f"(final normalization and output layer applied to an intermediate state), which is unreliable in early blocks.{_control_note(model)}",
        )
        save(fig, out / "confidence-inside.png", rect)
