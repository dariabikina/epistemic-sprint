# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Measures, intervals and figures for the token-probability experiment (P1 in PLAN.md). CPU only.

    wu probs-report --model qwen3-8b [--tag v1]

Reads what `wu probs` wrote for every setting and writes <RUNS>/<model>/probs-<tag>-report.json and
probs-<tag>-*.png.

The unit is one fact asked one way in one language: (fact, language, wording), with example set A.
Languages are never pooled, and candidates are compared by rank within a prompt, never by raw
log-probability across languages.

K, the content
    margin        log p of the best correct form minus log p of the best wrong candidate (roles `hard`
                  and `popmatched`). For occupation, other true answers count as correct; for other
                  relations they are left out of both sides.
    first         margin > 0: the gold answer outranks every wrong candidate.
    beaten        the share of wrong candidates the gold answer outranks.
    first_null    the same after subtracting each form's log p under the null prompt (an invented
                  subject), on the `question` wording only: corrects for answers that are likely for
                  any subject.
    short_correct the greedy short answer contains a gold answer (wu.match).
M, the report
    knows         log p(Yes) - log p(No) to "Do you know the answer to this question?", without the
                  end-of-turn token (the model rarely ends its turn there); p_yes is its logistic.
    real          AUROC with which `knows` separates real facts from invented subjects.
Coupling
    AUROC of `knows` for `first` and for `short_correct`, beside the AUROC of `margin` for
    `short_correct` (the counterpart of prediction confidence in L1).
    Lost facts: gold first in the base model and not in this setting. Kept: first in both.
    tracking      mean change in p_yes on lost facts minus that on kept facts (negative: the report
                  falls where the content fell).
    unaware       the share of lost facts with p_yes >= 0.5. Depends on the report's overall lean
                  towards Yes or No, which compression moves by several nats; read it with `lean`.
    Measures that a constant shift of `knows` does not change:
    kept_vs_lost  AUROC with which `knows` separates the facts this setting kept from those it lost;
                  d_kept_vs_lost is that minus the same AUROC for the base model's `knows` on the same
                  two groups (positive: the report moved towards the facts that survived).
    knows_order   Spearman correlation over facts between this setting's `knows` and the base model's.
    lean, spread  mean and standard deviation of `knows` over real facts; near_half is the share of
                  facts with p_yes between 0.4 and 0.6.
    displaced     the share of lost facts whose greedy answer has probability >= 0.5: a wrong answer
                  took over. The rest are diffuse.
Probe
    Ridge on 256 principal components of the residual stream at the last `knows` prompt token,
    cross-fitted in 5 folds by fact, refitted on each setting, predicting `first`. The layer is the one
    that is best in the base model.

Intervals: 95% percentile, 1,000 bootstrap draws of facts (a fact's wordings stay together), seed 0;
differences from the base model are paired on the same draws. Probes are not refitted inside the
bootstrap: their intervals are for the fitted scores.
"""

import json
import warnings

import numpy as np
from scipy.stats import spearmanr

from . import probs_poster
from .confidence_report import LABELS, SETTINGS, pca_scores, probe_scores
from .gt.stats import auc
from .match import match
from .models import RUNS, setting_dir
from .probs import PROMPTS

B = 1000
DATA = PROMPTS.parent
WRONG = ("hard", "popmatched")
BANDS = ("head", "middle", "tail")
LANGS = ("en", "ru")
YES_NO = {"Yes": True, "No": False, "Да": True, "Нет": False}


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


# ── loading ────────────────────────────────────────────────────────────────────────────────────────────


def metadata() -> dict:
    """Per fact or control id: relation, band, the gold answers in each language, PopQA ids."""
    en = {r["id"]: r for r in read_jsonl(DATA / "facts_en.jsonl")}
    ru = {r["id"]: r for r in read_jsonl(DATA / "facts_ru.jsonl")}
    out = {}
    for i, r in en.items():
        out[i] = {
            "relation": r["relation"],
            "band": r["band"],
            "gold": {"en": r["gold"]["en"], "ru": ru[i]["gold"]["ru"]},
            "popqa_ids": r["popqa_ids"],
        }
    return out


def _best(rows: list[dict], relation: str, null: dict | None = None) -> tuple[float, dict]:
    """From one prompt's score rows: the best correct form's log p, and each wrong candidate's best.
    `null`: {(role, qid, form): log p under the null prompt}, subtracted first."""
    correct = ("gold", "true_alt") if relation == "occupation" else ("gold",)
    gold, wrong = -np.inf, {}
    for r in rows:
        v = r["logprob"] - (null[(r["role"], r["qid"], r["form"])] if null is not None else 0.0)
        if r["role"] in correct:
            gold = max(gold, v)
        elif r["role"] in WRONG:
            wrong[r["qid"]] = max(wrong.get(r["qid"], -np.inf), v)
    return gold, wrong


def _log_odds(rows: list[dict], end: bool) -> float:
    lp = {YES_NO[r["label"]]: (r["logprob"] if end else sum(r["token_logprobs"][:-1])) for r in rows}
    return lp[True] - lp[False]


def load_setting(model: str, setting: str, tag: str, meta: dict, units: list | None = None) -> dict | None:
    """Arrays over units (fact, language, wording) and over control units, for one setting. `units`:
    the base model's, so that every setting is aligned to it; None builds them."""
    d = setting_dir(model, setting) / f"probs-{tag}"
    names = ("short.jsonl", "scores-scoring.jsonl", "scores-knows.jsonl", "position_stats-scoring.jsonl")
    if not all((d / n).exists() for n in names):
        return None
    by_prompt, knows = {}, {}
    for r in read_jsonl(d / "scores-scoring.jsonl"):
        by_prompt.setdefault((r["id"], r["lang"], r["variant"], r["exemplar_set"], r["is_null"]), []).append(r)
    for r in read_jsonl(d / "scores-knows.jsonl"):
        knows.setdefault((r["id"], r["lang"], r["variant"]), []).append(r)
    stats = {
        (r["id"], r["lang"], r["variant"], r["exemplar_set"], r["is_null"]): r
        for r in read_jsonl(d / "position_stats-scoring.jsonl")
    }
    short = {(r["id"], r["lang"], r["variant"]): r for r in read_jsonl(d / "short.jsonl")}
    if units is None:
        units = sorted(k[:3] for k in by_prompt if k[0] in meta and k[3] == "A" and not k[4])
    controls = sorted(k for k in knows if k[0] not in meta)
    n = len(units)
    out = {"units": units, "controls": controls, "dir": d}
    for name in ("margin", "margin_b", "margin_null", "beaten", "entropy", "kl", "knows", "knows_end", "mass"):
        out[name] = np.full(n, np.nan)
    for name in ("short_correct", "short_finished", "short_p"):
        out[name] = np.zeros(n)
    out["short_text"] = [short[u]["text"].strip() for u in units]
    for j, u in enumerate(units):
        relation = meta[u[0]]["relation"]
        for name, ex in (("margin", "A"), ("margin_b", "B")):
            gold, wrong = _best(by_prompt[(*u, ex, False)], relation)
            out[name][j] = gold - max(wrong.values())
            if ex == "A":
                out["beaten"][j] = np.mean([v < gold for v in wrong.values()])
        if (*u, "A", True) in by_prompt:
            null = {(r["role"], r["qid"], r["form"]): r["logprob"] for r in by_prompt[(*u, "A", True)]}
            gold, wrong = _best(by_prompt[(*u, "A", False)], relation, null)
            out["margin_null"][j] = gold - max(wrong.values())
        st = stats[(*u, "A", False)]
        out["entropy"][j], out["kl"][j] = st["entropy"], st.get("kl_from_base", 0.0)
        out["knows"][j], out["knows_end"][j] = _log_odds(knows[u], False), _log_odds(knows[u], True)
        out["mass"][j] = sum(np.exp(sum(r["token_logprobs"][:-1])) for r in knows[u])
        s = short[u]
        out["short_correct"][j] = match(s["text"], meta[u[0]]["gold"][u[1]])
        out["short_finished"][j] = s["finish_reason"] == "stop"
        out["short_p"][j] = np.exp(sum(s["token_logprobs"]))
    out["first"] = (out["margin"] > 0).astype(float)
    out["first_b"] = (out["margin_b"] > 0).astype(float)
    out["first_null"] = np.where(np.isnan(out["margin_null"]), np.nan, out["margin_null"] > 0)
    out["p_yes"] = 1 / (1 + np.exp(-out["knows"]))
    out["c_knows"] = np.array([_log_odds(knows[c], False) for c in controls])
    out["c_p_yes"] = 1 / (1 + np.exp(-out["c_knows"]))
    out["c_short"] = [short[c]["text"].strip() for c in controls]
    return out


# ── intervals ──────────────────────────────────────────────────────────────────────────────────────────


def cluster_draws(clusters: np.ndarray, seed: int = 0) -> list[np.ndarray]:
    """B resamples of the clusters (facts) with replacement, each as the row indices it selects."""
    ids = np.unique(clusters)
    rows = {c: np.flatnonzero(clusters == c) for c in ids}
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows[c] for c in rng.choice(ids, len(ids))]) for _ in range(B)]


def interval(fn, n: int, draws: list[np.ndarray]) -> dict:
    """fn(row indices) -> number: the estimate on all rows and a percentile interval over the draws."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        est = float(fn(np.arange(n)))
        got = np.array([fn(j) for j in draws], float)
        if np.isnan(got).all():
            return {"est": est, "lo": float("nan"), "hi": float("nan")}
        lo, hi = np.nanquantile(got, [0.025, 0.975])
        return {"est": est, "lo": float(lo), "hi": float(hi)}


def _mean(x, j, where=None):
    v = x[j] if where is None else x[j][where[j]]
    return np.nanmean(v) if len(v) else np.nan


def _auc(score, label, j, where=None):
    s, y = score[j], label[j]
    if where is not None:
        s, y = s[where[j]], y[where[j]]
    return auc(s, y)


# ── measures ───────────────────────────────────────────────────────────────────────────────────────────


def _wordings_r(c: dict, lang: str) -> float:
    """Correlation of `knows` between the two wordings of the same fact."""
    at = {u: j for j, u in enumerate(c["units"])}
    pairs = [(j, at[(u[0], lang, "paraphrase")]) for u, j in at.items() if u[1] == lang and u[2] == "question"]
    a, b = np.array(pairs).T
    return float(np.corrcoef(c["knows"][a], c["knows"][b])[0, 1])


def measures(c: dict, b: dict, lang: str, band: dict, draws, c_draws) -> dict:
    """One setting `c` against the base model `b`, one language."""
    units, n = c["units"], len(c["units"])
    L = np.array([u[1] == lang for u in units])
    lost, kept = L & (b["first"] == 1) & (c["first"] == 0), L & (b["first"] == 1) & (c["first"] == 1)
    wrong_now = L & (c["short_correct"] == 0)
    dp = c["p_yes"] - b["p_yes"]
    iv = lambda fn: interval(fn, n, draws)
    paired = lambda f: iv(lambda j: f(c, j) - f(b, j))
    m = {
        "n": int(L.sum()),
        "n_lost": int(lost.sum()),
        "n_kept": int(kept.sum()),
        # content
        "short_accuracy": iv(lambda j: _mean(c["short_correct"], j, L)),
        "short_unfinished": float(1 - c["short_finished"][L].mean()),
        "first": iv(lambda j: _mean(c["first"], j, L)),
        "d_first": paired(lambda s, j: _mean(s["first"], j, L)),
        "first_null": iv(lambda j: _mean(c["first_null"], j, L)),
        "beaten": iv(lambda j: _mean(c["beaten"], j, L)),
        "margin_median": iv(lambda j: np.nanmedian(c["margin"][j][L[j]])),
        "retention": iv(lambda j: _mean(c["first"], j, L & (b["first"] == 1))),
        "kl_median": iv(lambda j: np.nanmedian(c["kl"][j][L[j]])),
        "flip_sets": iv(lambda j: _mean((c["first"] != c["first_b"]).astype(float), j, L)),
        # the report
        "p_yes": iv(lambda j: _mean(c["p_yes"], j, L)),
        "d_p_yes": paired(lambda s, j: _mean(s["p_yes"], j, L)),
        "yes_no_mass": float(c["mass"][L].mean()),
        "knows_with_end_r": float(np.corrcoef(c["knows"][L], c["knows_end"][L])[0, 1]),
        # coupling
        "knows_for_first": iv(lambda j: _auc(c["knows"], c["first"], j, L)),
        "d_knows_for_first": paired(lambda s, j: _auc(s["knows"], s["first"], j, L)),
        "knows_for_correct": iv(lambda j: _auc(c["knows"], c["short_correct"], j, L)),
        "d_knows_for_correct": paired(lambda s, j: _auc(s["knows"], s["short_correct"], j, L)),
        "margin_for_correct": iv(lambda j: _auc(c["margin"], c["short_correct"], j, L)),
        "d_margin_for_correct": paired(lambda s, j: _auc(s["margin"], s["short_correct"], j, L)),
        "certainty_for_correct": iv(lambda j: _auc(-c["entropy"], c["short_correct"], j, L)),
        # lost facts
        "dp_lost": iv(lambda j: _mean(dp, j, lost)),
        "dp_kept": iv(lambda j: _mean(dp, j, kept)),
        "tracking": iv(lambda j: _mean(dp, j, lost) - _mean(dp, j, kept)),
        "kept_vs_lost": iv(lambda j: _auc(c["knows"], c["first"], j, lost | kept)),
        "kept_vs_lost_base": iv(lambda j: _auc(b["knows"], c["first"], j, lost | kept)),
        "d_kept_vs_lost": paired(lambda s, j: _auc(s["knows"], c["first"], j, lost | kept)),
        "knows_order": iv(lambda j: spearmanr(c["knows"][j][L[j]], b["knows"][j][L[j]]).statistic),
        "lean": iv(lambda j: _mean(c["knows"], j, L)),
        "d_lean": paired(lambda s, j: _mean(s["knows"], j, L)),
        "spread": iv(lambda j: np.nanstd(c["knows"][j][L[j]])),
        "near_half": iv(lambda j: _mean(((c["p_yes"] > 0.4) & (c["p_yes"] < 0.6)).astype(float), j, L)),
        "wordings_r": _wordings_r(c, lang),
        "unaware": iv(lambda j: _mean((c["p_yes"] >= 0.5).astype(float), j, lost)),
        "p_yes_near_base_on_lost": iv(lambda j: _mean((np.abs(dp) < 0.1).astype(float), j, lost)),
        "base_p_yes_on_lost": iv(lambda j: _mean(b["p_yes"], j, lost)),
        "p_yes_lost": iv(lambda j: _mean(c["p_yes"], j, lost)),
        "p_yes_kept": iv(lambda j: _mean(c["p_yes"], j, kept)),
        "displaced": iv(lambda j: _mean((c["short_p"] >= 0.5).astype(float), j, lost & wrong_now)),
        "d_entropy_lost": iv(lambda j: _mean(c["entropy"] - b["entropy"], j, lost)),
        "p_yes_when_wrong": iv(lambda j: _mean(c["p_yes"], j, wrong_now)),
    }
    # real facts against invented subjects: two independent resamples
    CL = np.array([u[1] == lang for u in c["controls"]])
    real = lambda j, k: auc(
        np.r_[c["knows"][j][L[j]], c["c_knows"][k][CL[k]]], np.r_[np.ones(L[j].sum()), np.zeros(CL[k].sum())]
    )
    got = np.array([real(j, k) for j, k in zip(draws, c_draws)])
    m["real"] = {
        "est": real(np.arange(n), np.arange(len(CL))),
        "lo": float(np.quantile(got, 0.025)),
        "hi": float(np.quantile(got, 0.975)),
    }
    m["p_yes_controls"] = interval(lambda k: _mean(c["c_p_yes"], k, CL), len(CL), c_draws)
    m["by_band"] = {}
    for name in BANDS:
        W = L & np.array([band[u[0]] == name for u in units])
        m["by_band"][name] = {
            "n": int(W.sum()),
            "first": iv(lambda j: _mean(c["first"], j, W)),
            "short_accuracy": iv(lambda j: _mean(c["short_correct"], j, W)),
            "p_yes": iv(lambda j: _mean(c["p_yes"], j, W)),
            "retention": iv(lambda j: _mean(c["first"], j, W & (b["first"] == 1))),
        }
    return m


def probe(c: dict, lang: str, fact_index: np.ndarray, draws, layer: int | None = None) -> dict | None:
    """AUROC of cross-fitted probes on the `knows` prompt's last-token state, one for `first` and one for
    `short_correct`. Without `layer`, every recorded layer is tried and the one best for `first` is
    returned (used for the base model only; the other settings use the base model's layer)."""
    path = c["dir"] / "knows-states.npz"
    if not path.exists():
        return None
    z = np.load(path)
    where = {(i, l, v): k for k, (i, l, v) in enumerate(zip(z["ids"], z["lang"], z["variant"]))}
    L = np.flatnonzero([u[1] == lang for u in c["units"]])
    rows = np.array([where[c["units"][j]] for j in L])
    groups = fact_index[L]
    layers = z["layers"].tolist()
    best = None
    for k in [layers.index(layer)] if layer is not None else range(len(layers)):
        Z = pca_scores(z["H"][rows, k])[0]
        scores = {t: probe_scores(Z, c[t][L], Z, groups) for t in ("first", "short_correct")}
        a = auc(scores["first"], c["first"][L])
        if best is None or a > best[0]:
            best = (a, layers[k], scores)
    mask = np.zeros(len(c["units"]), bool)
    mask[L] = True
    out = {"layer": best[1]}
    for t, score in best[2].items():
        full = np.full(len(c["units"]), np.nan)
        full[L] = score
        out[t] = interval(lambda j, full=full, t=t: _auc(full, c[t], j, mask), len(full), draws)
    return out


def shared_with_l1(model: str, setting: str, c: dict, meta: dict) -> dict | None:
    """The facts that are also in PopQA-3000 (English, `question` wording): do the two experiments agree?"""
    old_path = setting_dir(model, setting) / "confidence-facts-3000.npz"
    graded = setting_dir(model, setting) / "eval-short-3000.jsonl"
    if not (old_path.exists() and graded.exists()):
        return None
    old = np.load(old_path)
    at = {int(i): k for k, i in enumerate(old["ids"])}
    old_correct = np.array([r["correct"] for r in read_jsonl(graded)], float)
    pairs = [
        (j, at[p])
        for j, u in enumerate(c["units"])
        if u[1] == "en" and u[2] == "question"
        for p in meta[u[0]]["popqa_ids"][:1]
        if p in at
    ]
    new, was = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    stated = (old["own_correct_p_yes"][was] + 1 - old["own_wrong_p_yes"][was]) / 2
    draws = np.random.default_rng(0).integers(0, len(pairs), (B, len(pairs)))
    rho = lambda x, y: lambda j: spearmanr(x[j], y[j]).statistic
    iv = lambda fn: interval(fn, len(pairs), list(draws))
    return {
        "n": len(pairs),
        "short_agreement": iv(lambda j: np.mean(c["short_correct"][new][j] == old_correct[was][j])),
        "short_accuracy_here": float(c["short_correct"][new].mean()),
        "short_accuracy_l1": float(old_correct[was].mean()),
        "margin_vs_logp_gold": iv(rho(c["margin"][new], old["logp_gold"][was])),
        "knows_vs_stated": iv(rho(c["knows"][new], stated)),
        "knows_for_l1_correct": iv(lambda j: auc(c["knows"][new][j], old_correct[was][j])),
        "stated_for_l1_correct": iv(lambda j: auc(stated[j], old_correct[was][j])),
    }


# ── one worked example ─────────────────────────────────────────────────────────────────────────────────

# A question the base model answers correctly, one it answers wrongly, an invented subject, and the
# setting in which the first is shown again. Chosen on qwen3-8b: the first has a margin and a P(Yes)
# near the medians of correctly answered English questions (15.2 and 0.86), the second a P(Yes) near
# the median of wrongly answered ones (0.43).
EXAMPLE = {"correct": "Q468132_author", "wrong": "Q2291923_author", "invented": "ctrl_author_07", "then": "wanda-2:4"}


def example(base: dict, then: dict, meta: dict) -> dict:
    """What the model was asked and what it gave back, for the EXAMPLE questions (English, `question`
    wording, example set A), so that the poster can show the measurement on one case."""
    text = {r["id"]: r["question_en"] for f in ("facts_en.jsonl", "controls_en.jsonl") for r in read_jsonl(DATA / f)}

    def one(c: dict, i: str) -> dict:
        key = (i, "en", "question")
        rows = [
            r
            for r in read_jsonl(c["dir"] / "scores-scoring.jsonl")
            if (r["id"], r["lang"], r["variant"]) == key and r["exemplar_set"] == "A" and not r["is_null"]
        ]
        knows = {
            r["label"]: r["token_logprobs"][0]
            for r in read_jsonl(c["dir"] / "scores-knows.jsonl")
            if (r["id"], r["lang"], r["variant"]) == key
        }
        out = {
            "question": text[i],
            "p_yes_token": float(np.exp(knows["Yes"])),
            "p_no_token": float(np.exp(knows["No"])),
        }
        if i in meta:  # a real fact
            j = c["units"].index(key)
            best = lambda roles: max((r for r in rows if r["role"] in roles), key=lambda r: r["logprob"])
            gold, wrong = best(("gold",)), best(WRONG)
            out |= {
                "wrote": c["short_text"][j],
                "correct": bool(c["short_correct"][j]),
                "n_candidates": len(rows),
                "gold": gold["form"],
                "gold_logp": gold["logprob"],
                "best_wrong": wrong["form"],
                "best_wrong_logp": wrong["logprob"],
                "margin": float(c["margin"][j]),
                "p_yes": float(c["p_yes"][j]),
            }
        else:
            k = c["controls"].index(key)
            out |= {"wrote": c["c_short"][k], "p_yes": float(c["c_p_yes"][k])}
        return out

    L = np.array([u[1] == "en" for u in base["units"]])
    kept = L & (base["first"] == 1) & (then["first"] == 1)
    lost = L & (base["first"] == 1) & (then["first"] == 0)
    return {
        "correct": one(base, EXAMPLE["correct"]),
        "wrong": one(base, EXAMPLE["wrong"]),
        "invented": one(base, EXAMPLE["invented"]),
        "then": LABELS[EXAMPLE["then"]],
        "then_correct": one(then, EXAMPLE["correct"]),
        "p_yes_kept": [float(base["p_yes"][kept].mean()), float(then["p_yes"][kept].mean())],
        "p_yes_lost": [float(base["p_yes"][lost].mean()), float(then["p_yes"][lost].mean())],
    }


# ── the report ─────────────────────────────────────────────────────────────────────────────────────────


def _cell(v: dict, digits: int = 2) -> str:
    return f"{v['est']:.{digits}f} [{v['lo']:.{digits}f}, {v['hi']:.{digits}f}]"


def run(model: str, tag: str = "v1", say=print) -> dict:
    meta = metadata()
    band = {i: m["band"] for i, m in meta.items()}
    base = load_setting(model, "base", tag, meta)
    assert base, f"no base outputs for {model} probs-{tag}: run `wu probs --model {model}` first"
    units = base["units"]
    fact_ids = sorted({u[0] for u in units})
    fact_index = np.array([fact_ids.index(u[0]) for u in units])
    draws = cluster_draws(fact_index)
    c_draws = cluster_draws(np.array([sorted({c[0] for c in base["controls"]}).index(c[0]) for c in base["controls"]]))
    l1_path = RUNS / model / "confidence-report.json"
    l1 = json.loads(l1_path.read_text())["facts"] if l1_path.exists() else {}
    report = {"model": model, "tag": tag, "definitions": __doc__, "n_facts": len(fact_ids), "settings": {}}
    layer = {}
    for setting in ["base", *SETTINGS]:
        c = load_setting(model, setting, tag, meta, units)
        if c is None:
            continue
        assert c["controls"] == base["controls"]
        entry = {"label": LABELS[setting]}
        for lang in LANGS:
            entry[lang] = measures(c, base, lang, band, draws, c_draws)
            p = probe(c, lang, fact_index, draws, layer.get(lang))
            if p:
                layer.setdefault(lang, p["layer"])
                entry[lang]["probe"] = p
        if setting in l1:
            entry["l1"] = {
                "pred_auroc": l1[setting]["pred"]["auroc"],
                "stated_auroc": l1[setting]["stated"]["auroc"],
                "accuracy": l1[setting]["accuracy"],
            }
        shared = shared_with_l1(model, setting, c, meta)
        if shared:
            entry["shared_with_l1"] = shared
        report["settings"][setting] = entry
        if setting == EXAMPLE["then"] and model == "qwen3-8b":
            report["example"] = example(base, c, meta)
        e = entry["en"]
        say(
            f"[probs-report] {model} {setting:<12} en: gold first {_cell(e['first'])}  p_yes {_cell(e['p_yes'])}  "
            f"knows->first {_cell(e['knows_for_first'])}  lost {e['n_lost']}, unaware {_cell(e['unaware'])}"
        )
    b = base
    report["wording_floor"] = {
        lang: {
            "example_sets": report["settings"]["base"][lang]["flip_sets"],
            "wordings": float(
                np.mean(
                    [
                        b["first"][j] != b["first"][k]
                        for j, u in enumerate(units)
                        if u[1] == lang and u[2] == "question"
                        for k in [units.index((u[0], u[1], "paraphrase"))]
                    ]
                )
            ),
        }
        for lang in LANGS
    }
    out = RUNS / model / f"probs-{tag}-report.json"
    out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    say(f"[probs-report] wrote {out}")
    figures(report, RUNS / model)
    probs_poster.figures(report, RUNS / model)
    return report


def redraw(model: str, tag: str = "v1", say=print):
    out = RUNS / model
    report = json.loads((out / f"probs-{tag}-report.json").read_text())
    figures(report, out)
    probs_poster.figures(report, out)
    say(f"[probs-report] redrew the figures in {out}")


# ── figures ────────────────────────────────────────────────────────────────────────────────────────────

LANGUAGE = {"en": "English", "ru": "Russian"}
SETUP = (
    "Setup: {n} facts from PopQA, balanced over popular and obscure subjects, each asked in two wordings in "
    "{language}. Every model answers each question, and separately is asked 'Do you know the answer to this "
    "question?' before answering."
)
MARGIN = (
    "Answer margin: log-probability of the correct answer minus that of the most likely wrong candidate, "
    "from a fixed list of plausible wrong answers."
)
KNOWS = "Says it knows: the log-odds of 'Yes' against 'No' to 'Do you know the answer to this question?'."
PROBE = (
    "Hidden-state probe: a linear classifier on the model's activations at the end of the 'Do you know' "
    "prompt, trained to predict whether its answer is correct."
)
AUROC = (
    "AUROC: the probability that a randomly chosen correct answer has a higher signal than a randomly chosen "
    "incorrect one (0.5 = no separation, 1 = perfect)."
)
ROWS = "Rows: the uncompressed model, then one row per compression method."
FAMILIES = "Blue: quantization. Orange: WANDA pruning. Grey: magnitude pruning and block removal."
MIN_LOST = 20  # units; rows with fewer lost facts are not drawn in the lost-facts figure


def figures(report: dict, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .confidence_report import _dots, _key, _row_labels
    from .gt.poster import SHADE, frame, save, theme

    theme()
    model, tag, S = report["model"], report["tag"], report["settings"]
    name = "Qwen3-" + model.split("-")[1].upper()
    for lang in LANGS:
        R = {s: S[s][lang] for s in S}
        rows = [s for s in S if R[s]["short_accuracy"]["est"] >= 0.01]
        dropped = [LABELS[s] for s in S if s not in rows]
        dropped_note = f" Not shown because accuracy is below 1%: {', '.join(dropped)}." if dropped else ""
        setup = SETUP.format(n=report["n_facts"], language=LANGUAGE[lang])
        b = R["base"]
        comp = [s for s in rows if s != "base"]
        est = lambda key: [R[s][key]["est"] for s in comp] or [float("nan")]

        # 1. coupling: which signal separates correct from incorrect answers
        fig = plt.figure(figsize=(13, 11.4 + 0.62 * len(rows)))
        low = min(((R[s]["knows_for_correct"]["est"], s) for s in comp), default=(float("nan"), "base"))
        claim = min(R[s]["margin_for_correct"]["lo"] for s in rows) > max(
            R[s]["knows_for_correct"]["est"] for s in rows
        )
        rect = frame(
            fig,
            f"{name}, {LANGUAGE[lang]}: after compression the answer margin still separates correct from "
            "incorrect answers, and better than the model's 'I know' in every setting"
            if claim
            else f"{name}, {LANGUAGE[lang]}: how well the answer margin and the model's 'I know' separate correct "
            "from incorrect answers, per compression method",
            [
                setup + " We measure how well each of three signals separates the correct answers from the "
                "incorrect ones.",
                AUROC + " " + ROWS,
                f"Result: uncompressed, the answer margin scores {b['margin_for_correct']['est']:.2f} and 'says it "
                f"knows' {b['knows_for_correct']['est']:.2f}. After compression the margin stays between "
                f"{min(est('margin_for_correct')):.2f} and {max(est('margin_for_correct')):.2f}; 'says it knows' "
                f"ranges from {low[0]:.2f} ({LABELS[low[1]]}) to {max(est('knows_for_correct')):.2f}.",
            ],
            "Bars: 95% bootstrap intervals over facts (a fact's two wordings are resampled together). Vertical grey "
            "lines: the uncompressed model's answer margin (dotted) and 'says it knows' (dashed). The probe is "
            "trained separately on each model, 5-fold cross-fitted by fact; its interval is for the fitted scores. "
            f"Accuracy of the uncompressed model: {100 * b['short_accuracy']['est']:.0f}%. {FAMILIES}{dropped_note}",
        )
        h = fig.get_figheight()
        y = rect[3] - 0.05 / h
        top = y - _key(fig, y, [("○", MARGIN), ("●", KNOWS), ("♦", PROBE)]) - 0.25 / h
        ax = fig.add_axes([0.27, rect[1] + 0.95 / h, 0.68, top - rect[1] - 0.95 / h])
        ax.axvline(0.5, color="0.3", lw=1.2)
        ax.axvspan(0, 0.5, color=SHADE, zorder=0, lw=0)
        ax.axvline(b["margin_for_correct"]["est"], color="0.75", lw=1.2, ls=":", zorder=1)
        ax.axvline(b["knows_for_correct"]["est"], color="0.75", lw=1.2, ls="--", zorder=1)
        _dots(
            ax,
            R,
            rows,
            [
                (lambda r: r["margin_for_correct"], "hollow"),
                (lambda r: r["knows_for_correct"], "filled"),
                (lambda r: r.get("probe", {}).get("short_correct"), "diamond"),
            ],
        )
        lowest = min(
            [R[s][k]["lo"] for s in rows for k in ("margin_for_correct", "knows_for_correct")]
            + [R[s]["probe"]["short_correct"]["lo"] for s in rows if "probe" in R[s]]
        )
        ax.set_xlim(max(0.0, min(0.44, lowest - 0.03)), 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5\nno separation", "0.6", "0.7", "0.8", "0.9", "1\nperfect"])
        ax.set_xlabel("AUROC: separation of correct from incorrect answers")
        _row_labels(ax, rows)
        save(fig, out / f"probs-{tag}-coupling-{lang}.png")

        # 2. lost facts: how many, and what the model says about knowing them
        lost_rows = [s for s in comp if R[s]["n_lost"] >= MIN_LOST]
        if not lost_rows:
            continue
        few = [LABELS[s] for s in comp if s not in lost_rows]
        few_note = f" Not shown because fewer than {MIN_LOST} questions were lost: {', '.join(few)}." if few else ""
        fig = plt.figure(figsize=(15, 10.6 + 0.62 * len(lost_rows)))
        unaware = [R[s]["unaware"]["est"] for s in lost_rows]
        rect = frame(
            fig,
            f"{name}, {LANGUAGE[lang]}: facts lost under compression, and whether the model still says it knows them",
            [
                setup,
                "A fact is known when the correct answer is more probable than every wrong candidate, and lost "
                "when that holds for the uncompressed model and not for the compressed one.",
                "Left panel: the percentage of the uncompressed model's known facts that the compressed model "
                "lost. Right panel: on those lost facts, the mean probability of 'Yes' to 'Do you know the "
                "answer?', before compression and after, and for comparison on the facts it kept. A model that "
                "registered the loss would say Yes less on lost facts than on kept ones, near the level it "
                "gives invented subjects (dashed line).",
                f"Result: compressed models lose {100 * (1 - max(R[s]['retention']['est'] for s in lost_rows)):.0f}–"
                f"{100 * (1 - min(R[s]['retention']['est'] for s in lost_rows)):.0f}% of known facts. On lost facts "
                f"they still say Yes with mean probability {min(R[s]['p_yes_lost']['est'] for s in lost_rows):.2f}–"
                f"{max(R[s]['p_yes_lost']['est'] for s in lost_rows):.2f}; the share of lost facts with P(Yes) of at "
                f"least 0.5 is {100 * min(unaware):.0f}–{100 * max(unaware):.0f}%. On kept facts the mean is "
                f"{min(R[s]['p_yes_kept']['est'] for s in lost_rows):.2f}–"
                f"{max(R[s]['p_yes_kept']['est'] for s in lost_rows):.2f}. Each method moves P(Yes) as a whole, up "
                "or down, on lost and kept facts alike, so these levels say little about awareness; the companion "
                "figure compares lost with kept facts directly.",
            ],
            "Bars: 95% bootstrap intervals over facts. Dashed line: the uncompressed model's mean P(Yes) on 100 "
            f"invented subjects that have no answer. {FAMILIES}{few_note}{dropped_note}",
        )
        h = fig.get_figheight()
        y = rect[3] - 0.05 / h
        top = (
            y
            - _key(
                fig,
                y,
                [
                    ("○", "Before: the uncompressed model's P(Yes) on the facts this setting lost."),
                    ("●", "After: the compressed model's P(Yes) on the same facts."),
                    ("♦", "After, on the facts it kept: the compressed model's P(Yes) on facts it still knows."),
                ],
            )
            - 0.25 / h
        )
        height = top - rect[1] - 0.95 / h
        left = fig.add_axes([0.24, rect[1] + 0.95 / h, 0.3, height])
        right = fig.add_axes([0.6, rect[1] + 0.95 / h, 0.36, height])
        flip = lambda e: {"est": 1 - e["est"], "lo": 1 - e["hi"], "hi": 1 - e["lo"]}
        _dots(left, R, lost_rows, [(lambda r: flip(r["retention"]), "filled")], scale=100)
        left.set_xlim(0, 100)
        left.set_xlabel("known facts lost (%)")
        _row_labels(left, lost_rows)
        _dots(
            right,
            R,
            lost_rows,
            [
                (lambda r: r["base_p_yes_on_lost"], "hollow"),
                (lambda r: r["p_yes_lost"], "filled"),
                (lambda r: r["p_yes_kept"], "diamond"),
            ],
        )
        right.axvline(b["p_yes_controls"]["est"], color="0.5", lw=1.2, ls="--", zorder=1)
        right.set_xlim(0, 1)
        right.set_xlabel("mean P(Yes) to 'Do you know the answer?'")
        save(fig, out / f"probs-{tag}-lost-{lang}.png")

        # 3. does 'says it knows' move towards the facts that survived
        fig = plt.figure(figsize=(13, 10.9 + 0.62 * len(lost_rows)))
        d = {s: R[s]["d_kept_vs_lost"] for s in lost_rows}
        up = [LABELS[s] for s in lost_rows if d[s]["lo"] > 0]
        down = [LABELS[s] for s in lost_rows if d[s]["hi"] < 0]
        moved = "".join(
            f" It is higher after compression in {', '.join(names)}."
            if names and word == "up"
            else f" It is lower after compression in {', '.join(names)}."
            if names
            else ""
            for word, names in (("up", up), ("down", down))
        )
        flat = len(lost_rows) - len(up) - len(down)
        rect = frame(
            fig,
            f"{name}, {LANGUAGE[lang]}: after compression the model's 'I know' separates the facts it kept from "
            "the facts it lost no better than it did before the loss"
            if len(up) <= 1
            else f"{name}, {LANGUAGE[lang]}: does the model's 'I know' follow which facts survive compression",
            [
                setup,
                "A fact is known when the correct answer is more probable than every wrong candidate. For each "
                "compression method, the uncompressed model's known facts are split into those the compressed "
                "model kept and those it lost.",
                "We measure how well 'says it knows' separates kept from lost facts, as an AUROC (0.5 = no "
                "separation, 1 = perfect), using the uncompressed model's reply (before) and the compressed "
                "model's reply (after). Before is above 0.5 because the facts that get lost are those the model "
                "was already less sure of. If the compressed model registered what it lost, after would be "
                "higher than before. A shift of all replies towards Yes or No does not change this measure.",
                f"Result: before, {min(R[s]['kept_vs_lost_base']['est'] for s in lost_rows):.2f}–"
                f"{max(R[s]['kept_vs_lost_base']['est'] for s in lost_rows):.2f}; after, "
                f"{min(R[s]['kept_vs_lost']['est'] for s in lost_rows):.2f}–"
                f"{max(R[s]['kept_vs_lost']['est'] for s in lost_rows):.2f}. The paired difference (after minus "
                f"before) ranges from {min(v['est'] for v in d.values()):+.2f} to "
                f"{max(v['est'] for v in d.values()):+.2f}, and its 95% interval includes zero in {flat} of "
                f"{len(lost_rows)} methods.{moved}",
            ],
            "Bars: 95% bootstrap intervals over facts (a fact's two wordings are resampled together). "
            f"{FAMILIES}{few_note}{dropped_note}",
        )
        h = fig.get_figheight()
        y = rect[3] - 0.05 / h
        top = (
            y
            - _key(
                fig,
                y,
                [
                    ("○", "Before: the uncompressed model's 'says it knows', on the facts this method kept and lost."),
                    ("●", "After: the compressed model's 'says it knows', on the same facts."),
                ],
            )
            - 0.25 / h
        )
        ax = fig.add_axes([0.27, rect[1] + 0.95 / h, 0.68, top - rect[1] - 0.95 / h])
        ax.axvline(0.5, color="0.3", lw=1.2)
        ax.axvspan(0, 0.5, color=SHADE, zorder=0, lw=0)
        _dots(ax, R, lost_rows, [(lambda r: r["kept_vs_lost_base"], "hollow"), (lambda r: r["kept_vs_lost"], "filled")])
        ax.set_xlim(0.38, 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5\nno separation", "0.6", "0.7", "0.8", "0.9", "1\nperfect"])
        ax.set_xlabel("AUROC: 'says it knows' separating kept facts from lost facts")
        _row_labels(ax, lost_rows)
        save(fig, out / f"probs-{tag}-tracking-{lang}.png")
