# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Compress a trained ground-truth model with `wu.compress` (the code used for the LLMs) and evaluate it.

    gt compress RUN [--settings ...]  ->  <GT>/compress/RUN/{SETTING}.npz, report.json

Settings:
    rtn-B, gptq-B, awq-B     weight quantization to B bits, group size 64 (GPTQ: Frantar et al., ICLR 2023;
                             AWQ: Lin et al., MLSys 2024)
    magnitude-P, wanda-P     unstructured pruning of P% of weights; wanda-N:M semi-structured (WANDA: Sun et
                             al., ICLR 2024)
    SETTING+emb              also quantize the tied embedding (otherwise left in bf16, as in Wu et al.)
    SETTING@L                only block L compressed; every other block as trained. GPTQ, AWQ and WANDA
                             calibrate block L on the base model's inputs to it (blockwise.walk, only={L}).
Calibration (GPTQ, AWQ, WANDA): 128 training sequences drawn with seed 42, BOS prepended.

Each setting's per-position evaluation is cached as an .npz. report.json is recomputed from the caches on
every call (measures.summarize).
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import data as D
from . import priors as P
from .measures import RANK_BANDS
from .stats import fmt
from .train import load_model, with_bos

GROUP = 64
SETTINGS = [
    "rtn-8",
    "rtn-4",
    "rtn-3",
    "rtn-2",
    "gptq-4",
    "gptq-3",
    "gptq-2",
    "awq-4",
    "awq-3",
    "rtn-4+emb",
    "rtn-3+emb",
    "magnitude-30",
    "magnitude-50",
    "magnitude-70",
    "wanda-30",
    "wanda-50",
    "wanda-70",
    "wanda-2:4",
]


@contextmanager
def group_size(g):
    from ..compress import quant

    old, quant.GROUP_SIZE = quant.GROUP_SIZE, g
    try:
        yield
    finally:
        quant.GROUP_SIZE = old


def calibration(world: str, n: int = 128, seed: int = 42) -> list[list[int]]:
    xt = np.load(D.data_dir(world) / "train.npy", mmap_mode="r")
    rows = np.sort(np.random.default_rng(seed).choice(len(xt), n, replace=False))
    return with_bos(np.asarray(xt[rows])).tolist()


def compressed(run: Path, setting: str, calib):
    from ..compress import prune, quant

    setting, _, at = setting.partition("@")
    blocks = {int(at)} if at else None
    model = load_model(run)
    if setting == "base":
        return model
    name, emb = setting.removesuffix("+emb"), setting.endswith("+emb")
    method, level = name.split("-", 1)
    if blocks and method in ("rtn", "magnitude"):  # no calibration: compress all, keep block L only
        full = compressed(run, setting, calib)
        model.layers[int(at)] = full.layers[int(at)]
        mx.eval(model.parameters())
        return model
    with group_size(GROUP):
        if method == "rtn":
            quant.rtn(model, int(level))
        elif method == "gptq":
            quant.gptq(model, None, int(level), progress=None, calibration=calib, blocks=blocks)
        elif method == "awq":
            quant.awq(model, None, int(level), progress=None, calibration=calib, blocks=blocks)
        elif method == "magnitude":
            prune.magnitude(model, level)
        elif method == "wanda":
            prune.wanda_recipe(model, None, level, progress=None, calibration=calib, blocks=blocks)
        else:
            raise ValueError(setting)
        if emb:
            nn.quantize(
                model, group_size=GROUP, bits=int(level), class_predicate=lambda p, m: isinstance(m, nn.Embedding)
            )
    mx.eval(model.parameters())
    return model


def bayes_predictive(world: str, x: np.ndarray, chunk: int = 32) -> np.ndarray:
    """(n, T, 256) float32: the Bayes predictive at every position (float32 to halve memory)."""
    w = P.worlds()[world]
    return np.concatenate([w.predict(x[s : s + chunk])[0].astype(np.float32) for s in range(0, len(x), chunk)])


def evaluate(model, x: np.ndarray, bayes: np.ndarray, chunk: int = 32) -> dict:
    """Per position (n, T), over the 256 symbols (the BOS logit is dropped and the rest renormalized):
    logp, entropy and novel_model (mass on symbols not yet seen) of the model; the same for Bayes; and
    kl = KL(Bayes || model). Also q1, the model's predictive at the BOS position."""
    out = {k: [] for k in ("logp", "entropy", "kl", "novel_model", "novel_bayes", "bayes_logp", "bayes_entropy")}
    q1 = None
    for s in range(0, len(x), chunk):
        xs = x[s : s + chunk]
        tok = mx.array(with_bos(xs))
        lg = model(tok[:, :-1]).astype(mx.float32)[..., : P.NBINS]
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        mx.eval(lp)
        lp = np.array(lp).astype(np.float64)
        q = np.exp(lp)
        B = bayes[s : s + chunk].astype(np.float64)
        lB = np.log(np.maximum(B, P.TINY))
        seen = P.seen_before(xs)
        out["logp"].append(P.take(lp, xs))
        out["entropy"].append(-(q * lp).sum(-1))
        out["kl"].append((B * (lB - lp)).sum(-1))
        out["novel_model"].append((q * ~seen).sum(-1))
        out["novel_bayes"].append((B * ~seen).sum(-1))
        out["bayes_logp"].append(P.take(lB, xs))
        out["bayes_entropy"].append(P.entropy(B))
        if q1 is None:
            q1 = q[0, 0]
    res = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    res["q1"] = q1
    return res


def run(run_name: str, settings: list[str], say=print) -> Path:
    from ..models import apply_memory_policy
    from .measures import summarize
    from .train import provenance

    path = D.GT / "runs" / run_name
    world = json.loads((path / "config.json").read_text())["world"]
    _, xv, _, _ = D.load(world)
    out = D.GT / "compress" / run_name
    out.mkdir(parents=True, exist_ok=True)
    calib = calibration(world)
    p0 = P.worlds()[world].p0()
    apply_memory_policy()
    old = json.loads((out / "report.json").read_text())["settings"] if (out / "report.json").exists() else {}
    wanted = [
        "base",
        *[s for s in settings if s != "base"],
        *[k for k in old if k not in settings and k != "base"],
    ]  # keep earlier settings
    results, base, bayes = {}, None, None
    for setting in wanted:
        t0 = time.time()
        f = out / f"{setting.replace(':', 'of')}.npz"
        if f.exists():
            ev = dict(np.load(f))
        else:
            if bayes is None:
                bayes = bayes_predictive(world, xv)
            model = compressed(path, setting, calib)
            ev = evaluate(model, xv, bayes)
            np.savez(f, **ev)
            del model
            mx.clear_cache()
        if setting == "base":
            base = ev
        results[setting] = s = summarize(base, ev, xv, p0)
        say(
            f"[gt compress] {run_name} {setting:>12}: damage +{fmt(s['dkl'], 4)} nats/token; "
            f"tracking AUC {fmt(s['tracking_auc'], 2)}  ({time.time() - t0:.0f}s)"
        )
    (out / "report.json").write_text(
        json.dumps(
            {
                "run": run_name,
                "world": world,
                "group_size": GROUP,
                "calibration": "128 training sequences, seed 42",
                "intervals": "95% cluster bootstrap over validation sequences, paired (stats.py)",
                "provenance": provenance(world),
                "settings": results,
            },
            indent=1,
        )
    )
    say(table(results))
    return out


def table(results: dict) -> str:
    """Markdown: one row per setting; intervals in brackets."""
    cols = [
        ("damage ΔKL", "dkl", 4, True),
        ("Δexcess", "d_excess", 4, True),
        ("Δexcess: new or repeat?", "d_excess_kind", 4, True),
        ("Δexcess: which symbol", "d_excess_which", 4, True),
        ("Δexcess new", "d_excess_new", 3, True),
        ("Δexcess repeat", "d_excess_repeat", 3, True),
        ("entropy gap", "entropy_gap", 3, True),
        ("tracking AUC", "tracking_auc", 2, False),
    ]
    bands = [f"{a + 1}-{b}" for a, b in RANK_BANDS]
    head = (
        "| setting | "
        + " | ".join(c[0] for c in cols)
        + " | prior KL | "
        + " | ".join(f"Δlog p(symbol given new) (+ = higher), rank {b}" for b in bands)
        + " |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for k, s in results.items():
        cells = [fmt(s[key], d, sign) for _, key, d, sign in cols]
        cells.append(f"{s['prior_kl']:.3f}")
        cells += [fmt(s["d_which_new_by_rank"][b], 3, True) for b in bands]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
