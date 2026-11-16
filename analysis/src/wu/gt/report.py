# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""E1: an explicit confidence report, and whether it survives compression (Zipf world).

    gt report-train [--steps 32000] [--seed 0]    ->  <GT>/runs/zipf-report-gt-6m-q-s{seed}-{tag}/
    gt report RUN [--settings ...]                ->  <GT>/report/RUN/{SETTING}.npz, report.json

Sequences. A document x_0..x_255 is fed with a query token `?` inserted after some symbols:
    BOS x_0 ... x_t ? x_{t+1} ...
The model predicts x_{t+1} at the position of x_t, as usual; at the `?` it must answer Y or N. Its reported
confidence is r = P(Y) / (P(Y) + P(N)) at that position.

Target of the report. c = p_Bayes(x_{t+1} = g | x_<=t), where g is the model's own top guess for x_{t+1}
(argmax of its prediction at x_t over the 256 symbols, taken without gradient in the same forward pass): the
true probability that its guess is right. Loss = cross-entropy on symbols + binary cross-entropy of r against c
(soft label). Training: 16 queries per document at random positions; evaluation: queries after symbols
15, 31, ..., 239 (15 per document), the same for every model. Compression is calibrated on 128 training
documents (seed 42) with queries inserted as in training.

After compression, for every query: c (the compressed model's guess, scored by Bayes), r (its report) and
m = max_k p_model(k) (its implicit confidence in the same guess), and the same for the base model. Measures
(stats.py intervals, by sequence):
    report_gap, implicit_gap     mean r - c, mean m - c (calibration in the large; + = overconfident)
    report_brier, implicit_brier mean (r - c)^2, mean (m - c)^2
    follow_slope                 OLS slope of Δr on Δc over queries (Δ = compressed - base): 1 = the report
                                 moves with the guess's correctness, 0 = it does not move; also for m
    worse_dr, worse_dm           mean Δr, Δm on queries where the guess got worse (Δc < -0.1)
    worse_auc                    AUC separating Δc < -0.1 from |Δc| < 0.02 by -Δr (1 = the report drops exactly
                                 where the guess got worse; 0.5 = blind); also for m
"""

import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from scipy.special import gammaln, logsumexp

from . import data as D
from . import priors as P
from .stats import auc_over, interval, mean_over, resample_sequences

QUERY, YES, NO = P.NBINS + 1, P.NBINS + 2, P.NBINS + 3
VOCAB = P.NBINS + 4
N_TRAIN_QUERIES = 16
EVAL_AFTER = np.arange(15, 240, 16)  # symbols after which the evaluation asks


# ── exact target ──────────────────────────────────────────────────────────────────────────────────


def bayes_after(x: np.ndarray, after: np.ndarray) -> np.ndarray:
    """(n, Q, 256): the Bayes predictive of x_{t+1} given x_0..x_t, for t = after[i, q]. Zipf world: a mixture
    over alpha of (counts + alpha G0) / (t + 1 + alpha), weighted by the Dirichlet-multinomial likelihood of the
    prefix (Blackwell & MacQueen 1973; the same predictive as priors.Zipf.predict)."""
    w = P.worlds()["zipf"]
    p0, alphas = w.p0(), np.array(w.params["alphas"], dtype=float)
    n, Q = after.shape
    out = np.empty((n, Q, P.NBINS))
    for i in range(n):
        for q in range(Q):
            t = after[i, q]
            c = np.bincount(x[i, : t + 1].astype(np.int64), minlength=P.NBINS).astype(float)
            ll = np.array(
                [gammaln(a) - gammaln(a + t + 1) + (gammaln(c + a * p0) - gammaln(a * p0)).sum() for a in alphas]
            )
            wts = np.exp(ll - logsumexp(ll))
            out[i, q] = sum(wa * (c + a * p0) / (t + 1 + a) for wa, a in zip(wts, alphas))
    return out


# ── sequences with queries ────────────────────────────────────────────────────────────────────────


def with_queries(x: np.ndarray, after: np.ndarray) -> dict:
    """Inputs with `?` inserted after the symbols in `after` (n, Q), sorted per row. Returns tokens (n, L),
    sym_pos (n, 256): input index whose output predicts x_t (BOS for x_0), query_pos (n, Q), and guess_pos (n, Q):
    input index of x_t, whose output is the guess the query asks about."""
    n, T = x.shape
    Q = after.shape[1]
    L = 1 + T + Q
    tokens = np.empty((n, L), dtype=np.int32)
    sym_pos = np.empty((n, T), dtype=np.int32)
    query_pos = np.empty((n, Q), dtype=np.int32)
    guess_pos = np.empty((n, Q), dtype=np.int32)
    for i in range(n):
        ask = np.zeros(T, bool)
        ask[after[i]] = True
        seq, prev, q = [P.BOS], 0, 0  # prev: input index whose output predicts the next symbol
        for t in range(T):
            sym_pos[i, t] = prev
            seq.append(int(x[i, t]))
            prev = len(seq) - 1  # x_t predicts x_{t+1}, also when a `?` follows it
            if ask[t]:
                guess_pos[i, q] = prev
                seq.append(QUERY)
                query_pos[i, q] = len(seq) - 1
                q += 1
        tokens[i] = seq
    return {"tokens": tokens, "sym_pos": sym_pos, "query_pos": query_pos, "guess_pos": guess_pos}


def random_after(n: int, rng, Q: int = N_TRAIN_QUERIES) -> np.ndarray:
    return np.sort(np.stack([rng.choice(P.T - 1, Q, replace=False) for _ in range(n)]), axis=1)


# ── model ─────────────────────────────────────────────────────────────────────────────────────────


def config(shape: str = "gt-6m-q") -> dict:
    from .train import SHAPES

    return {**SHAPES[shape].config(), "vocab_size": VOCAB, "max_position_embeddings": 1 + P.T + P.T}


def _gather(a, idx):
    """a (n, L, V), idx (n, K) -> (n, K, V)."""
    return mx.take_along_axis(a, idx[..., None], axis=1)


def forward(model, b: dict):
    """Symbol log-probs over the 256 symbols at sym_pos (n, 256, 256), the guess's log-probs at guess_pos
    (n, Q, 256), and the report r at query_pos (n, Q)."""
    lg = model(b["tokens"]).astype(mx.float32)
    sym = _gather(lg, b["sym_pos"])[..., : P.NBINS]
    guess = _gather(lg, b["guess_pos"])[..., : P.NBINS]
    q = _gather(lg, b["query_pos"])
    r = mx.sigmoid(q[..., YES] - q[..., NO])
    norm = lambda z: z - mx.logsumexp(z, axis=-1, keepdims=True)
    return norm(sym), norm(guess), r, lg, q


def loss_fn(model, tokens, sym_pos, query_pos, guess_pos, x, bayes):
    lg = model(tokens).astype(mx.float32)
    sym_logits = _gather(lg, sym_pos)  # over the full vocabulary: other tokens must get no mass at symbols
    sym_loss = nn.losses.cross_entropy(sym_logits, x, reduction="mean")
    guess = mx.stop_gradient(mx.argmax(_gather(lg, guess_pos)[..., : P.NBINS], axis=-1))
    c = mx.take_along_axis(bayes, guess[..., None], axis=-1)[..., 0]
    q = _gather(lg, query_pos)
    z = q[..., YES] - q[..., NO]
    rep_loss = mx.mean(mx.logaddexp(0.0, z) - c * z)  # BCE with soft target c, on the logit z
    return sym_loss + rep_loss, (sym_loss, rep_loss)


# ── training ──────────────────────────────────────────────────────────────────────────────────────


def batch(xs: np.ndarray, rng) -> dict:
    after = random_after(len(xs), rng)
    b = with_queries(xs, after)
    b["x"] = xs.astype(np.int32)
    b["bayes"] = bayes_after(xs, after).astype(np.float32)
    return b


def evaluate(model, x: np.ndarray, bayes_eval: np.ndarray, valid_logp: np.ndarray, chunk: int = 64) -> dict:
    """Per query: c, r, m for this model; and symbol excess over Bayes (positions without a query: the same
    as with, since the model is causal and queries come after the symbol)."""
    after = np.tile(EVAL_AFTER, (len(x), 1))
    out = {"c": [], "r": [], "m": [], "logp": []}
    for s in range(0, len(x), chunk):
        b = with_queries(x[s : s + chunk], after[s : s + chunk])
        sym, guess, r, _, _ = forward(model, {k: mx.array(v) for k, v in b.items()})
        mx.eval(sym, guess, r)
        guess, sym = np.array(guess), np.array(sym)
        g = guess.argmax(-1)
        out["c"].append(np.take_along_axis(bayes_eval[s : s + chunk], g[..., None], -1)[..., 0])
        out["r"].append(np.array(r))
        out["m"].append(np.exp(guess.max(-1)))
        out["logp"].append(np.take_along_axis(sym, x[s : s + chunk, :, None].astype(np.int64), -1)[..., 0])
    res = {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}
    res["excess"] = valid_logp - res.pop("logp")
    return res


def summary(e: dict) -> dict:
    return {
        "excess": float(e["excess"].mean()),
        "accuracy": float(e["c"].mean()),
        "report": float(e["r"].mean()),
        "report_mae": float(np.abs(e["r"] - e["c"]).mean()),
        "corr_r_c": float(np.corrcoef(e["r"].ravel(), e["c"].ravel())[0, 1]),
        "implicit_mae": float(np.abs(e["m"] - e["c"]).mean()),
    }


def train(
    seed: int = 0,
    steps: int = 32000,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup: int = 200,
    eval_every: int = 2000,
    tag: str = "long",
    shape: str = "gt-6m-q",
    say=print,
) -> Path:
    from mlx_lm.models import llama

    from ..models import apply_memory_policy
    from .train import code_hashes, initialize, run_dir

    apply_memory_policy()
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    xt, xv, vb, manifest = D.load("zipf")
    xv = xv[:256]
    bayes_eval = bayes_after(xv, np.tile(EVAL_AFTER, (len(xv), 1)))
    valid_logp = vb["logp"][:256].astype(np.float64)
    out = run_dir("zipf-report", shape, seed, tag)
    out.mkdir(parents=True, exist_ok=True)
    cfg_model = config(shape)
    model = llama.Model(llama.ModelArgs.from_dict(cfg_model))
    initialize(model, cfg_model["num_hidden_layers"])
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    cfg = {
        "world": "zipf",
        "experiment": "E1 report",
        "shape": shape,
        "architecture": cfg_model,
        "parameters": n_params,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "warmup": warmup,
        "queries_per_document": N_TRAIN_QUERIES,
        "data": str(D.data_dir("zipf")),
        "code": code_hashes(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    say(f"[gt report-train] {n_params / 1e6:.2f}M parameters, {steps} steps, seed {seed} -> {out}")

    master = tree_map(lambda p: p.astype(mx.float32), model.parameters())
    model.set_dtype(mx.bfloat16)
    schedule = lambda s: lr * min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.1, betas=(0.9, 0.95))
    opt.init(master)
    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, master, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(rate, tokens, sym_pos, query_pos, guess_pos, x, bayes):
        (loss, parts), grads = lg(model, tokens, sym_pos, query_pos, guess_pos, x, bayes)
        grads, _ = optim.clip_grad_norm(tree_map(lambda g: g.astype(mx.float32), grads), 1.0)
        opt.learning_rate = rate
        master.update(opt.apply_gradients(grads, master))
        model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
        return loss, parts[0], parts[1]

    trace = out / "trace.jsonl"
    trace.write_text("")

    def log_eval(step):
        s = summary(evaluate(model, xv, bayes_eval, valid_logp))
        with trace.open("a") as f:
            f.write(json.dumps({"step": step, "valid": s}) + "\n")
        say(
            f"[gt report-train] step {step:5d}: excess {s['excess']:.4f}  guess right {s['accuracy']:.3f}  "
            f"report {s['report']:.3f}  |r - c| {s['report_mae']:.3f}  corr(r, c) {s['corr_r_c']:.2f}  "
            f"|m - c| {s['implicit_mae']:.3f}"
        )
        return s

    log_eval(0)
    t0 = time.time()
    for s in range(1, steps + 1):
        b = batch(np.asarray(xt[rng.integers(0, len(xt), batch_size)]), rng)
        loss, sl, rl = step_fn(
            mx.array(schedule(s - 1), dtype=mx.float32),
            *(mx.array(b[k]) for k in ("tokens", "sym_pos", "query_pos", "guess_pos", "x", "bayes")),
        )
        mx.eval(state, loss)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss is {loss.item()} at step {s}")
        if s % 50 == 0:
            with trace.open("a") as f:
                f.write(
                    json.dumps({"step": s, "loss": loss.item(), "symbol_loss": sl.item(), "report_loss": rl.item()})
                    + "\n"
                )
        if s % eval_every == 0 or s == steps:
            final = log_eval(s)
    (out / "model").mkdir(exist_ok=True)
    mx.save_safetensors(str(out / "model" / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (out / "model" / "config.json").write_text(json.dumps(cfg_model, indent=1))
    (out / "summary.json").write_text(
        json.dumps(
            {
                **cfg,
                "final": final,
                "train_seconds": round(time.time() - t0, 1),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=1,
        )
    )
    say(f"[gt report-train] done in {(time.time() - t0) / 60:.1f} min -> {out}")
    return out


# ── after compression ─────────────────────────────────────────────────────────────────────────────


def _slope(dy, dx, B=1000, seed=0) -> dict:
    """OLS slope of dy on dx over all queries (n, Q), with a bootstrap over sequences."""

    def fit(a, b):
        a, b = a.ravel(), b.ravel()
        return float(((a - a.mean()) * (b - b.mean())).sum() / ((b - b.mean()) ** 2).sum())

    draws = [fit(dy[i], dx[i]) for i in resample_sequences(len(dy), B, seed)]
    return interval(fit(dy, dx), np.array(draws))


def measures(base: dict, comp: dict) -> dict:
    dc, dr, dm = comp["c"] - base["c"], comp["r"] - base["r"], comp["m"] - base["m"]
    worse, same = dc < -0.1, np.abs(dc) < 0.02
    out = {
        "accuracy": mean_over(comp["c"]),
        "d_accuracy": mean_over(dc),
        "report": mean_over(comp["r"]),
        "report_gap": mean_over(comp["r"] - comp["c"]),
        "implicit_gap": mean_over(comp["m"] - comp["c"]),
        "report_brier": mean_over((comp["r"] - comp["c"]) ** 2),
        "implicit_brier": mean_over((comp["m"] - comp["c"]) ** 2),
        "share_worse": float(worse.mean()),
    }
    if worse.sum() >= 20:
        out |= {
            "follow_slope_report": _slope(dr, dc),
            "follow_slope_implicit": _slope(dm, dc),
            "worse_dr": mean_over(dr, worse),
            "worse_dm": mean_over(dm, worse),
            "worse_auc_report": auc_over(-dr, worse, worse | same),
            "worse_auc_implicit": auc_over(-dm, worse, worse | same),
        }
    return out


def calibration_with_queries(n: int = 128, seed: int = 42) -> list[list[int]]:
    """As compress.calibration (128 training documents, seed 42), with queries inserted as in training."""
    xt = np.load(D.data_dir("zipf") / "train.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(xt), n, replace=False))
    return with_queries(np.asarray(xt[rows]), random_after(n, rng))["tokens"].tolist()


def run(run_name: str, settings: list[str], say=print) -> Path:
    from ..models import apply_memory_policy
    from .compress import compressed
    from .train import provenance

    apply_memory_policy()
    path = D.GT / "runs" / run_name
    _, xv, vb, _ = D.load("zipf")
    bayes_eval = bayes_after(xv, np.tile(EVAL_AFTER, (len(xv), 1)))
    valid_logp = vb["logp"].astype(np.float64)
    out = D.GT / "report" / run_name
    out.mkdir(parents=True, exist_ok=True)
    calib = calibration_with_queries()
    results, base = {}, None
    for st in ["base", *settings]:
        f = out / f"{st.replace(':', 'of')}.npz"
        if f.exists():
            ev = dict(np.load(f))
        else:
            ev = evaluate(compressed(path, st, calib), xv, bayes_eval, valid_logp)
            np.savez(f, **ev)
            mx.clear_cache()
        base = ev if st == "base" else base
        results[st] = {"summary": summary(ev), **(measures(base, ev) if st != "base" else {})}
        r = results[st]
        msg = f"[gt report] {st:>12}: guess right {r['summary']['accuracy']:.3f}  report {r['summary']['report']:.3f}"
        if "follow_slope_report" in r:
            msg += (
                f"  follows: report {r['follow_slope_report']['est']:.2f}, implicit {r['follow_slope_implicit']['est']:.2f}"
                f"  AUC report {r['worse_auc_report']['est']:.2f}, implicit {r['worse_auc_implicit']['est']:.2f}"
            )
        say(msg)
    (out / "report.json").write_text(
        json.dumps(
            {
                "run": run_name,
                "queries_after": EVAL_AFTER.tolist(),
                "provenance": provenance("zipf"),
                "settings": results,
            },
            indent=1,
        )
    )
    return out
