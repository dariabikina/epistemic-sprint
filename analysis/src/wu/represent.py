# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""What compression does to the model's internal states, and whether the model "knows that it knows".

Not the embedding table: every method here keeps it in bf16 (as in Wu). What changes is the residual stream,
so we record it: the hidden state at the *last prompt token* of each PopQA question (the state from which the
answer is generated), after every few blocks. Recording works on any architecture: each block is wrapped and
its output kept.

Three questions, all asked of the same 3,000 questions under the base and a compressed model:

1. Drift.  How far does each question's state move (paired cosine distance, per layer)? Do questions keep
   their nearest neighbours (k-NN overlap)? Is drift larger for facts the compressed model lost?
2. Probe ("P(IK)", Kadavath et al. 2022).  A linear probe on the base model's states predicts whether the base
   answers correctly. Applied to the compressed model's states, does it still say "known" on lost facts?
3. Self-report.  P(Yes) for "Do you know the answer to this question? Answer Yes or No." Does the model's
   own report track what it actually knows, before and after compression?

Pictures (PCA, or UMAP if installed) are fitted once on base and compressed states together; drift is measured
in the original space, never in the picture.
"""

from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .generate import MODES, prompt_tokens

SELF_REPORT = "Do you know the answer to the following question? Answer only Yes or No.\nQuestion: {q}"


def record_layers(n: int) -> list[int]:
    return sorted({*range(3, n, 4), n - 1})


class _Keep(nn.Module):
    def __init__(self, block, sink: list):
        super().__init__()
        self.block, self.sink = block, sink

    def __call__(self, x, *args, **kwargs):
        y = self.block(x, *args, **kwargs)
        self.sink.append(y[:, -1, :].astype(mx.float32))
        return y


def states(model, tokenizer, rows: list[dict], batch: int = 32, dropped: list[int] = ()) -> dict:
    """Last-prompt-token hidden states after the recorded blocks (short-answer prompt), and P(Yes) for the
    self-report prompt. Prompts are batched by equal length, so no padding is needed.

    Layers are named by the *original* block index. When blocks were removed (ShortGPT, `dropped`), the state
    after original block L is the output of the last kept block at or before L: a removed block is the
    identity on the residual stream. (Before 14 Nov this used positions in the shortened stack, which
    misaligned ShortGPT states with the base model's after the first removed block.)"""
    n_orig = len(model.layers) + len(dropped)
    kept = [i for i in range(n_orig) if i not in set(dropped)]
    layers = record_layers(n_orig)
    where = {L: max(j for j, k in enumerate(kept) if k <= L) for L in layers}
    short_messages = MODES["short"][0]
    yes = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes")]
    no = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No")]
    prompts = [prompt_tokens(tokenizer, short_messages(r["question"])) for r in rows]
    asks = [prompt_tokens(tokenizer, [{"role": "user", "content": SELF_REPORT.format(q=r["question"])}]) for r in rows]
    H = np.zeros((len(rows), len(layers), model.args.hidden_size), dtype=np.float16)
    p_yes = np.zeros(len(rows), dtype=np.float32)
    originals = list(model.layers)
    sinks = {j: [] for j in set(where.values())}
    for j in sinks:
        model.layers[j] = _Keep(originals[j], sinks[j])
    try:
        by_len = defaultdict(list)
        for k, p in enumerate(prompts):
            by_len[len(p)].append(k)
        for ks in by_len.values():
            for s in range(0, len(ks), batch):
                part = ks[s : s + batch]
                for sink in sinks.values():
                    sink.clear()
                model.model(mx.array([prompts[k] for k in part]))
                for j, L in enumerate(layers):
                    H[part, j] = np.array(sinks[where[L]][0]).astype(np.float16)
    finally:
        for j in sinks:
            model.layers[j] = originals[j]
    by_len = defaultdict(list)
    for k, p in enumerate(asks):
        by_len[len(p)].append(k)
    for ks in by_len.values():
        for s in range(0, len(ks), batch):
            part = ks[s : s + batch]
            logits = model(mx.array([asks[k] for k in part]))[:, -1, :].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            py = mx.logsumexp(lp[:, yes], axis=-1)
            pn = mx.logsumexp(lp[:, no], axis=-1)
            p_yes[part] = np.array(mx.exp(py - mx.logaddexp(py, pn)))
    return {"layers": np.array(layers), "H": H, "p_yes": p_yes}


# ── analysis ───────────────────────────────────────────────────────────────────────────────────


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks of ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cosine_drift(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A, B = A.astype(np.float64), B.astype(np.float64)
    return 1 - (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def knn_overlap(A: np.ndarray, B: np.ndarray, k: int = 10) -> np.ndarray:
    """Per question: share of its k nearest neighbours (cosine) under A that are still among its k under B."""

    def nn_idx(X):
        X = X.astype(np.float64)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]

    a, b = nn_idx(A), nn_idx(B)
    return np.array([len(set(a[i]) & set(b[i])) / k for i in range(len(A))])


class Probe:
    """Ridge-regression probe on all standardized dimensions. No PCA first: the 'knows it' direction need not
    be a high-variance one, and PCA would drop it. The ridge strength is chosen by inner 3-fold AUC."""

    LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __init__(self, ridge: float | None = None, seed: int = 0):
        self.ridge, self.seed = ridge, seed

    @staticmethod
    def _solve(Z, y, lam):
        P = np.c_[Z, np.ones(len(Z))]
        reg = lam * np.eye(P.shape[1])
        reg[-1, -1] = 0  # don't shrink the intercept
        return np.linalg.solve(P.T @ P + reg, P.T @ y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X.astype(np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        lam = self.ridge
        if lam is None:
            fold = np.random.default_rng(self.seed).integers(0, 3, len(Z))
            scores = []
            for l in self.LAMBDAS:
                s = np.zeros(len(Z))
                for f in range(3):
                    w = self._solve(Z[fold != f], y[fold != f], l)
                    s[fold == f] = np.c_[Z[fold == f], np.ones((fold == f).sum())] @ w
                scores.append(auc(s, y))
            lam = self.LAMBDAS[int(np.nanargmax(scores))]
        self.lam, self.w = lam, self._solve(Z, y, lam)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = (X.astype(np.float64) - self.mu) / self.sd
        return np.c_[Z, np.ones(len(Z))] @ self.w


def outcome(base: list[dict], comp: list[dict]) -> np.ndarray:
    """0 never known, 1 kept, 2 lost, 3 gained."""
    b = np.array([r["correct"] for r in base])
    c = np.array([r["correct"] for r in comp])
    return np.where(b & c, 1, np.where(b & ~c, 2, np.where(~b & c, 3, 0)))


def analyse(
    base_states: dict, comp_states: dict, base: list[dict], comp: list[dict], folds: int = 5, seed: int = 0
) -> dict:
    """Drift, neighbourhoods, probe and self-report, per recorded layer."""
    y0 = np.array([r["correct"] for r in base], dtype=float)
    yc = np.array([r["correct"] for r in comp], dtype=float)
    out_cat = outcome(base, comp)
    names = {0: "never", 1: "kept", 2: "lost", 3: "gained"}
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(base))
    res = {"layers": [], "self_report": {}}
    for j, layer in enumerate(base_states["layers"]):
        A, B = base_states["H"][:, j], comp_states["H"][:, j]
        drift = cosine_drift(A, B)
        knn = knn_overlap(A, B)
        s0, sc = np.zeros(len(base)), np.zeros(len(base))
        for f in range(folds):  # cross-fitted: every question is scored by a probe that never saw it
            tr, te = fold != f, fold == f
            probe = Probe().fit(A[tr], y0[tr])
            s0[te], sc[te] = probe.score(A[te]), probe.score(B[te])
        row = {
            "layer": int(layer),
            "drift": {names[k]: float(np.median(drift[out_cat == k])) for k in names if (out_cat == k).any()},
            "knn_overlap": float(knn.mean()),
            "probe_auc_base": auc(s0, y0),  # does the base model's state say whether it knows?
            "probe_auc_comp_same_probe": auc(sc, yc),  # does the same direction still work after compression?
            "probe_on_lost_vs_kept": auc(
                sc[out_cat >= 1][(out_cat[out_cat >= 1] != 3)],
                (out_cat[(out_cat >= 1) & (out_cat != 3)] == 1).astype(float),
            ),
            "probe_mean": {names[k]: float(sc[out_cat == k].mean()) for k in names if (out_cat == k).any()},
        }
        res["layers"].append(row)
    p0, pc = base_states["p_yes"], comp_states["p_yes"]
    res["self_report"] = {
        "auc_base": auc(p0, y0),
        "auc_comp": auc(pc, yc),
        "mean_p_yes_comp": {names[k]: float(pc[out_cat == k].mean()) for k in names if (out_cat == k).any()},
        "mean_p_yes_base": {names[k]: float(p0[out_cat == k].mean()) for k in names if (out_cat == k).any()},
    }
    return res
