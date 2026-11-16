# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Layer-by-layer readings of a Zipf-world model, base and compressed.

    gt inside RUN [--settings ...] [--n 256]  ->  <GT>/inside/RUN/{report.json, lens.png, layers.png, layer_points.npz}
    gt inside RUN --plot                      ->  layers.png redrawn from layer_points.npz

States. h_0 = embedding output, h_l = output of block l, at every position of the first n validation
sequences. State at position t predicts token t.

Lenses (per layer; intervals by cluster bootstrap over sequences):
    logit lens      (nostalgebraist (2020) "interpreting GPT: the logit lens", LessWrong) decode(h_l) = log softmax of the tied unembedding of the final RMSNorm of h_l, over the
                    256 symbols.
    Jacobian lens   decode(Jbar_l h_l), Jbar_l = mean of d h_L[t] / d h_l[t] over 32 validation sequences x
                    positions (3, 31, 127, 255), by autodiff in float32 (Gurnee et al. 2026). Split-half
                    check: Jbar from sequences 1-16 and 17-32 separately.
    Readings: KL(Bayes || lens); log p of the actual token on new and on repeat tokens, and on new tokens by
    G0 rank band; mass on already-seen symbols; KL(G0 || lens at BOS); for compressed models the paired
    change in KL against base.

Probes. Ridge regression from h_l to: Bayes P(new), entropy of the Bayes posterior over alpha, number of
distinct symbols seen, position. 6,000 random (sequence, position) points. 4-fold cross-fitting by
sequence; ridge strength from {1, 10, 100, 1000} on an inner split by sequence; features standardized,
intercept unpenalized (ridge: Hoerl, A. E., Kennard, R. W. (1970) Technometrics 12, 55-67; cross-fitting
by group as in Chernozhukov, V. et al. (2018) Double/debiased machine learning, Econometrics Journal 21,
C1-C68). Reported as R^2 with an interval by resampling sequences; also "within document",
after subtracting each sequence's mean from features and target.

Drift. 150 points of each kind (new, G0 rank < 16; new, rank >= 16; repeat seen <= 8 tokens back; repeat
seen earlier). Per layer and kind: mean cosine distance between base and compressed state at the same
point, interval by resampling sequences.

Layer figure (for looking only). States at 2,000 random (sequence, position) points, saved in
layer_points.npz. One UMAP (McInnes, Healy & Melville 2018, arXiv:1802.03426; cosine, 30 neighbours,
min_dist 0.2) fitted on the base states of all layers; compressed states placed with `transform`. Rows =
models, columns = layers; colour = log10 Bayes P(next symbol is new).
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import data as D
from . import priors as P
from .measures import RANK_BANDS, token_kinds
from .stats import fmt, interval, mean_over, r2_over
from .train import provenance, with_bos

KINDS = ["new · head (rank ≤ 16)", "new · tail (rank > 16)", "repeat · recent (≤ 8 back)", "repeat · old (> 8 back)"]
JACOBIAN_POSITIONS = (3, 31, 127, 255)


# ── states and lenses ─────────────────────────────────────────────────────────────────────────────


def forward_states(model, tokens: mx.array) -> list[mx.array]:
    """[h_0, h_1, ..., h_L] for a batch of token ids."""
    hs = [model.model.embed_tokens(tokens)]
    for layer in model.layers:
        hs.append(layer(hs[-1], "causal", None))
    return hs


def all_states(model, x: np.ndarray, batch: int = 32) -> np.ndarray:
    """(L + 1, n, T, d) float16."""
    parts = []
    for s in range(0, len(x), batch):
        hs = mx.stack(forward_states(model, mx.array(with_bos(x[s : s + batch]))[:, :-1])).astype(mx.float16)
        mx.eval(hs)
        parts.append(np.array(hs))
    return np.concatenate(parts, axis=1)


def decode(model, h: mx.array) -> mx.array:
    """Log-probabilities over the 256 symbols from a residual-stream vector: final norm, tied unembedding."""
    lg = model.model.embed_tokens.as_linear(model.model.norm(h)).astype(mx.float32)[..., : P.NBINS]
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def jacobian(model, h_l: mx.array, l: int, t: int) -> np.ndarray:
    """d h_L[t] / d h_l[t] for one sequence (h_l: (1, T, d)): one backward pass over d copies, where copy k's
    loss is component k of the output, so its gradient is row k of the Jacobian. Positions 0..max(t, 31) are
    passed: causal attention makes later ones irrelevant, and MLX's attention has a gradient only for its
    full kernel, which very short sequences do not use."""
    d = h_l.shape[-1]
    h_in = mx.repeat(h_l[:, : max(t + 1, 32)], d, axis=0).astype(mx.float32)
    eye = mx.eye(d)

    def component_k_of_copy_k(h):
        for layer in model.layers[l:]:
            h = layer(h, "causal", None)
        return (h[:, t, :] * eye).sum()

    g = mx.grad(component_k_of_copy_k)(h_in)[:, t, :]
    mx.eval(g)
    return np.array(g)


def mean_jacobians(model, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Jbar_l for l = 0..L-1, (L, d, d): the mean over `rows` x JACOBIAN_POSITIONS."""
    L, d = len(model.layers), model.args.hidden_size
    J = np.zeros((L, d, d))
    for i in rows:
        hs = forward_states(model, mx.array(with_bos(x[i : i + 1]))[:, :-1])
        for t in JACOBIAN_POSITIONS:
            for l in range(L):
                J[l] += jacobian(model, hs[l], l, t)
    return J / (len(rows) * len(JACOBIAN_POSITIONS))


def labels(x: np.ndarray, p0: np.ndarray) -> dict:
    """Per position: new or repeat, G0 rank, which symbols were already seen, and the four trajectory kinds."""
    k = token_kinds(x, p0)
    seen = P.seen_before(x)
    back = np.full(x.shape, -1)  # distance to the previous occurrence of the same symbol
    for i in range(len(x)):
        last = {}
        for t, v in enumerate(x[i]):
            back[i, t] = t - last[v] if v in last else -1
            last[v] = t
    kind = np.where(k["new"], np.where(k["rank"] < 16, 0, 1), np.where(back <= 8, 2, 3))
    return {"new": k["new"], "rank": k["rank"], "seen": seen, "kind": kind, "distinct": seen.sum(-1)}


def read_lens(model, h: np.ndarray, x: np.ndarray, B: np.ndarray, lab: dict, p0: np.ndarray) -> dict:
    """The readings of one lens at one layer. h: (n, T, d) already multiplied by Jbar for the Jacobian lens."""
    lp = np.array(decode(model, mx.array(h.astype(np.float32)))).astype(np.float64)
    a = P.take(lp, x)
    lB = np.log(np.maximum(B, P.TINY))
    kl = (B * (lB - lp)).sum(-1)
    return {
        "_kl_per_position": kl,  # kept in memory for paired base-vs-compressed differences; not written
        "kl_bayes": mean_over(kl),
        "logp_new": mean_over(a, lab["new"]),
        "logp_repeat": mean_over(a, ~lab["new"]),
        "logp_new_by_rank": {
            f"{lo + 1}-{hi}": mean_over(a, lab["new"] & (lab["rank"] >= lo) & (lab["rank"] < hi))
            for lo, hi in RANK_BANDS
        },
        "seen_mass": mean_over((np.exp(lp) * lab["seen"]).sum(-1)),
        "prior_kl": float((p0 * (np.log(p0) - lp[0, 0])).sum()),  # BOS position: the same for every sequence
    }


def lenses(model, H: np.ndarray, x: np.ndarray, B: np.ndarray, lab: dict, p0: np.ndarray, seed: int = 0) -> dict:
    L = H.shape[0] - 1
    rows = np.random.default_rng(seed).choice(len(x), 32, replace=False)
    halves = [mean_jacobians(model, x, rows[:16]), mean_jacobians(model, x, rows[16:])]
    J = (halves[0] + halves[1]) / 2
    out = {
        "logit": [],
        "jacobian": [],
        "jacobian_split_half": [],
        "seen_mass_bayes": mean_over((B * lab["seen"]).sum(-1)),
    }
    for l in range(L + 1):
        out["logit"].append(read_lens(model, H[l], x, B, lab, p0))
        if l == L:  # at the last layer the Jacobian is the identity
            out["jacobian"].append(dict(out["logit"][-1]))
            continue
        out["jacobian"].append(read_lens(model, H[l].astype(np.float32) @ J[l].T, x, B, lab, p0))
        kl_halves = [
            read_lens(model, H[l].astype(np.float32) @ Jh[l].T, x, B, lab, p0)["kl_bayes"]["est"] for Jh in halves
        ]
        out["jacobian_split_half"].append(
            {
                "kl_bayes_each_half": kl_halves,
                "relative_difference": float(np.linalg.norm(halves[0][l] - halves[1][l]) / np.linalg.norm(J[l])),
            }
        )
    return out


# ── probes ────────────────────────────────────────────────────────────────────────────────────────


def cross_fitted_ridge(X, y, groups, lams=(1.0, 10.0, 100.0, 1000.0), folds=4) -> np.ndarray:
    """Predictions for every row from a ridge fitted without that row's sequence. Centred on the training fold
    (the intercept is not shrunk); the ridge strength is chosen on an inner split by sequence."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    def fit(Xa, ya, lam):
        mx_, my = Xa.mean(0), ya.mean()
        w = np.linalg.solve((Xa - mx_).T @ (Xa - mx_) + lam * np.eye(Xa.shape[1]), (Xa - mx_).T @ (ya - my))
        return lambda Z: (Z - mx_) @ w + my

    pred = np.zeros_like(y)
    for f in range(folds):
        test = groups % folds == f
        Xa, ya, inner = X[~test], y[~test], (groups[~test] // folds) % 2 == 0
        lam = min(lams, key=lambda l: ((fit(Xa[inner], ya[inner], l)(Xa[~inner]) - ya[~inner]) ** 2).mean())
        pred[test] = fit(Xa, ya, lam)(X[test])
    return pred


def _within(A: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Remove each sequence's own mean: what varies inside a document."""
    A = A.astype(np.float64).copy()
    for g in np.unique(groups):
        A[groups == g] -= A[groups == g].mean(0)
    return A


def probes(H: np.ndarray, x: np.ndarray, world: str, lab: dict, n_points: int = 6000, seed: int = 0) -> dict:
    _, ex = P.worlds()[world].predict(x)
    rng = np.random.default_rng(seed)
    i, t = rng.integers(0, len(x), n_points), rng.integers(1, P.T, n_points)
    targets = {
        "P(new) (Bayes)": ex["novel"][i, t],
        "alpha uncertainty (Bayes, nats)": ex["hyp_entropy"][i, t],
        "distinct symbols seen": lab["distinct"][i, t],
        "position (a cue)": t,
    }
    out = {}
    for name, y in targets.items():
        y = y.astype(np.float64)
        out[name], out[name + " · within document"] = [], []
        for l in range(H.shape[0]):
            X = H[l, i, t].astype(np.float64)
            out[name].append(r2_over(y, cross_fitted_ridge(X, y, i), i))
            yw = _within(y[:, None], i)[:, 0]
            out[name + " · within document"].append(r2_over(yw, cross_fitted_ridge(_within(X, i), yw, i), i))
    return out


# ── trajectories ──────────────────────────────────────────────────────────────────────────────────


def _unit(A):
    A = A.astype(np.float32)
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-9)


def trajectory_points(lab: dict, per_kind: int = 150, seed: int = 0) -> np.ndarray:
    """(m, 3): sequence, position, kind."""
    rng = np.random.default_rng(seed)
    pts = []
    for k in range(4):
        ii, tt = np.nonzero(lab["kind"][:, 1:] == k)
        for s in rng.choice(len(ii), min(per_kind, len(ii)), replace=False):
            pts.append((ii[s], tt[s] + 1, k))
    return np.array(pts)


def drift(Hb: np.ndarray, Hc: np.ndarray, pts: np.ndarray, B: int = 1000, seed: int = 0) -> dict:
    """Per kind and layer: mean cosine distance base -> compressed at the same point, with an interval from
    resampling sequences."""
    out = {}
    rng = np.random.default_rng(seed)
    for k in range(4):
        p = pts[pts[:, 2] == k]
        seqs = np.unique(p[:, 0])
        out[KINDS[k]] = []
        for l in range(Hb.shape[0]):
            d = 1 - (_unit(Hb[l, p[:, 0], p[:, 1]]) * _unit(Hc[l, p[:, 0], p[:, 1]])).sum(-1)
            by_seq = {s: d[p[:, 0] == s] for s in seqs}
            draws = [np.concatenate([by_seq[s] for s in rng.choice(seqs, len(seqs))]).mean() for _ in range(B)]
            out[KINDS[k]].append(interval(d.mean(), np.array(draws)))
    return out


def layer_points(x: np.ndarray, world: str, m: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """m random (sequence, position >= 1) points and log10 Bayes P(next symbol is new) at each."""
    rng = np.random.default_rng(seed)
    i, t = rng.integers(0, len(x), m), rng.integers(1, P.T, m)
    pnew = P.worlds()[world].predict(x)[1]["novel"][i, t]
    return np.c_[i, t], np.log10(np.maximum(pnew, 1e-3))


def plot_layers(out: Path, title: str, seed: int = 0):
    """Grid from <out>/layer_points.npz: rows = models, columns = layers. One UMAP (cosine) fitted on the base
    states of all layers; other models placed with `transform`. Colour: log10 Bayes P(new)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    z = np.load(out / "layer_points.npz")
    names = [k.removeprefix("H_") for k in z.files if k.startswith("H_")]
    L1, m = z["H_base"].shape[:2]
    flat = lambda H: _unit(H.reshape(-1, H.shape[-1]))
    mapper = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=seed).fit(flat(z["H_base"]))
    Y = {
        n: (mapper.embedding_ if n == "base" else mapper.transform(flat(z["H_" + n]))).reshape(L1, m, 2) for n in names
    }
    lo, hi = np.percentile(Y["base"].reshape(-1, 2), [1, 99], axis=0)
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    fig, axes = plt.subplots(len(names), L1, figsize=(1.5 * L1 + 1, 1.5 * len(names) + 0.5), squeeze=False)
    for r, n in enumerate(names):
        for l in range(L1):
            ax = axes[r, l]
            ax.scatter(*Y["base"].reshape(-1, 2).T, s=0.3, color="0.9", lw=0, rasterized=True, zorder=0)
            sc = ax.scatter(
                *Y[n][l].T, s=1.5, c=z["colour"], cmap="viridis", vmin=-3, vmax=0, lw=0, rasterized=True, zorder=1
            )
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title("embedding" if l == 0 else f"after block {l - 1}", fontsize=7)
            if l == 0:
                ax.set_ylabel(n, fontsize=7)
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01, ticks=[-3, -2, -1, 0])
    cb.ax.set_yticklabels(["≤ 0.001", "0.01", "0.1", "1"], fontsize=6)
    cb.set_label("Bayes P(next symbol is new)", fontsize=7)
    fig.suptitle(
        f"{title}: one UMAP of all base states (grey); each panel colours one model's states at one layer", fontsize=7
    )
    fig.savefig(out / "layers.png", dpi=200)
    plt.close(fig)


def plot_lens(report: dict, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = list(report["settings"])
    panels = [
        ("kl_bayes", "KL(Bayes ‖ lens), nats"),
        ("logp_new", "log p, next token new"),
        ("logp_repeat", "log p, next token a repeat"),
        ("seen_mass", "mass on symbols already seen"),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 5.4), sharex=True)
    colors = ["black"] + [f"C{i}" for i in range(len(settings))]
    for r, lens in enumerate(("logit", "jacobian")):
        for c, (key, label) in enumerate(panels):
            ax = axes[r, c]
            for s, color in zip(settings, colors):
                v = [e[key] for e in report["settings"][s]["lens"][lens]]
                x = np.arange(len(v))
                ax.fill_between(x, [e["lo"] for e in v], [e["hi"] for e in v], color=color, alpha=0.15, lw=0)
                ax.plot(x, [e["est"] for e in v], color=color, lw=1.6 if s == "base" else 1.0, label=s)
            if key == "seen_mass":
                ax.axhline(report["settings"]["base"]["lens"]["seen_mass_bayes"]["est"], color="0.5", ls="--", lw=0.8)
            if key == "kl_bayes":
                ax.set_yscale("log")
            ax.set_title(f"{lens} lens: {label}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("layer (0 = embedding)", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title}. Bands: 95% cluster-bootstrap intervals over sequences; dashed: Bayes.", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── the experiment ────────────────────────────────────────────────────────────────────────────────


def run(run_name: str, settings: list[str], n: int = 256, say=print) -> Path:
    from ..models import apply_memory_policy
    from .compress import calibration, compressed

    apply_memory_policy()
    path = D.GT / "runs" / run_name
    world = json.loads((path / "config.json").read_text())["world"]
    x = D.load(world)[1][:n]
    p0 = P.worlds()[world].p0()
    lab = labels(x, p0)
    B = P.worlds()[world].predict(x)[0]
    out = D.GT / "inside" / run_name
    out.mkdir(parents=True, exist_ok=True)
    calib = calibration(world)
    report = {
        "run": run_name,
        "world": world,
        "n_sequences": n,
        "settings": {},
        "intervals": "95% cluster bootstrap over sequences (stats.py)",
        "provenance": provenance(world),
    }
    H = {}
    for setting in ["base", *settings]:
        model = compressed(path, setting, calib)
        H[setting] = all_states(model, x)
        r = report["settings"][setting] = {
            "lens": lenses(model, H[setting], x, B, lab, p0),
            "probes": probes(H[setting], x, world, lab),
        }
        say(
            f"[gt inside] {run_name} {setting}: KL(Bayes‖lens) by layer — logit "
            + ", ".join(fmt(e["kl_bayes"], 2) for e in r["lens"]["logit"])
            + " — jacobian "
            + ", ".join(fmt(e["kl_bayes"], 2) for e in r["lens"]["jacobian"])
        )
        for name, vals in r["probes"].items():
            say(f"           probe R², {name}: " + ", ".join(fmt(v, 2) for v in vals))
        del model
        mx.clear_cache()
    # paired: per layer and lens, KL(compressed) - KL(base) on the same positions
    for r in report["settings"].values():
        for lens in ("logit", "jacobian"):
            for e, e0 in zip(r["lens"][lens], report["settings"]["base"]["lens"][lens]):
                e["d_kl_bayes_vs_base"] = mean_over(e["_kl_per_position"] - e0["_kl_per_position"])
    for r in report["settings"].values():
        for lens in ("logit", "jacobian"):
            for e in r["lens"][lens]:
                e.pop("_kl_per_position", None)
    pts = trajectory_points(lab)
    report["drift"] = {name: drift(H["base"], Hc, pts) for name, Hc in H.items() if name != "base"}
    idx, colour = layer_points(x, world)
    np.savez(
        out / "layer_points.npz",
        points=idx,
        colour=colour,
        **{"H_" + name: Hs[:, idx[:, 0], idx[:, 1]] for name, Hs in H.items()},
    )
    plot_layers(out, run_name)
    plot_lens(report, out / "lens.png", run_name)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    say(f"[gt inside] wrote {out}")
    return out
