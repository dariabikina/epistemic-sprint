# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Train a small Llama from scratch on one world, and watch it approach the Bayes floor.

Model: Llama blocks as MLX-LM implements them (RMSNorm, RoPE, SwiGLU, tied embeddings), vocabulary = 256
bins + BOS. GPT-2-style initialization; bf16 compute, fp32 master weights and AdamW moments; linear warmup,
cosine decay to 10%; gradients clipped at 1.

Data: random rows of the frozen training pool, each BOS + 256 tokens; every token is a target.

Monitoring (trace.jsonl). Every 50 steps the training loss; every `eval_every` steps, on the frozen
validation pool (whose Bayes answers are known):
    nats           model log loss per token (over the 256 bins; the BOS logit is excluded)
    bayes          the floor: Bayes log loss on the same tokens
    excess         nats - bayes = an unbiased estimate of KL(Bayes || model), with a standard error over
                   sequences; also per position band (1, 2-8, 9-32, 33-128, 129-256)
    entropy_gap    mean model entropy - mean Bayes entropy: over- (+) or under- (-) dispersed
    tracking       correlation of model and Bayes predictive entropy over positions
    tracking_within  the same after removing each position's mean from both: does the model's uncertainty
                   follow *this* prefix's informativeness, or only how far into the sequence it is?
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models import llama

from . import data as D
from . import priors as P

BANDS = [(0, 1), (1, 8), (8, 32), (32, 128), (128, 256)]


@dataclass(frozen=True)
class Shape:
    layers: int
    width: int
    heads: int
    mlp: int

    def config(self) -> dict:
        return {
            "model_type": "llama",
            "hidden_size": self.width,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.heads,
            "num_key_value_heads": self.heads,
            "intermediate_size": self.mlp,
            "vocab_size": P.VOCAB,
            "max_position_embeddings": P.T + 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
        }


SHAPES = {
    "gt-1m": Shape(4, 128, 4, 352),
    "gt-6m": Shape(6, 256, 4, 688),
    "gt-20m": Shape(8, 384, 6, 1024),
    # for compression: every input width a multiple of 64, so group-wise quantization covers every matrix
    "gt-6m-q": Shape(6, 256, 4, 768),
    "gt-1m-q": Shape(4, 128, 4, 384),
}


def run_dir(world: str, shape: str, seed: int, tag: str = "") -> Path:
    return D.GT / "runs" / f"{world}-{shape}-s{seed}{('-' + tag) if tag else ''}"


def code_hashes() -> dict:
    here = Path(__file__).parent
    out = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(here.glob("*.py"))}
    lock = here.parents[2] / "pixi.lock"
    if lock.exists():
        out["pixi.lock"] = hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    return out


def provenance(world: str | None = None, data_tag: str = "") -> dict:
    """What an output depends on: code hashes, the data manifest's hash, library versions, time."""
    out = {
        "code": code_hashes(),
        "mlx": mx.__version__,
        "numpy": np.__version__,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if world:
        m = D.data_dir(world, data_tag) / "MANIFEST.json"
        out["data_manifest_sha256"] = hashlib.sha256(m.read_bytes()).hexdigest()
    return out


def initialize(model, layers: int):
    weights = []
    for name, value in tree_flatten(model.parameters()):
        if name.endswith(("o_proj.weight", "down_proj.weight")):
            weights.append((name, 0.02 / math.sqrt(2 * layers) * mx.random.normal(value.shape)))
        elif value.ndim >= 2:
            weights.append((name, 0.02 * mx.random.normal(value.shape)))
    model.load_weights(weights, strict=False)


def with_bos(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.full((len(x), 1), P.BOS, dtype=np.int32), x.astype(np.int32)], 1)


def loss_fn(model, tokens):
    return nn.losses.cross_entropy(model(tokens[:, :-1]).astype(mx.float32), tokens[:, 1:], reduction="mean")


def predict(model, x: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Model log-probability of each actual token and predictive entropy, over the 256 bins."""
    lp_out, h_out = [], []
    for s in range(0, len(x), batch):
        tok = mx.array(with_bos(x[s : s + batch]))
        lg = model(tok[:, :-1]).astype(mx.float32)[..., : P.NBINS]
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        actual = mx.take_along_axis(lp, tok[:, 1:, None], axis=-1)[..., 0]
        ent = -(mx.exp(lp) * lp).sum(-1)
        mx.eval(actual, ent)
        lp_out.append(np.array(actual))
        h_out.append(np.array(ent))
    return np.concatenate(lp_out), np.concatenate(h_out)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def compare(model_lp, model_h, bayes: dict) -> dict:
    ex = -model_lp + bayes["logp"]  # per position: model loss - Bayes loss
    per_seq = ex.mean(1)
    out = {
        "nats": float(-model_lp.mean()),
        "bayes": float(-bayes["logp"].mean()),
        "excess": float(per_seq.mean()),
        "excess_se": float(per_seq.std(ddof=1) / math.sqrt(len(per_seq))),
        "excess_by_band": {f"{a + 1}-{b}": float(ex[:, a:b].mean()) for a, b in BANDS},
        "entropy_gap": float(model_h.mean() - bayes["entropy"].mean()),
        "tracking": _corr(model_h.ravel(), bayes["entropy"].ravel()),
        "tracking_within": _corr(
            (model_h - model_h.mean(0)).ravel(), (bayes["entropy"] - bayes["entropy"].mean(0)).ravel()
        ),
    }
    return out


def train(
    world: str,
    shape: str = "gt-6m",
    seed: int = 0,
    steps: int = 8000,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup: int = 200,
    eval_every: int = 500,
    tag: str = "",
    data_tag: str = "",
    say=print,
) -> Path:
    from ..models import apply_memory_policy

    apply_memory_policy()
    mx.random.seed(seed)
    xt, xv, bayes, manifest = D.load(world, data_tag)
    sh = SHAPES[shape]
    out = run_dir(world, shape, seed, tag)
    out.mkdir(parents=True, exist_ok=True)
    model = llama.Model(llama.ModelArgs.from_dict(sh.config()))
    initialize(model, sh.layers)
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    tokens_per_step = batch_size * P.T
    cfg = {
        "world": world,
        "shape": shape,
        "architecture": sh.config(),
        "parameters": n_params,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "warmup": warmup,
        "eval_every": eval_every,
        "data": str(D.data_dir(world, data_tag)),
        "data_manifest_sha256": hashlib.sha256(
            (D.data_dir(world, data_tag) / "MANIFEST.json").read_bytes()
        ).hexdigest(),
        "code": code_hashes(),
        "mlx": mx.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    say(
        f"[gt train] {world} · {shape} ({n_params / 1e6:.2f}M) · seed {seed} · {steps} steps x {tokens_per_step} tokens "
        f"= {steps * tokens_per_step / 1e6:.0f}M tokens ({steps * batch_size / len(xt):.2f} passes) -> {out}\n"
        f"           floor (Bayes) {manifest['bayes_nats_per_token']:.3f} nats/token, unigram {manifest['unigram_nats_per_token']:.3f}"
    )

    pool = mx.array(with_bos(xt))
    master = tree_map(lambda p: p.astype(mx.float32), model.parameters())
    model.set_dtype(mx.bfloat16)
    schedule = lambda s: lr * min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.1, betas=(0.9, 0.95))
    opt.init(master)
    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, master, opt.state, mx.random.state]
    mx.eval(pool, state)

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(rate):
        rows = mx.random.randint(0, pool.shape[0], (batch_size,))
        loss, grads = lg(model, pool[rows])
        grads, _ = optim.clip_grad_norm(tree_map(lambda g: g.astype(mx.float32), grads), 1.0)
        opt.learning_rate = rate
        master.update(opt.apply_gradients(grads, master))
        model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
        return loss

    trace = out / "trace.jsonl"
    trace.write_text("")

    def log(rec):
        with trace.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def evaluate(step):
        lp, h = predict(model, xv)
        c = compare(lp, h, bayes)
        log({"step": step, "tokens": step * tokens_per_step, "valid": c})
        say(
            f"[gt train] {world} step {step:5d}: {c['nats']:.3f} nats (floor {c['bayes']:.3f}) excess {c['excess']:.4f}"
            f"±{c['excess_se']:.4f}  first-token {c['excess_by_band']['1-1']:+.3f}  entropy gap {c['entropy_gap']:+.3f}"
            f"  tracking {c['tracking']:.2f} / within {c['tracking_within']:.2f}"
        )
        return lp, h, c

    evaluate(0)
    seconds = 0.0
    for s in range(1, steps + 1):
        t0 = time.perf_counter()
        loss = step_fn(mx.array(schedule(s - 1), dtype=mx.float32))
        mx.eval(state, loss)
        seconds += time.perf_counter() - t0
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss is {loss.item()} at step {s}")
        if s % 50 == 0:
            log({"step": s, "tokens": s * tokens_per_step, "loss": loss.item(), "lr": schedule(s - 1)})
        if s % eval_every == 0 or s == steps:
            lp, h, c = evaluate(s)

    out_model = out / "model"
    out_model.mkdir(exist_ok=True)
    mx.save_safetensors(str(out_model / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (out_model / "config.json").write_text(json.dumps(sh.config(), indent=1))
    np.savez(out / "valid_model.npz", logp=lp, entropy=h)
    summary = {
        "world": world,
        "shape": shape,
        "seed": seed,
        "parameters": n_params,
        "steps": steps,
        "tokens": steps * tokens_per_step,
        "train_seconds": round(seconds, 1),
        "tokens_per_second": round(steps * tokens_per_step / seconds),
        "final": c,
        "unigram": manifest["unigram_nats_per_token"],
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    say(f"[gt train] {world} done in {seconds / 60:.1f} min: excess {c['excess']:.4f} nats/token over Bayes -> {out}")
    return out


def load_model(path: Path):
    cfg = json.loads((path / "model" / "config.json").read_text())
    model = llama.Model(llama.ModelArgs.from_dict(cfg))
    model.load_weights(str(path / "model" / "model.safetensors"))
    return model
