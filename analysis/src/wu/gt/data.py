# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Frozen pools for each world: training tokens, and validation tokens with their Bayes answers.

    <GT>/data/<world>/train.npy      (n_train, 256) int16 tokens
    <GT>/data/<world>/valid.npy      (n_valid, 256) int16 tokens
    <GT>/data/<world>/valid_bayes.npz  per position: logp (Bayes log-probability of the actual token), entropy,
                                     alea_var, epi_var, hyp_entropy (and novel for zipf); per sequence: latent
    <GT>/data/<world>/MANIFEST.json  world parameters, seeds, sizes, sha256 of every file

Seeds are fixed per world (train 1000 + i, validation 2000 + i, i = the world's place in the ladder), so a pool
rebuilt anywhere is identical; `verify` recomputes the hashes before training. (Identical given the same
numpy/BLAS: samplers use matrix products, so a different BLAS can move a value across a bin edge; pools also
depend on `chunk`. Train and validation streams are independent, but worlds with atoms repeat whole
sequences: 11.6% of zipf-bursty validation sequences also occur in its training pool. That is not leakage,
since both are i.i.d. from the prior, but those positions do not test generalization to unseen sequences.)
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import priors as P

GT = Path(os.environ.get("WU_RUNS", Path.home() / "models" / "wu-runs")) / "gt"


def data_dir(world: str, tag: str = "") -> Path:
    return GT / "data" / (world + (f"-{tag}" if tag else ""))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(world: str, n: int, seed: int, chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    w = P.worlds()[world]
    rng = np.random.default_rng(seed)
    xs, hs = [], []
    for s in range(0, n, chunk):
        x, h = w.sample(min(chunk, n - s), rng)
        xs.append(x)
        hs.append(h)
    return np.concatenate(xs), np.concatenate(hs)


def bayes(world: str, x: np.ndarray, chunk: int = 16, say=print) -> dict:
    w = P.worlds()[world]
    out = {}
    t0 = time.time()
    for s in range(0, len(x), chunk):
        Pr, ex = w.predict(x[s : s + chunk])
        part = {"logp": np.log(P.take(Pr, x[s : s + chunk])), **ex}
        for k, v in part.items():
            out.setdefault(k, []).append(v.astype(np.float32))
        if s and s % (chunk * 16) == 0:
            say(f"  bayes {world}: {s}/{len(x)} sequences, {time.time() - t0:.0f}s")
    return {k: np.concatenate(v) for k, v in out.items()}


def build(world: str, n_train: int = 200_000, n_valid: int = 1024, tag: str = "", say=print) -> Path:
    i = P.ALL.index(world)
    d = data_dir(world, tag)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    xt, ht = sample(world, n_train, 1000 + i)
    xv, hv = sample(world, n_valid, 2000 + i)
    say(f"[gt data] {world}: sampled {n_train} + {n_valid} sequences in {time.time() - t0:.0f}s")
    b = bayes(world, xv, say=say)
    np.save(d / "train.npy", xt)
    np.save(d / "valid.npy", xv)
    np.savez(d / "valid_bayes.npz", latent=hv, **b)
    w = P.worlds()[world]
    counts = np.bincount(xt.ravel().astype(np.int64), minlength=P.NBINS) + 1.0
    unigram = float(-np.log((counts / counts.sum())[xv.astype(np.int64)]).mean())
    manifest = {
        "world": world,
        "blurb": w.blurb,
        "params": w.params,
        "tokens": {"bins": P.NBINS, "range": [P.LO, P.HI], "length": P.T},
        "seeds": {"train": 1000 + i, "valid": 2000 + i},
        "n_train": n_train,
        "n_valid": n_valid,
        "bayes_nats_per_token": float(-b["logp"].mean()),
        "unigram_nats_per_token": unigram,
        "clipped_share": float(((xt == 0) | (xt == P.NBINS - 1)).mean()),
        "files": {f.name: _sha(f) for f in sorted(d.glob("*.np*"))},
        "built_seconds": round(time.time() - t0, 1),
    }
    (d / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=float))
    say(
        f"[gt data] {world}: Bayes {manifest['bayes_nats_per_token']:.3f} nats/token, unigram {unigram:.3f}, "
        f"clipped {manifest['clipped_share']:.2%}, {time.time() - t0:.0f}s -> {d}"
    )
    return d


def verify(world: str, tag: str = "") -> list[str]:
    d = data_dir(world, tag)
    m = d / "MANIFEST.json"
    if not m.exists():
        return [f"{d} has no manifest: run `gt data {world}`"]
    files = json.loads(m.read_text())["files"]
    return [f"{d / f} changed" for f, h in files.items() if not (d / f).exists() or _sha(d / f) != h]


def load(world: str, tag: str = "") -> tuple[np.ndarray, np.ndarray, dict, dict]:
    d = data_dir(world, tag)
    problems = verify(world, tag)
    if problems:
        raise SystemExit("\n".join(problems))
    b = dict(np.load(d / "valid_bayes.npz"))
    return np.load(d / "train.npy"), np.load(d / "valid.npy"), b, json.loads((d / "MANIFEST.json").read_text())
