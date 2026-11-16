# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Base models (pinned) and how a compression *setting* turns one into the model we evaluate.

A setting is a short name: "base", "rtn-4", "gptq-3", "awq-4", "magnitude-30", "wanda-50",
"wanda-2:4", "shortgpt-10". Quantized settings are built once and saved; pruned settings are
rebuilt at load time from the base weights plus a small recipe (masks are cheap to recompute,
and a pruned 8B model stored dense would cost another 16 GB per setting).
"""

import os
from pathlib import Path

import mlx.core as mx
import mlx_lm.generate
from mlx_lm import load as mlx_load

# Outside the cowrite-synced folder on purpose: weights, generations, figures.
RUNS = Path(os.environ.get("WU_RUNS", "~/models/wu-runs")).expanduser()

PRETRAINED = {  # name -> (Hugging Face repo, pinned revision); more in models.json (`wu pin`)
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen3-8b-base": ("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),  # only to check Wu Table 3
}

REGISTRY = Path(__file__).resolve().parents[2] / "models.json"
if REGISTRY.exists():
    import json as _json

    PRETRAINED.update({k: tuple(v) for k, v in _json.loads(REGISTRY.read_text()).items()})


def pin(name: str, repo: str) -> str:
    """Add a model to models.json at the repo's current revision (a pinned commit, so runs stay comparable)."""
    import json

    from huggingface_hub import HfApi

    sha = HfApi().model_info(repo).sha
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    registry[name] = [repo, sha]
    REGISTRY.write_text(json.dumps(registry, indent=1) + "\n")
    return sha


MEMORY_LIMIT = 48_000_000_000  # the whole working set


def apply_memory_policy():
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(4_000_000_000)
    mlx_lm.generate.maybe_set_recommended_wired_limit = lambda: mx.set_wired_limit(MEMORY_LIMIT)


def setting_dir(model: str, setting: str) -> Path:
    return RUNS / model / setting.replace(":", "of")


def parse(setting: str) -> tuple[str, str | None]:
    """'gptq-3' -> ('gptq', '3'); 'base' -> ('base', None)."""
    method, _, level = setting.partition("-")
    return method, level or None


QUANT = ("gptq", "awq")  # built once by `wu compress`, saved as MLX weights
PRUNE = ("wanda", "shortgpt")  # built once by `wu compress`, saved as a recipe and applied at load


def load(model: str, setting: str = "base"):
    """(model, tokenizer) for a base model under a compression setting."""
    repo, revision = PRETRAINED[model]
    method, level = parse(setting)
    built = setting_dir(model, setting)
    if method in QUANT:
        if not (built / "model" / "config.json").exists():
            raise FileNotFoundError(f"{setting} is not built: run `wu compress --model {model} --setting {setting}`")
        return mlx_load(str(built / "model"))
    m, tok = mlx_load(repo, revision=revision)
    if method == "base":
        return m, tok
    if method == "rtn":
        from .compress.quant import rtn

        return rtn(m, bits=int(level)), tok
    from .compress import prune

    if method == "magnitude":
        return prune.magnitude(m, level), tok
    if method in PRUNE and not (built / "recipe.json").exists():
        raise FileNotFoundError(f"{setting} is not built: run `wu compress --model {model} --setting {setting}`")
    if method == "wanda":
        return prune.wanda_apply(m, level, prune.load_recipe(built)), tok
    if method == "shortgpt":
        return prune.shortgpt_apply(m, prune.load_recipe(built)["dropped"]), tok
    raise ValueError(f"unknown setting {setting!r}")


def compress(model: str, setting: str, progress=print):
    """Build a setting that needs calibration data: GPTQ/AWQ weights, or a WANDA/ShortGPT recipe."""
    import json
    import time

    from mlx_lm.utils import save

    from .compress import prune, quant

    repo, revision = PRETRAINED[model]
    method, level = parse(setting)
    out, start = setting_dir(model, setting), time.time()
    if method in QUANT:
        m, tok, config = mlx_load(repo, revision=revision, return_config=True)
        getattr(quant, method)(m, tok, int(level), progress=progress)
        config["quantization"] = {"group_size": quant.GROUP_SIZE, "bits": int(level)}
        from huggingface_hub import snapshot_download

        save(
            out / "model", snapshot_download(repo, revision=revision), m, tok, config
        )  # a local path: save() resolves a bare repo name offline, without the revision
    elif method == "wanda":
        m, tok = mlx_load(repo, revision=revision)
        prune.save_recipe(out, {"method": "wanda", "level": level, **prune.wanda_recipe(m, tok, level, progress)})
    elif method == "shortgpt":
        m, tok = mlx_load(repo, revision=revision)
        influence = prune.block_influence(m, tok)
        prune.save_recipe(
            out,
            {
                "method": "shortgpt",
                "level": level,
                "block_influence": influence,
                "dropped": prune.shortgpt_drop(influence, int(level)),
            },
        )
    else:
        raise ValueError(f"{setting!r} needs no building (rtn and magnitude are applied at load)")
    (out / "built.json").write_text(json.dumps({"setting": setting, "seconds": round(time.time() - start)}) + "\n")
