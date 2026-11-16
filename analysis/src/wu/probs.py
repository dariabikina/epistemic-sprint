# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Token probabilities on the epistemic-probabilities prompts (the dataset in ../epistemic-probabilities).

400 PopQA facts and 100 invented controls, in English and Russian, already rendered with Qwen3's chat
template. Three conditions, each run on one compression setting:

    short    the model's greedy answer, at most 24 new tokens                      -> short.jsonl
    scoring  log p of every candidate answer, under the prompt and under a null     -> scores.jsonl
             prompt (the same question about an invented subject); and at the          position_stats.jsonl
             answer position the entropy, the top 20 tokens and KL from the base model
    knows    log p of " Yes" and " No" to "Do you know the answer to this question?" -> scores.jsonl
             and the residual stream at the last prompt token                          knows-states.npz

Prompts are fed as raw strings (the template is already in them) and tokenized without special tokens
added. Each prompt is read once and its key-value cache shared by its continuations
(confidence.continuation_logps). Outputs go to <RUNS>/<model>/<setting>/probs-<tag>/, with run.json
recording what they depend on. The measures are in probs_report.py.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .confidence import continuation_logps, prefix_cache, read
from .generate import generate

HERE = Path(__file__).resolve().parent
# The prompt file is in the dataset repository. This code runs from <development repository>/model/src/wu,
# beside a clone of it,
# or from <epistemic-probabilities>/analysis/src/wu, inside it.
_UP = HERE.parents[2]
PROMPTS = (
    _UP if (_UP / "prompts_rendered.jsonl").exists() else _UP / "epistemic-probabilities"
) / "prompts_rendered.jsonl"
CONDITIONS = ("short", "scoring", "knows")
FILES = {"short": "short.jsonl", "scoring": "scores-scoring.jsonl", "knows": "scores-knows.jsonl"}
TOP = 20
BATCH = 16  # prompts read together
PREFILL = {"en": "Answer:", "ru": "Ответ:"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompts(path: Path = PROMPTS, n: int = 0) -> list[dict]:
    """Every rendered prompt. `n`: smoke runs, about n facts and controls spread over the file."""
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    if n:
        ids = sorted({r["id"] for r in rows})
        keep = set(ids[:: max(1, len(ids) // n)])
        rows = [r for r in rows if r["id"] in keep]
    return rows


def template_matches(tokenizer, rows: list[dict]) -> bool:
    """Does this model's own chat template (thinking off) reproduce the rendered prompts? Checked on the
    `knows` prompts, which are one user turn followed by the prefill."""
    for r in rows:
        if r["condition"] != "knows":
            continue
        user = r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        ours = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        if ours + PREFILL[r["lang"]] != r["prompt"]:
            return False
    return True


def provenance(model: str, setting: str, prompts: Path, rows: list[dict], tokenizer) -> dict:
    """What a run depends on: the pinned model, the prompt file, the code, the libraries."""
    import mlx.core as mx
    import mlx_lm
    import transformers

    from .models import PRETRAINED

    code = {str(f.relative_to(HERE)): sha256(f)[:16] for f in sorted(HERE.rglob("*.py")) if "gt" not in f.parts}
    code["pixi.lock"] = sha256(HERE.parents[1] / "pixi.lock")[:16]
    tokens = hashlib.sha256()
    for r in rows:
        tokens.update(np.array(tokenizer.encode(r["prompt"], add_special_tokens=False), np.int32).tobytes())
    return {
        "model": model,
        "repo": PRETRAINED[model][0],
        "revision": PRETRAINED[model][1],
        "setting": setting,
        "thinking": "off (empty <think> block in the rendered prompt)",
        "prompts": str(prompts),
        "prompts_sha256": sha256(prompts),
        "prompt_rows": len(rows),
        "prompt_tokens_sha256": tokens.hexdigest(),  # equal across models whose tokenizers agree on the prompts
        "template_matches": template_matches(tokenizer, rows),
        "code": code,
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _tag(r: dict) -> dict:
    return {"id": r["id"], "lang": r["lang"], "variant": r["variant"], "exemplar_set": r.get("exemplar_set")}


def short(model, tokenizer, rows: list[dict], batch: int = 64) -> list[dict]:
    rows = [r for r in rows if r["condition"] == "short"]
    assert all(r["max_new_tokens"] == 24 and r["decoding"] == "greedy" for r in rows)
    prompts = [tokenizer.encode(r["prompt"], add_special_tokens=False) for r in rows]
    done = generate(model, tokenizer, prompts, max_tokens=24, batch_size=batch)
    return [
        {
            **_tag(r),
            "text": c.text,
            "token_ids": c.tokens,
            "token_logprobs": c.logprobs,
            "finish_reason": "stop" if c.finished else "length",
        }
        for r, c in zip(rows, done)
    ]


def passes(rows: list[dict], condition: str) -> list[tuple[dict, bool, str]]:
    """(row, is_null, prompt) for every prompt to read: each row's own prompt, and its null prompt if any."""
    out = []
    for r in rows:
        if r["condition"] == condition:
            out.append((r, False, r["prompt"]))
            if r.get("null_prompt"):
                out.append((r, True, r["null_prompt"]))
    return out


def pass_key(r: dict, is_null: bool) -> str:
    return "|".join([r["id"], r["lang"], r["condition"], r["variant"], str(r.get("exemplar_set")), str(int(is_null))])


def shared_prefixes(todo: list[tuple[dict, bool, str]], tokens: list[list[int]]) -> dict:
    """Prompts of one relation, language and example set start with the same worked examples. For each
    such group: how many leading tokens all its prompts share (always leaving at least one to read)."""
    groups = {}
    for (r, _, _), p in zip(todo, tokens):
        key = _group(r)
        if key not in groups:
            groups[key] = p[:-1]
        else:
            g, n = groups[key], 0
            while n < min(len(g), len(p) - 1) and g[n] == p[n]:
                n += 1
            groups[key] = g[:n]
    return groups


def _group(r: dict) -> tuple:
    return (r["lang"], r["relation"], r.get("exemplar_set"))


def score(model, tokenizer, rows: list[dict], condition: str, base_full: np.ndarray | None, progress=print):
    """Scores per continuation, statistics per prompt, and the full next-token log-probabilities
    (float16, one row per prompt) at the answer position. `base_full`: the base model's, for the KL.

    Prompts are read in batches of one group (shared worked examples) and one length."""
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    todo = passes(rows, condition)
    tokens = [enc(prompt) for _, _, prompt in todo]
    prefixes = shared_prefixes(todo, tokens)
    alike = {}
    for j, (r, _, _) in enumerate(todo):
        alike.setdefault((_group(r), len(tokens[j])), []).append(j)
    batches = [(g, js[s : s + BATCH]) for (g, _), js in alike.items() for s in range(0, len(js), BATCH)]
    batches.sort(key=lambda b: str(b[0]))  # one group's shared cache at a time
    scores, stats, start = [None] * len(todo), [None] * len(todo), time.time()
    full, current, done = None, (None, None), 0
    for group, js in batches:
        if current[0] != group:
            shared = prefixes[group]
            current = (group, (len(shared), prefix_cache(model, shared)) if len(shared) >= 8 else None)
        conts = []
        for j in js:
            r, _, prompt = todo[j]
            conts.append([])
            for c in r["continuations"]:
                whole = enc(prompt + c["text"])
                assert whole[: len(tokens[j])] == tokens[j], f"prompt tokens are not a prefix: {r['id']} {c['text']!r}"
                conts[-1].append(whole[len(tokens[j]) :])
        lps, firsts = continuation_logps(model, [tokens[j] for j in js], conts, current[1])
        if full is None:
            full = np.zeros((len(todo), firsts.shape[1]), np.float16)
        for j, lp_j, first in zip(js, lps, firsts.astype(np.float64)):
            r, is_null, _ = todo[j]
            head = {**_tag(r), "condition": condition, "is_null": is_null}
            scores[j] = [
                {
                    **head,
                    **{k: c[k] for k in ("qid", "role", "form", "label") if k in c},
                    "logprob": float(lp.sum()),
                    "first_logprob": float(lp[0]),
                    "n_tokens": len(lp),
                    "token_logprobs": [round(float(x), 4) for x in lp],
                }
                for c, lp in zip(r["continuations"], lp_j)
            ]
            top = np.argsort(-first)[:TOP]
            stats[j] = {
                **head,
                "entropy": float(-(np.exp(first) * first).sum()),
                "top20": [{"token": tokenizer.decode([int(t)]), "id": int(t), "logprob": float(first[t])} for t in top],
            }
            if base_full is not None:  # KL(base || this setting), in nats
                b = base_full[j].astype(np.float64)
                b -= np.logaddexp.reduce(b)
                stats[j]["kl_from_base"] = float((np.exp(b) * (b - first)).sum())
            full[j] = first
        if (done + len(js)) // 500 > done // 500 or done + len(js) == len(todo):
            rate = 60 * (done + len(js)) / (time.time() - start)
            progress(f"  {condition}: {done + len(js)}/{len(todo)} prompts, {rate:.0f}/min")
        done += len(js)
    scores = [s for per_prompt in scores for s in per_prompt]
    return scores, stats, full, [pass_key(r, n) for r, n, _ in todo]


def knows_states(model, tokenizer, rows: list[dict], dropped=(), batch: int = 32) -> dict:
    """The residual stream at the last token of each `knows` prompt (the prefill's colon), every 4th block."""
    rows = [r for r in rows if r["condition"] == "knows"]
    seqs = [tokenizer.encode(r["prompt"], add_special_tokens=False) for r in rows]
    groups = [[tokenizer.encode(c["text"], add_special_tokens=False)[0]] for c in rows[0]["continuations"]]
    got = read(model, seqs, groups, dropped, batch, states=True)
    return {
        "ids": np.array([r["id"] for r in rows]),
        "lang": np.array([r["lang"] for r in rows]),
        "variant": np.array([r["variant"] for r in rows]),
        "layers": got["layers"],
        "H": got["H"],
        "lens_yes_no": got["lens"],  # English first tokens; kept only for the logit lens
    }


def _write_jsonl(path: Path, records: list[dict]):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    tmp.rename(path)  # a file that exists is complete


def run(model_name: str, setting: str, what: list[str], prompts: Path = PROMPTS, tag: str = "v1", n: int = 0):
    """Run the conditions in `what` that have no output yet."""
    from .compress import prune
    from .models import load, setting_dir

    rows = load_prompts(prompts, n)
    out = setting_dir(model_name, setting) / f"probs-{tag}{f'-n{n}' if n else ''}"
    base = setting_dir(model_name, "base") / out.name
    what = [w for w in what if not (out / FILES[w]).exists()]
    if not what:
        print(f"{model_name} / {setting}: nothing to do")
        return
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load(model_name, setting)
    dropped = prune.load_recipe(setting_dir(model_name, setting))["dropped"] if setting.startswith("shortgpt") else []
    record = provenance(model_name, setting, prompts, rows, tok)
    record["minutes"] = {}
    add = lambda recs: [{**r, "setting": setting} for r in recs]
    for w in what:
        t = time.time()
        if w == "short":
            _write_jsonl(out / FILES[w], add(short(model, tok, rows)))
        else:
            base_full = None
            if setting != "base" and (base / f"full-{w}.npy").exists():
                base_full = np.load(base / f"full-{w}.npy", mmap_mode="r")
                base_keys = json.loads((base / f"full-{w}.keys.json").read_text())
            scores, stats, full, keys = score(model, tok, rows, w, base_full)
            if base_full is not None:
                assert keys == base_keys, "the base model's vectors are for other prompts"
            if setting == "base":
                np.save(out / f"full-{w}.npy", full)
                (out / f"full-{w}.keys.json").write_text(json.dumps(keys))
            if w == "knows":
                np.savez(out / "knows-states.npz", **knows_states(model, tok, rows, dropped))
            _write_jsonl(out / f"position_stats-{w}.jsonl", add(stats))
            _write_jsonl(out / FILES[w], add(scores))
        record["minutes"][w] = round((time.time() - t) / 60, 2)
        print(f"{model_name} / {setting}: {w} in {record['minutes'][w]:.1f} min -> {out}", flush=True)
    (out / f"run-{'+'.join(what)}.json").write_text(json.dumps(record, indent=1) + "\n")
