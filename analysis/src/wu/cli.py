# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""wu: the command line.

wu data [--per-bucket 1000]                            freeze the PopQA subsample (once)
wu compress --model qwen3-1.7b --setting gptq-4      build a setting that needs calibration (gptq, awq, wanda, shortgpt)
wu ppl  --model qwen3-1.7b --setting rtn-4             WikiText-2 perplexity
wu eval --model qwen3-1.7b --setting base --mode short [--n 300]
wu pin gemma3-27b google/gemma-3-27b-it               add any Hugging Face model, pinned to today's revision
wu download --model gemma3-27b
wu sweep --model qwen3-8b --settings base,gptq-4,awq-4 [--modes short,chat-idk]   everything, skipping what exists
wu report --model qwen3-1.7b [--mode short]            RQ1/RQ2 tables for every evaluated setting
wu confidence --model qwen3-8b --setting gptq-3        facts and marbles: prediction, stated confidence, states
wu confidence-report --model qwen3-8b                  their measures, intervals and figures (CPU only)
wu probs --model qwen3-8b --setting gptq-3             epistemic-probabilities: short answers, scores, knows
wu probs-report --model qwen3-8b                       their measures, intervals and figures (CPU only)
"""

import argparse
import json
import math
import time
from pathlib import Path

from . import data
from .compress import prune
from .models import PRETRAINED, RUNS, apply_memory_policy, load, setting_dir


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=float) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def cmd_data(a):
    path = data.freeze_popqa(per_bucket=a.per_bucket)
    print(f"wrote {path} ({path.read_text().count(chr(10))} questions); manifest in {path.parent / 'MANIFEST.json'}")


def cmd_compress(a):
    from .models import compress

    apply_memory_policy()
    compress(a.model, a.setting)
    print(f"built {a.model} / {a.setting} in {setting_dir(a.model, a.setting)}")


def cmd_pin(a):
    from .models import pin

    print(f"{a.name}: {a.repo} @ {pin(a.name, a.repo)} (in models.json); fetch it with `wu download --model {a.name}`")


def cmd_download(a):
    from huggingface_hub import snapshot_download

    repo, revision = PRETRAINED[a.model]
    print(snapshot_download(repo, revision=revision))


def cmd_sweep(a):
    """compress (if needed) -> ppl (if missing) -> eval (if missing), each step in its own process, so one
    failure (or a GPU error) does not stop the rest; then the report."""
    import subprocess
    import sys

    from .models import PRUNE, QUANT, parse

    failed = []
    for setting in a.settings.split(","):
        out = setting_dir(a.model, setting)
        steps = []
        if parse(setting)[0] in QUANT + PRUNE and not (out / "built.json").exists():
            steps.append(["compress", "--setting", setting])
        if not (out / "ppl.json").exists():
            steps.append(["ppl", "--setting", setting])
        for mode in a.modes.split(","):
            if not (out / f"eval-{mode}-{a.n}.jsonl").exists():
                steps.append(["eval", "--setting", setting, "--mode", mode])
        for step in steps:
            print(f"[{a.model} {setting}] {step[0]}", flush=True)
            if subprocess.run([sys.executable, "-m", "wu.cli", step[0], "--model", a.model, *step[1:]]).returncode:
                failed.append(f"{setting}:{step[0]}")
                break
    for mode in a.modes.split(","):
        if (RUNS / a.model / "base" / f"eval-{mode}-{a.n}.jsonl").exists():
            subprocess.run(
                [sys.executable, "-m", "wu.cli", "report", "--model", a.model, "--mode", mode, "--n", str(a.n)]
            )
    print("failed: " + (", ".join(failed) if failed else "nothing"))


def cmd_ppl(a):
    from .ppl import PAPER_QWEN3_8B, perplexity

    apply_memory_policy()
    model, tok = load(a.model, a.setting)
    t = time.time()
    result = {"model": a.model, "setting": a.setting, **perplexity(model, tok), "seconds": round(time.time() - t, 1)}
    if a.model == "qwen3-8b" and a.setting in PAPER_QWEN3_8B:
        result["paper"] = PAPER_QWEN3_8B[a.setting]
    write_json(setting_dir(a.model, a.setting) / "ppl.json", result)
    print(json.dumps(result))


def cmd_eval(a):
    from .generate import ask

    apply_memory_policy()
    rows = data.popqa(a.data)
    if a.n:
        rows = rows[: a.n]  # rows are sorted by id, so buckets stay mixed
    tag = str(len(rows))
    if a.subset == "known+400":  # Stage 2: every fact the base model knew (can be lost or kept) + 400 it never knew
        import random

        base = {r["id"]: r["correct"] for r in read_jsonl(setting_dir(a.model, "base") / f"eval-short-{a.data}.jsonl")}
        known = [r for r in rows if base[r["id"]]]
        unknown = random.Random(0).sample([r for r in rows if not base[r["id"]]], 400)
        rows = sorted(known + unknown, key=lambda r: int(r["id"]))
        tag = "known+400"
    model, tok = load(a.model, a.setting)
    print(f"{a.model} / {a.setting} / {a.mode}: {len(rows)} questions")
    t = time.time()
    records = ask(model, tok, rows, a.mode, batch_size=a.batch_size)
    out = setting_dir(a.model, a.setting) / f"eval-{a.mode}-{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    minutes = (time.time() - t) / 60
    print(f"wrote {out}: {minutes:.1f} min, {len(rows) / minutes:.0f} questions/min")


def cmd_report(a):
    from .metrics import accuracy_table, compare

    root = RUNS / a.model
    name = f"eval-{a.mode}-{a.n}.jsonl"
    base_path = root / "base" / name
    base = read_jsonl(base_path)
    print(
        f"{a.model}, {a.mode}, n={a.n}. base accuracy: "
        + ", ".join(f"{k} {v:.3f}" for k, v in accuracy_table(base).items())
    )
    props = {r["id"]: r["prop"] for r in data.popqa(a.n)}
    print("base accuracy by relation (n per bucket head/middle/tail, then accuracy):")
    for prop in sorted({props[r["id"]] for r in base}):
        cells = []
        for b in data.BUCKETS:
            xs = [r["correct"] for r in base if props[r["id"]] == prop and r["bucket"] == b]
            cells.append(f"{len(xs):>4} {sum(xs) / len(xs) if xs else float('nan'):.2f}")
        print(f"  {prop:<16}" + "   ".join(cells))
    print(
        f"{'setting':<14}{'ppl':>9}{'acc':>7}{'r':>7}{'RS head':>9}{'RS mid':>8}{'RS tail':>9}"
        f"{'LR head':>9}{'LR tail':>9}{'conf lost':>11}"
    )
    results = {}
    for path in sorted(root.glob(f"*/{name}")):
        setting = path.parent.name
        if setting == "base":
            continue
        c = compare(base, read_jsonl(path))
        results[setting] = c
        ppl_path = path.parent / "ppl.json"
        ppl = json.loads(ppl_path.read_text())["ppl"] if ppl_path.exists() else math.nan
        print(
            f"{setting:<14}{ppl:>9.2f}{c['acc']['all']:>7.3f}{c['retention']['all']:>7.2f}"
            f"{c['rs_pp']['head']:>+9.1f}{c['rs_pp']['middle']:>+8.1f}{c['rs_pp']['tail']:>+9.1f}"
            f"{c['loss_rate']['head']:>9.2f}{c['loss_rate']['tail']:>9.2f}"
            f"{c['conf_lost']['all'].get('median', math.nan):>11.2f}"
        )
    write_json(root / f"report-{a.mode}-{a.n}.json", results)


def cmd_uncertainty(a):
    """Credible intervals for every evaluated setting, and how they narrow with more questions."""
    from .bayes import posterior, summarize, width_by_n

    root = RUNS / a.model
    name = f"eval-{a.mode}-{a.n}.jsonl"
    base = read_jsonl(root / "base" / name)
    keys = ("r", "RS_head", "RS_tail", "RS_tail-head", "LR_head", "LR_tail")
    print(f"{a.model}, {a.mode}, n={a.n}: posterior median [95% interval]; P(>0) for the tail-head difference")
    print(f"{'setting':<14}" + "".join(f"{k:>24}" for k in keys) + f"{'P(tail>head)':>14}")
    results = {}
    for path in sorted(root.glob(f"*/{name}")):
        if path.parent.name == "base":
            continue
        comp = read_jsonl(path)
        d = posterior(base, comp)
        s = {k: summarize(d[k]) for k in keys}
        results[path.parent.name] = s
        cell = lambda v: (
            f"{v['median']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"
            if abs(v["median"]) < 2
            else f"{v['median']:.1f} [{v['lo']:.1f}, {v['hi']:.1f}]"
        )
        print(
            f"{path.parent.name:<14}"
            + "".join(f"{cell(s[k]):>24}" for k in keys)
            + f"{s['RS_tail-head']['p_positive']:>14.2f}"
        )
    if a.curve:
        comp = read_jsonl(root / a.curve / name)
        curve = width_by_n(base, comp, "RS_tail-head")
        print(
            f"width of the 95% interval of RS_tail - RS_head ({a.curve}), by questions per group: "
            + ", ".join(f"n={n}: {w:.1f} pp" for n, w in curve.items())
        )
        results["_curve"] = {a.curve: curve}
    write_json(root / f"uncertainty-{a.mode}-{a.n}.json", results)


def cmd_ppl_spread(a):
    """Perplexity as an estimate: its spread over which 8,192 tokens are read, unpaired and paired."""
    from .bayes import perplexity_spread

    root = RUNS / a.model
    base = json.loads((root / "base" / "ppl-windows.json").read_text())
    s = perplexity_spread(base, None)
    lo, med, hi = s["base_subsets"]
    print(
        f"{a.model} base: all {s['windows']} windows {s['base_all']:.2f}; first 4 (paper protocol) {s['base_first_k']:.2f}; "
        f"random 4-window subsets {med:.2f} [{lo:.2f}, {hi:.2f}]"
    )
    print(
        f"{'setting':<14}{'ppl, 4 windows: 95% range':>30}{'ratio, all':>12}{'ratio, first 4':>16}{'ratio, 4 windows: 95% range':>32}"
    )
    results = {"base": s}
    for path in sorted(root.glob("*/ppl-windows.json")):
        if path.parent.name == "base":
            continue
        c = perplexity_spread(base, json.loads(path.read_text()))
        results[path.parent.name] = c
        print(
            f"{path.parent.name:<14}{'{:.2f} [{:.2f}, {:.2f}]'.format(c['comp_subsets'][1], c['comp_subsets'][0], c['comp_subsets'][2]):>30}"
            f"{c['ratio_all']:>12.3f}{c['ratio_first_k']:>16.3f}{'{:.3f} [{:.3f}, {:.3f}]'.format(c['ratio_subsets'][1], c['ratio_subsets'][0], c['ratio_subsets'][2]):>32}"
        )
    write_json(root / "ppl-spread.json", results)


def cmd_represent(a):
    """Hidden-state drift, a 'do I know this?' probe, and the model's self-report, base vs a compressed setting."""
    import mlx.core as mx
    import numpy as np

    from .represent import analyse, states

    if a.cpu:
        mx.set_default_device(mx.cpu)
    else:
        apply_memory_policy()
    rows = data.popqa(3000)[: a.n] if a.n else data.popqa(3000)
    n = len(rows)
    got = {}
    for setting in ("base", a.setting):
        path = setting_dir(a.model, setting) / f"states-{n}.npz"
        if not path.exists():
            m, tok = load(a.model, setting)
            dropped = (
                prune.load_recipe(setting_dir(a.model, setting))["dropped"] if setting.startswith("shortgpt") else []
            )
            st = states(m, tok, rows, dropped=dropped)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **st)
            del m
            print(f"recorded {setting}: layers {st['layers'].tolist()}", flush=True)
        got[setting] = dict(np.load(path))
    evals = {s: read_jsonl(setting_dir(a.model, s) / "eval-short-3000.jsonl")[:n] for s in ("base", a.setting)}
    r = analyse(got["base"], got[a.setting], evals["base"], evals[a.setting])
    write_json(setting_dir(a.model, a.setting) / f"represent-{n}.json", r)
    print(
        f"{a.model} {a.setting}, n={n}. drift = median cosine distance base->compressed; probe trained on base states (cross-fitted)"
    )
    print(
        f"{'layer':>5}{'drift kept':>11}{'drift lost':>11}{'kNN kept':>9}{'AUC base':>9}{'AUC comp':>9}{'lost vs kept':>13}"
    )
    for L in r["layers"]:
        d = L["drift"]
        print(
            f"{L['layer']:>5}{d.get('kept', float('nan')):>11.3f}{d.get('lost', float('nan')):>11.3f}{L['knn_overlap']:>9.2f}"
            f"{L['probe_auc_base']:>9.2f}{L['probe_auc_comp_same_probe']:>9.2f}{L['probe_on_lost_vs_kept']:>13.2f}"
        )
    sr = r["self_report"]
    print(f"self-report P(Yes) AUC for correctness: base {sr['auc_base']:.2f}, compressed {sr['auc_comp']:.2f}")
    print(
        "mean P(Yes), compressed model, by outcome: "
        + ", ".join(f"{k} {v:.2f}" for k, v in sr["mean_p_yes_comp"].items())
    )
    print(
        "mean P(Yes), base model,       by outcome: "
        + ", ".join(f"{k} {v:.2f}" for k, v in sr["mean_p_yes_base"].items())
    )


def cmd_confidence(a):
    """Prediction and stated confidence with the truth known (wu.confidence): PopQA facts, and marbles
    (wu.urn). One .npz per setting and experiment; existing files are skipped."""
    import numpy as np

    from . import confidence, urn

    apply_memory_policy()
    out = setting_dir(a.model, a.setting)
    todo = {}
    if "facts" in a.what:
        rows = data.popqa(3000)[: a.n] if a.n else data.popqa(3000)
        base = read_jsonl(setting_dir(a.model, "base") / "eval-short-3000.jsonl")[: len(rows)]
        todo[f"confidence-facts-{len(rows)}.npz"] = lambda m, tok, d: confidence.facts(
            m, tok, rows, read_jsonl(out / "eval-short-3000.jsonl")[: len(rows)], base, d, a.batch_size
        )
    if "marbles" in a.what:
        items = urn.marbles()
        items = items[:: max(1, len(items) // a.n)] if a.n else items  # a spread over kinds, for smoke runs
        todo[f"confidence-marbles-{len(items)}.npz"] = lambda m, tok, d: confidence.marbles(
            m, tok, items, d, a.batch_size
        )
    todo = {name: run for name, run in todo.items() if not (out / name).exists()}
    if not todo:
        print(f"{a.model} / {a.setting}: nothing to do")
        return
    model, tok = load(a.model, a.setting)
    dropped = prune.load_recipe(out)["dropped"] if a.setting.startswith("shortgpt") else []
    for name, run in todo.items():
        t = time.time()
        result = run(model, tok, dropped)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / name, **result)
        print(f"{a.model} / {a.setting}: wrote {name} in {(time.time() - t) / 60:.1f} min", flush=True)


def cmd_confidence_report(a):
    from .confidence_report import redraw, run

    if a.figures_only:
        redraw(a.model)
    else:
        run(a.model, inside=not a.no_inside)


def cmd_probs(a):
    """The epistemic-probabilities prompts on one setting (wu.probs); existing outputs are skipped."""
    from . import probs

    apply_memory_policy()
    probs.run(a.model, a.setting, a.what.split(","), Path(a.prompts), a.tag, a.n)


def cmd_probs_report(a):
    from .probs_report import redraw, run

    if a.figures_only:
        redraw(a.model, a.tag)
    else:
        run(a.model, a.tag)


def cmd_splice(a):
    """Recovery of lost facts when only a prefix or a suffix of the blocks is compressed."""
    from .generate import ask
    from .splice import hybrid, questions

    apply_memory_policy()
    rows = data.popqa(3000)
    base = read_jsonl(setting_dir(a.model, "base") / "eval-short-3000.jsonl")
    comp = read_jsonl(setting_dir(a.model, a.setting) / "eval-short-3000.jsonl")
    qs, labels = questions(base, comp, rows, a.per_group)

    n = len(load(a.model, "base")[0].layers)
    bounds = sorted({*range(0, n + 1, a.step), n})
    out = {"labels": labels, "ids": [q["id"] for q in qs], "prefix": {}, "suffix": {}}
    for kind in ("prefix", "suffix"):
        for L in bounds:
            lo, hi = (0, L) if kind == "prefix" else (L, n)
            m, tok = hybrid(a.model, a.setting, lo, hi)
            rec = ask(m, tok, qs, "short", progress=lambda *_: None)
            acc = {
                g: sum(r["correct"] for r, l in zip(rec, labels) if l == g) / max(1, labels.count(g))
                for g in ("lost", "kept")
            }
            out[kind][L] = {**acc, "correct": [r["correct"] for r in rec]}
            print(
                f"{kind:<7} compressed blocks [{lo:>2},{hi:>2}): lost recovered {acc['lost']:.2f}  kept still right {acc['kept']:.2f}",
                flush=True,
            )
            del m
    write_json(setting_dir(a.model, a.setting) / f"splice-{a.per_group}.json", out)


def cmd_lens(a):
    """Logit lens on lost and kept facts: log p(first token of the base model's answer) per layer, base vs compressed."""
    import numpy as np

    from .compress import prune
    from .lens import lens
    from .splice import questions

    apply_memory_policy()
    rows = data.popqa(3000)
    base = read_jsonl(setting_dir(a.model, "base") / "eval-short-3000.jsonl")
    comp = read_jsonl(setting_dir(a.model, a.setting) / "eval-short-3000.jsonl")
    qs, labels = questions(base, comp, rows, a.per_group)
    by_id = {r["id"]: r for r in base}
    m0, tok = load(a.model, "base")
    targets = [tok.encode(by_id[q["id"]]["response"].strip(), add_special_tokens=False)[0] for q in qs]
    res = {"base": lens(m0, tok, qs, targets)}
    n = len(m0.layers)
    del m0
    mc, _ = load(a.model, a.setting)
    r = lens(mc, tok, qs, targets)
    if len(mc.layers) < n:  # ShortGPT: put kept blocks back at their original depth, removed ones as NaN
        dropped = set(prune.load_recipe(setting_dir(a.model, a.setting))["dropped"])
        kept = [i for i in range(n) if i not in dropped]
        full = {k: np.full((len(qs), n), np.nan) for k in r}
        for j, i in enumerate(kept):
            for k in r:
                full[k][:, i] = r[k][:, j]
        r = full
    res["comp"] = r
    lab = np.array(labels)
    np.savez_compressed(
        setting_dir(a.model, a.setting) / f"lens-{a.per_group}.npz",
        labels=lab,
        **{f"{w}_{k}": v for w in res for k, v in res[w].items()},
    )
    print(f"{a.model} {a.setting}: median log p(answer's first token) at the last prompt token, by layer")
    print(f"{'layer':>5}{'base lost':>11}{'comp lost':>11}{'base kept':>11}{'comp kept':>11}")
    for i in range(0, n, max(1, n // 12)):
        f = lambda w, g: float(np.nanmedian(res[w]["logp"][lab == g, i]))
        print(
            f"{i:>5}{f('base', 'lost'):>11.2f}{f('comp', 'lost'):>11.2f}{f('base', 'kept'):>11.2f}{f('comp', 'kept'):>11.2f}"
        )


def cmd_wrongtype(a):
    """When the model is wrong, is its answer the right *kind* of thing (a valid answer to the same relation for
    another question)? Text-level proxy for 'the relation's subspace survived, the entity did not'."""
    from .match import normalize

    pool = {}
    for r in data.popqa_all():
        pool.setdefault(r["prop"], set()).update(normalize(x) for x in r["answers"] if normalize(x))
    rows = {r["id"]: r for r in data.popqa(3000)}

    def kind(rec):
        resp = normalize(rec["response"])
        if not resp:
            return "empty"
        prop = rows[rec["id"]]["prop"]
        padded = f" {resp} "
        return "same-type" if any(f" {g} " in padded for g in pool[prop] if len(g) > 2) else "other"

    print(f"{'model/setting':<26}{'n lost':>7}{'lost: same-type':>16}{'never-known: same-type':>23}{'ratio':>7}")
    for model in a.models.split(","):
        base = read_jsonl(setting_dir(model, "base") / "eval-short-3000.jsonl")
        for path in sorted((RUNS / model).glob("*/eval-short-3000.jsonl")):
            comp = read_jsonl(path)
            lost = [c for b, c in zip(base, comp) if b["correct"] and not c["correct"]]
            never = [c for b, c in zip(base, comp) if not b["correct"] and not c["correct"]]
            if not lost:
                continue
            fl = sum(kind(c) == "same-type" for c in lost) / len(lost)
            fn = sum(kind(c) == "same-type" for c in never) / len(never)
            print(
                f"{model + '/' + path.parent.name:<26}{len(lost):>7}{fl:>16.2f}{fn:>23.2f}{fl / fn if fn else float('nan'):>7.2f}"
            )


def cmd_geometry(a):
    """One picture per method: UMAP fitted on the base model's states, compressed states mapped into the same map."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    from .represent import outcome

    base_st = np.load(setting_dir(a.model, "base") / f"states-{a.n}.npz")
    j = list(base_st["layers"]).index(a.layer) if a.layer >= 0 else len(base_st["layers"]) - 1
    layer = int(base_st["layers"][j])
    X0 = base_st["H"][:, j].astype(np.float32)
    X0n = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="euclidean", random_state=0).fit(X0n)
    E0 = reducer.embedding_
    base = read_jsonl(setting_dir(a.model, "base") / f"eval-short-{a.n}.jsonl")
    settings = [s for s in a.settings.split(",") if (setting_dir(a.model, s) / f"states-{a.n}.npz").exists()]
    fig, axes = plt.subplots(1, len(settings), figsize=(4.2 * len(settings), 4.4), squeeze=False)
    for ax, s in zip(axes[0], settings):
        Xc = np.load(setting_dir(a.model, s) / f"states-{a.n}.npz")["H"][:, j].astype(np.float32)
        Ec = reducer.transform(Xc / np.linalg.norm(Xc, axis=1, keepdims=True))
        cat = outcome(base, read_jsonl(setting_dir(a.model, s) / f"eval-short-{a.n}.jsonl"))
        ax.scatter(E0[:, 0], E0[:, 1], s=2, c="#cfcfcf", lw=0)
        for k, col in ((1, "#111111"), (2, "#d9480f")):
            idx = np.where(cat == k)[0][:120]
            for i in idx:
                ax.plot([E0[i, 0], Ec[i, 0]], [E0[i, 1], Ec[i, 1]], color=col, lw=0.5, alpha=0.8)
            ax.scatter(Ec[idx, 0], Ec[idx, 1], s=5, c=col, lw=0)
        props = [r["prop"] for r in data.popqa(a.n)]
        for prop in sorted(set(props)):  # name each island by the relation type of its questions
            idx = [i for i, q in enumerate(props) if q == prop]
            if len(idx) >= 40:
                cx, cy = np.median(E0[idx, 0]), np.median(E0[idx, 1])
                ax.text(
                    cx,
                    cy,
                    prop,
                    fontsize=7,
                    color="#555555",
                    ha="center",
                    va="center",
                    path_effects=[
                        __import__("matplotlib.patheffects", fromlist=["x"]).withStroke(
                            linewidth=2.5, foreground="white"
                        )
                    ],
                )
        ax.set_title(f"{s}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(
        f"{a.model}, layer {layer}: UMAP of the base model's states (grey). Lines: where each question's state "
        f"moves under compression (black: kept facts, orange: lost facts; 120 each)",
        fontsize=9,
    )
    out = RUNS / a.model / f"geometry-layer{layer}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(prog="wu")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--per-bucket", type=int, default=1000)
    pn = sub.add_parser("pin", help="add a Hugging Face model to models.json at its current revision")
    pn.add_argument("name")
    pn.add_argument("repo")
    sw = sub.add_parser("sweep", help="build, score and evaluate a list of settings, skipping what exists")
    sw.add_argument("--model", required=True)
    sw.add_argument("--settings", required=True, help="comma-separated, e.g. base,gptq-4,awq-4,wanda-50,shortgpt-10")
    sw.add_argument("--modes", default="short")
    sw.add_argument("--n", type=int, default=3000)
    un = sub.add_parser("uncertainty", help="credible intervals for RQ1/RQ2 metrics (paired Dirichlet posteriors)")
    un.add_argument("--model", required=True)
    un.add_argument("--mode", default="short")
    un.add_argument("--n", type=int, default=3000)
    un.add_argument("--curve", help="a setting for which to show how the interval narrows with n")
    ps = sub.add_parser(
        "ppl-spread", help="how much perplexity depends on which tokens are read (needs ppl-windows.json)"
    )
    ps.add_argument("--model", required=True)
    rp = sub.add_parser("represent", help="hidden-state drift, a P(IK) probe and self-report, base vs a setting")
    rp.add_argument("--model", required=True)
    rp.add_argument("--setting", required=True)
    rp.add_argument("--n", type=int, default=0, help="first n questions (0: all 3,000)")
    rp.add_argument("--cpu", action="store_true", help="run on the CPU (leave the GPU to a long build)")
    cf = sub.add_parser("confidence", help="facts and marbles: prediction, stated confidence, hidden states")
    cf.add_argument("--model", required=True)
    cf.add_argument("--setting", default="base")
    cf.add_argument("--what", default="facts,marbles")
    cf.add_argument("--n", type=int, default=0, help="smoke runs: the first n questions, about n marble items")
    cf.add_argument("--batch-size", type=int, default=32)
    cr = sub.add_parser("confidence-report", help="measures, intervals and figures from `wu confidence` (CPU only)")
    cr.add_argument("--model", required=True)
    cr.add_argument("--no-inside", action="store_true", help="skip the probes on hidden states")
    cr.add_argument("--figures-only", action="store_true", help="redraw from confidence-report.json")
    from .probs import PROMPTS

    pb = sub.add_parser("probs", help="epistemic-probabilities prompts: short answers, candidate scores, knows")
    pb.add_argument("--model", required=True)
    pb.add_argument("--setting", default="base")
    pb.add_argument("--what", default="short,scoring,knows")
    pb.add_argument("--prompts", default=str(PROMPTS), help="prompts_rendered.jsonl")
    pb.add_argument("--tag", default="v1", help="the version of the prompt file; names the output folder")
    pb.add_argument("--n", type=int, default=0, help="smoke runs: about n facts and controls")
    pr = sub.add_parser("probs-report", help="measures, intervals and figures from `wu probs` (CPU only)")
    pr.add_argument("--model", required=True)
    pr.add_argument("--tag", default="v1")
    pr.add_argument("--figures-only", action="store_true", help="redraw from probs-report.json")
    sp = sub.add_parser("splice", help="recovery of lost facts with only a prefix / suffix of blocks compressed")
    sp.add_argument("--model", required=True)
    sp.add_argument("--setting", required=True)
    sp.add_argument("--per-group", type=int, default=150)
    sp.add_argument("--step", type=int, default=4)
    ln = sub.add_parser("lens", help="logit lens on lost and kept facts, base vs compressed")
    ln.add_argument("--model", required=True)
    ln.add_argument("--setting", required=True)
    ln.add_argument("--per-group", type=int, default=150)
    wt = sub.add_parser("wrongtype", help="are wrong answers the right kind of thing? (CPU only)")
    wt.add_argument("--models", default="qwen3-1.7b,qwen3-8b")
    ge = sub.add_parser("geometry", help="UMAP of base states, compressed states mapped in (CPU only)")
    ge.add_argument("--model", required=True)
    ge.add_argument("--settings", required=True)
    ge.add_argument("--n", type=int, default=3000)
    ge.add_argument("--layer", type=int, default=-1)
    for name in ("download", "compress", "ppl", "eval", "report"):
        s = sub.add_parser(name)
        s.add_argument("--model", choices=PRETRAINED, default="qwen3-1.7b")
        if name == "download":
            continue
        if name != "report":
            s.add_argument("--setting", default="base")
        if name in ("eval", "report"):
            s.add_argument("--mode", choices=("short", "chat", "chat-idk"), default="short")
        if name == "eval":
            s.add_argument("--data", type=int, default=3000, help="which frozen subsample")
            s.add_argument("--n", type=int, default=0, help="only the first n questions (0: all)")
            s.add_argument("--batch-size", type=int, default=64)
            s.add_argument(
                "--subset",
                choices=("all", "known+400"),
                default="all",
                help="known+400: facts the base model got right in short mode, plus 400 it got wrong",
            )
        if name == "report":
            s.add_argument("--n", type=int, default=3000)
    a = p.parse_args()
    {
        "data": cmd_data,
        "pin": cmd_pin,
        "download": cmd_download,
        "sweep": cmd_sweep,
        "uncertainty": cmd_uncertainty,
        "ppl-spread": cmd_ppl_spread,
        "represent": cmd_represent,
        "confidence": cmd_confidence,
        "confidence-report": cmd_confidence_report,
        "probs": cmd_probs,
        "probs-report": cmd_probs_report,
        "splice": cmd_splice,
        "lens": cmd_lens,
        "wrongtype": cmd_wrongtype,
        "geometry": cmd_geometry,
        "compress": cmd_compress,
        "ppl": cmd_ppl,
        "eval": cmd_eval,
        "report": cmd_report,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
