# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""P1: the results as files that can be read without the run directories, in --out
(default <RUNS>/probs-<tag>-results/).

    results-wide.csv    one row per model, setting, language; one column per main measure (estimates only)
    results-long.csv    one row per model, setting, language, measure (and popularity band): est, lo, hi
    per-question.csv    one row per model, setting and question as asked (fact or invented subject, language,
                        wording): the model's short answer, whether it is correct, the answer margin, P(Yes)
    <model>/            the report JSON, the figures, the short answers of every setting (short.jsonl),
                        what was run with which code and prompt file (runs.json), and the export note

    <model>/raw/<setting>/   what the epistemic-probabilities README asks to be sent back, gzipped, one
                        folder per setting: scores.jsonl.gz (the `scoring` rows, then the `knows` rows) and
                        position_stats.jsonl.gz (entropy, top 20 tokens, KL from the base model)

Not copied: the base model's full next-token distributions (full-*.npy, about 2 GB per model) and the
hidden states for the probe (knows-states.npz); they stay in <RUNS>/<model>/<setting>/probs-<tag>/.

    pixi run probs-results [--tag v1] [--out DIR]
"""

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

from wu.confidence_report import LABELS, SETTINGS
from wu.models import RUNS, setting_dir
from wu.probs_report import load_setting, metadata

MAIN = [
    "n", "n_lost", "n_kept", "short_accuracy", "first", "retention", "margin_for_correct", "knows_for_correct",
    "knows_for_first", "real", "p_yes", "p_yes_controls", "lean", "d_lean", "spread", "knows_order",
    "kept_vs_lost_base", "kept_vs_lost", "d_kept_vs_lost", "tracking", "p_yes_lost", "p_yes_kept", "yes_no_mass",
]  # fmt: skip
QUESTION = [
    "model", "setting", "label", "id", "language", "wording", "invented_subject", "relation", "popularity_band",
    "correct_answers", "short_answer", "short_correct", "short_finished", "answer_margin", "knows_fact",
    "answer_margin_null_corrected", "share_of_wrong_answers_beaten", "entropy_at_answer", "kl_from_base",
    "yes_minus_no_log_odds", "p_yes", "yes_plus_no_mass",
]  # fmt: skip


def summary(model: str, report: dict):
    """Rows of results-long.csv for one model: every number in its report."""

    def walk(d: dict, lead: list, prefix: str = "", band: str = ""):
        for k, v in d.items():
            if k == "by_band":
                for b, inner in v.items():
                    yield from walk(inner, lead, band=b)
            elif isinstance(v, dict) and "est" in v:
                yield [*lead, band, prefix + k, v["est"], v.get("lo"), v.get("hi")]
            elif isinstance(v, dict):
                yield from walk(v, lead, prefix + k + ".", band)
            elif isinstance(v, int | float):
                yield [*lead, band, prefix + k, v, None, None]

    for setting, entry in report["settings"].items():
        for lang in ("en", "ru"):
            yield from walk(entry[lang], [model, setting, entry["label"], lang])
        # English only: the facts that are also in PopQA-3000
        yield from walk(entry.get("shared_with_l1", {}), [model, setting, entry["label"], "en"], "shared_with_l1.")


def per_question(model: str, tag: str):
    """Rows of per-question.csv for one model."""
    meta = metadata()
    base = load_setting(model, "base", tag, meta)
    for setting in ["base", *SETTINGS]:
        c = load_setting(model, setting, tag, meta, base["units"]) if base else None
        if c is None:
            continue
        for j, (i, lang, wording) in enumerate(c["units"]):
            m = meta[i]
            yield [
                model, setting, LABELS[setting], i, lang, wording, 0, m["relation"], m["band"], " | ".join(m["gold"][lang]),
                c["short_text"][j], int(c["short_correct"][j]), int(c["short_finished"][j]), c["margin"][j],
                int(c["first"][j]), c["margin_null"][j], c["beaten"][j], c["entropy"][j], c["kl"][j], c["knows"][j],
                c["p_yes"][j], c["mass"][j],
            ]  # fmt: skip
        for k, (i, lang, wording) in enumerate(c["controls"]):
            yield [
                model, setting, LABELS[setting], i, lang, wording, 1, "", "", "", c["c_short"][k], "", "", "", "", "",
                "", "", "", c["c_knows"][k], c["c_p_yes"][k], "",
            ]  # fmt: skip


RAW = {
    "scores.jsonl.gz": ["scores-scoring.jsonl", "scores-knows.jsonl"],
    "position_stats.jsonl.gz": ["position_stats-scoring.jsonl", "position_stats-knows.jsonl"],
}


def raw(model: str, tag: str, out: Path) -> int:
    """Gzip each setting's score files into out/<setting>/. Returns the bytes written."""
    size = 0
    for setting in ["base", *SETTINGS]:
        d = setting_dir(model, setting) / f"probs-{tag}"
        if not all((d / part).exists() for parts in RAW.values() for part in parts):
            continue
        (out / d.parent.name).mkdir(parents=True, exist_ok=True)
        for name, parts in RAW.items():
            target = out / d.parent.name / name
            with open(target, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as z:  # mtime 0: same bytes each run
                for part in parts:
                    with open(d / part, "rb") as g:
                        shutil.copyfileobj(g, z)
            size += target.stat().st_size
    return size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or RUNS / f"probs-{a.tag}-results"
    out.mkdir(parents=True, exist_ok=True)
    reports = sorted(RUNS.glob(f"*/probs-{a.tag}-report.json"))

    long = [row for path in reports for row in summary(path.parent.name, json.loads(path.read_text()))]
    with open(out / "results-long.csv", "w", newline="") as f:
        csv.writer(f).writerows(
            [["model", "setting", "label", "language", "band", "measure", "est", "lo", "hi"], *long]
        )
    wide = {}
    for model, setting, label, lang, band, measure, est, _, _ in long:
        if not band and measure in MAIN:
            wide.setdefault((model, setting, label, lang), {})[measure] = est
    with open(out / "results-wide.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "setting", "label", "language", *MAIN])
        for key, m in wide.items():
            w.writerow([*key, *(m.get(k, "") for k in MAIN)])

    n = 0
    with open(out / "per-question.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(QUESTION)
        for path in reports:
            model = path.parent.name
            for row in per_question(model, a.tag):
                w.writerow(row)
                n += 1
            d = out / model
            d.mkdir(exist_ok=True)
            shutil.copy(path, d / path.name)
            smoke = f"probs-{a.tag}-n"  # figures of the small trial runs, e.g. probs-v1-n40-*
            for fig in sorted(path.parent.glob(f"probs-{a.tag}-*.png")):
                if not fig.name.startswith(smoke):
                    shutil.copy(fig, d / fig.name)
            for name in ("short.jsonl", "runs.json", "NOTE.md"):
                if (path.parent / f"probs-{a.tag}-export" / name).exists():
                    shutil.copy(path.parent / f"probs-{a.tag}-export" / name, d / name)
            print(f"{d}/raw: {raw(model, a.tag, d / 'raw') / 1e6:.0f} MB")
    print(f"{out}: results-wide.csv {len(wide)} rows, results-long.csv {len(long)} rows, per-question.csv {n} rows")


if __name__ == "__main__":
    main()
