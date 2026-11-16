# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""P1: gather what `wu probs` wrote for every setting into the three files the epistemic-probabilities
README asks for, in <RUNS>/<model>/probs-<tag>-export/:

    short.jsonl            {id, lang, setting, variant, exemplar_set, text, token_ids, token_logprobs, finish_reason}
    scores.jsonl           scoring and knows rows, each with `setting`, `condition`, `is_null`
    position_stats.jsonl   entropy, top 20 tokens and KL from the base model at the answer position
    runs.json              per setting: model revision, prompt-file hash, code hashes, minutes

    pixi run probs-export --model qwen3-8b [--tag v1]
"""

import argparse
import json
import shutil

from wu.confidence_report import SETTINGS
from wu.models import RUNS, setting_dir

PARTS = {
    "short.jsonl": ["short.jsonl"],
    "scores.jsonl": ["scores-scoring.jsonl", "scores-knows.jsonl"],
    "position_stats.jsonl": ["position_stats-scoring.jsonl", "position_stats-knows.jsonl"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--tag", default="v1")
    a = p.parse_args()
    out = RUNS / a.model / f"probs-{a.tag}-export"
    out.mkdir(parents=True, exist_ok=True)
    done, runs = [], {}
    needed = [n for names in PARTS.values() for n in names]
    for setting in ["base", *SETTINGS]:
        d = setting_dir(a.model, setting) / f"probs-{a.tag}"
        if all((d / n).exists() for n in needed):
            done.append(setting)
            runs[setting] = [json.loads(f.read_text()) for f in sorted(d.glob("run-*.json"))]
    for name, parts in PARTS.items():
        with open(out / name, "wb") as f:
            for setting in done:
                for part in parts:
                    with open(setting_dir(a.model, setting) / f"probs-{a.tag}" / part, "rb") as g:
                        shutil.copyfileobj(g, f)
        print(f"{out / name}: {(out / name).stat().st_size / 1e6:.0f} MB")
    (out / "runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    print(f"{len(done)} settings: {', '.join(done)}")


if __name__ == "__main__":
    main()
