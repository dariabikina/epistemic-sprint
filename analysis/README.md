# Running the prompts on compressed Qwen3 (MLX)

Runs of `../prompts_rendered.jsonl` on Qwen3-8B (13 settings) and Qwen3-4B (7 settings), on
Apple silicon with MLX, 15 Nov 2026, prompt file v1. This folder holds the plan, the code, the pinned
environment and the results, including the files the top-level README asks to be sent back. Model
weights are not here (see "What is not here").

Nothing above this folder was changed.

## What is here

| path | what |
|---|---|
| `AGENTS.md` | for an agent picking this up: what we need back (v2), how to read the files, what to watch for |
| `PLAN.md` | design, decisions, predictions written before the runs, and the results against them |
| `PLAN-approved.md` | the plan as approved before any code was written |
| `results/v1/` | the results (below) |
| `src/wu/` | the code. For this experiment: `probs.py` (runs the prompts), `probs_report.py` (measures, intervals, figures), `probs_poster.py` (plain-language figures, worked example). The rest of the package is what they need: model loading and compression (`models.py`, `compress/`), generation, scoring, figure helpers. |
| `scripts/` | `probs_queue.sh` (runs all settings), `probs_export.py`, `probs_results.py`; the other scripts belong to earlier experiments |
| `tests/` | `test_probs.py` for this experiment; the others cover the rest of the package |
| `pyproject.toml`, `pixi.lock` | pinned environment and the pixi tasks |
| `data/`, `models.json`, `ruff.toml` | small frozen files and settings the package expects |

## Results (`results/v1/`)

| file | contents |
|---|---|
| `results-wide.csv` | one row per model, setting and language; one column per main measure (estimates). Start here. |
| `results-long.csv` | every measure with its 95% interval (`est`, `lo`, `hi`), also by popularity band |
| `per-question.csv` | one row per model, setting and question as asked (fact or invented subject, language, wording): the short answer, whether it is correct, the answer margin, whether the fact counts as known, the Yes-minus-No log-odds and P(Yes) |
| `EXAMPLE.md` | one question taken through every step for the uncompressed 8B model, with its real numbers |
| `qwen3-8b/raw/<setting>/`, `qwen3-4b/raw/<setting>/` | **the files asked for:** `scores.jsonl.gz` (`scoring` rows, then `knows` rows) and `position_stats.jsonl.gz`, one folder per setting, every row with `setting`. `wanda-2of4` is WANDA 2:4. |
| `qwen3-8b/`, `qwen3-4b/` | `short.jsonl` (the third file asked for), `probs-v1-report.json` (source of the two summary CSVs), the figures, `runs.json` (what was run, with model revision, prompt-file hash and code hashes), `NOTE.md` (answers to the README's "please confirm" items, caveats) |

Figures: `probs-v1-plain-*.png` are the plain-language ones, English and Russian side by side;
`probs-v1-{coupling,lost,tracking}-{en,ru}.png` are the detailed ones.

The measure names are defined in the docstring of `src/wu/probs_report.py`. Intervals are 95% bootstrap
over facts (a fact's wordings are resampled together), 1,000 draws, seed 0; differences from the
uncompressed model are paired.

## What the results say, in short

Numbers are for Qwen3-8B; Qwen3-4B agrees. Details and limits are in `PLAN.md`.

- **Facts are lost.** A model knows a fact if it gives the correct answer a higher probability than
  every wrong candidate. Compression removes 3–40% of the facts the model knew in English and 7–50%
  in Russian.
- **The answer probabilities still show what the model gets right.** The margin of the correct answer
  over the best wrong candidate separates correctly from wrongly answered questions at AUROC 0.84–0.88
  in English, with or without compression. The `knows` log-odds reach 0.68–0.75.
- **`knows` does not pick out the lost facts.** It separates the facts a compressed model kept from
  those it lost no better than the uncompressed model's `knows` did before the loss (paired
  difference includes zero in 19 of 23 cases on 8B and 12 of 12 on 4B).
- **The level of P(Yes) moves as a whole.** Each compression method shifts it by several nats for real
  and invented subjects alike, in a direction that differs between methods, languages and model
  sizes. Compare within a setting, not mean P(Yes) across settings.

## Reproducing

Needs an Apple-silicon Mac (48 GB was enough for 8B), [pixi](https://pixi.sh), and access to Hugging
Face for the base weights and the calibration text.

```sh
cd analysis
pixi install                 # the environment in pixi.lock
pixi run test                # all tests
```

Run directories go to `~/models/wu-runs` (set `WU_RUNS` to move them). The prompt file is read from
`../prompts_rendered.jsonl`.

```sh
# 1. base weights (pinned revisions) and the compressed settings; built once, then cached
pixi run wu download --model qwen3-8b
pixi run wu compress --model qwen3-8b --setting gptq-3      # likewise gptq-4, awq-3, awq-4, wanda-30,
                                                            # wanda-50, wanda-2:4, shortgpt-5, shortgpt-10
                                                            # (rtn-* and magnitude-30 need no build)
# 2. a small trial, then everything: short answers, candidate scores, `knows`
pixi run probs-smoke                                        # about 40 facts on the uncompressed model
pixi run probs                                              # 13 settings of qwen3-8b, about 4.5 h
pixi run probs-4b                                           # 7 settings of qwen3-4b, about 2 h

# 3. measures, intervals, figures (no GPU)
pixi run probs-report --model qwen3-8b
pixi run probs-report --model qwen3-4b

# 4. the files in results/v1 (CSVs, report, figures, short answers, gzipped raw scores)
pixi run probs-export --model qwen3-8b      # gathers short.jsonl and runs.json; likewise qwen3-4b
pixi run probs-results --out results/v1
```

For a new prompt file: `bash scripts/probs_queue.sh qwen3-8b v2` writes to `probs-v2/` beside the v1
outputs; add `--tag v2` to the report commands. If only the candidate list changed, only the `scoring`
condition needs rerunning (`wu probs --model M --setting S --tag v2 --what scoring`).

Compression details (own block-wise GPTQ, group size 128, calibration windows and seeds) are in
`results/v1/qwen3-8b/NOTE.md`.

## What is not here

- **Model weights.** Base weights come from Hugging Face; compressed ones are rebuilt by `wu compress`.
- **The uncompressed model's full next-token distribution** at every answer position (float16, about 2 GB
  per model) and the hidden states used for the probe. They are in the local run directory
  (`~/models/wu-runs/<model>/base/probs-v1/full-*.npy`); `position_stats` carries the KL from the
  uncompressed model computed from them.
- **WANDA 70%** was not built, so it was not run.
- **One comparison cannot be rerun from here alone:** `shared_with_l1` in the report joins 92 of these
  facts with an earlier PopQA-3000 run, whose outputs are not in this repository. The
  report skips it when they are absent.

## Relation to the development repository

`src/`, `tests/`, `scripts/`, `data/`, `pyproject.toml` and `pixi.lock` are a copy of
the development repository's `model/` folder as of 15 Nov 2026. That folder remains where the code is developed; this copy
was checked by running its tests (58 pass) against the pinned environment.

The code hashes recorded in `results/v1/*/runs.json` are those of the code as it was run on 15 Nov 2026.
The files have since gained licence headers and small path changes, so their hashes differ; the
computation is unchanged.

## License

MIT; see [`LICENSE`](../LICENSE). Copyright (c) 2026 Daria Bikina.
