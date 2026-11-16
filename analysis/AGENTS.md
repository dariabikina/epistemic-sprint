# For an agent working in this repository

This folder (`analysis/`) was added on 15 Nov 2026. It holds the results of running
`../prompts_rendered.jsonl` (the version on `main` that day, called v1 here) on compressed Qwen3-8B and
Qwen3-4B, plus the code and plan. `README.md` beside this file describes everything; this note covers
what you need in order to act on it.

## What we need back

An updated `prompts_rendered.jsonl` (v2) with the candidate list frozen, as in step 2 of the top-level
README's "Order of work". To build it you need each setting's short answers:

- `results/v1/qwen3-8b/short.jsonl` and `results/v1/qwen3-4b/short.jsonl`: one row per prompt and
  setting, with `setting`, `text`, `token_ids`, `token_logprobs`, `finish_reason`.
- `results/v1/per-question.csv` is the same information joined to our grading: `short_answer`,
  `short_correct`, `correct_answers`, and `answer_margin` per model, setting and question. Rows with
  `short_correct = 0` are where to look for wrong answers to add and for spellings we missed.

Our grading (`short_correct`) is lenient containment: lower-cased, punctuation and English articles
removed (accents are kept), the answer counts if any form in `gold[lang]` of `facts_en.jsonl` / `facts_ru.jsonl` appears in it as
whole words. A correct answer in a spelling missing from `gold` is therefore marked wrong. Please
review those rather than trusting the column.

When you make v2, keeping these fixed means we rerun only `scoring` (about 16 minutes per setting):

- the `id`s, and the `short` and `knows` rows exactly as they are;
- the continuation format `" " + form + "<|im_end|>"`;
- the property that a prompt's tokens are a prefix of the tokens of prompt + continuation.

If any of these has to change, say so in the PR or commit message and we will rerun what it affects.

## Reading the files

```python
import gzip, json

def rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]

scores = rows("results/v1/qwen3-8b/raw/wanda-2of4/scores.jsonl.gz")
stats  = rows("results/v1/qwen3-8b/raw/wanda-2of4/position_stats.jsonl.gz")
```

- One folder per setting under `raw/`. The folder for WANDA 2:4 is `wanda-2of4`; inside the rows the
  `setting` field says `wanda-2:4`. `base` is the uncompressed model.
- `scores.jsonl.gz` holds the `scoring` rows and then the `knows` rows. Scoring rows have `qid`, `role`,
  `form`, `exemplar_set` (A or B) and `is_null`; `knows` rows have `label` (Yes / No / Да / Нет) and no
  `exemplar_set`.
- Every row has `logprob`, `first_logprob`, `n_tokens` as the README specifies, and also
  `token_logprobs` (one value per continuation token, the last being `<|im_end|>`).
- `position_stats.jsonl.gz` has `entropy`, `top20` and `kl_from_base` (KL from the uncompressed model's
  next-token distribution to this setting's) per prompt.
- `runs.json` records, per setting, the model revision, the sha256 of the prompt file that was run, and
  hashes of the code.

## Before you draw conclusions from the numbers

Each of these changed a result for us; `results/v1/qwen3-8b/NOTE.md` has the details.

- **`knows`: leave out the end token.** After "Yes" or "No" the model rarely ends its turn, so `logprob`
  carries a term that has nothing to do with knowing. Use `sum(token_logprobs[:-1])` for Yes minus No.
  That is what `yes_minus_no_log_odds` and `p_yes` in `per-question.csv` are.
- **Do not compare the level of P(Yes) between settings.** Compression shifts it as a whole, by several
  nats, for real and invented subjects alike, and the direction differs between methods, languages and
  the two model sizes. "Lost but unaware" defined as P(Yes) staying high inherits that shift. Compare
  within a setting: ranks, AUROC, lost facts against kept facts.
- **Differences under about 0.3 nats are not signal.** The runs are in bf16; the batch a prompt is
  scored in moves a log-prob by up to 0.34 nats on likely candidates.
- **Yes and No do not always hold the probability mass** (`yes_plus_no_mass` in `per-question.csv`).
  Where it is low the model would mostly have replied with something else.
- **Qwen3-4B with 10 blocks removed is partly broken in Russian:** about a quarter of its short answers
  are not names.
- **Settings:** we ran 13 on 8B and 7 on 4B, not the README's six, and WANDA 70% is not among them.
  Take the subset you need.

## Things in the prompt file worth fixing in v2

- Example set B has two worked examples for `director` and `father`, three for the other relations.
- `facts_en.jsonl` `gold` includes PopQA answers that are not among the `gold` candidates in scoring
  (for "Who is the author of New Mutants?", `gold` lists Chris Claremont; the scored gold forms are
  Bill Sienkiewicz's).
- Line 39 of the top-level README ("only facts whose subject has both a Russian Wikipedia article")
  is unfinished.

## What you can and cannot rerun

- **Cannot, without an Apple-silicon Mac:** the model runs. The code uses MLX, and the weights are not
  in the repository. Send v2 and `scoring` is rerun.
- **Can, on any machine:** everything from the files here. `per-question.csv` and the raw files need
  only Python. The measure definitions are in the docstring at the top of `src/wu/probs_report.py`.
- **Do not edit files under `results/` by hand.** They are written by `scripts/probs_results.py` from
  the run directories and are overwritten when it runs again.

Nothing outside `analysis/` was changed by this work.
