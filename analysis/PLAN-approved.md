# Plan: run the epistemic-probabilities experiments on the compressed Qwen3 models

## Context

The `epistemic-probabilities/` repository holds data and a spec, no code: 400 PopQA facts and 100 invented
controls, in English and Russian, as 7,982 pre-rendered Qwen3-8B prompts. Three conditions:

- `short`: the model's greedy answer (1,994 prompts).
- `scoring`: log-probability of each candidate answer (3,988 prompts, about 76k continuations, plus
  the same continuations under 1,594 null prompts with an invented subject).
- `knows`: log P(Yes) − log P(No) to "Do you know the answer to this question?" (2,000 prompts).

Its reference code is a torch/CUDA sketch. We run on MLX with the settings already built in
`~/models/wu-runs`. Decisions taken:

- **Settings:** all 13 built for qwen3-8b (base, rtn-3/4, gptq-3/4, awq-3/4, magnitude-30,
  wanda-30/50/2:4, shortgpt-5/10). Wanda 70% is not built and not added.
- **Staging:** run all three conditions on the current prompt file now, labelled v1. Rerun `scoring`
  when v2 arrives (`knows` and `short` do not depend on the candidate list).
- **Sizes:** qwen3-8b first, then qwen3-4b (7 settings).

The aim, in our K / M / coupling terms: K = does the gold answer still outrank the wrong ones;
M = does the model say it knows; coupling = does M follow K when K is lost.

## What gets built (all under `model/`)

1. **`src/wu/probs.py`** (new): the GPU side.
   - Load `prompts_rendered.jsonl` (path from `--prompts`, default `../epistemic-probabilities/`),
     sha256 it, tokenise the raw string with `add_special_tokens=False`, and assert the prefix
     property for every prompt + continuation pair.
   - `short`: `generate.generate(..., max_tokens=24)` (`src/wu/generate.py:55`). Write `text,
     token_ids, token_logprobs, finish_reason`.
   - `scoring` and `knows`: batched full pass over prompt + continuation, reusing `_batches` and
     `_logits` from `src/wu/confidence.py`. Add a small `token_logps()` beside `score()`
     (`confidence.py:138`) that returns per-token log-probs, so `logprob`, `first_logprob` and
     `n_tokens` come from one place. No KV-cache reuse: the batched pass gives the same numbers and
     its padding invariance is already tested. Null prompts are scored the same way with
     `is_null=true`.
   - Answer position: entropy, top-20, and for base the full log-softmax as float16.
   - `knows` prompts also go through `read(..., states=True)` (`confidence.py:95`) so the hidden
     state before the answer is kept for a probe.
   - `knows` rows carry `label` (yes/no) in place of `qid/role/form`.
2. **CLI** in `src/wu/cli.py`: `wu probs --model M --setting S [--what short,scoring,knows] [--n N]
   [--prompts PATH] [--tag v1]` and `wu probs-report --model M [--figures-only]`. Skips files that
   exist, like `wu confidence`.
3. **Outputs** in `~/models/wu-runs/<model>/<setting>/probs-<tag>/`: `short.jsonl`, `scores.jsonl`,
   `position_stats.jsonl`, `knows-states.npz`, `answer-logsoftmax.npz` (base only), `run.json`.
   - `run.json` records model revision, setting, prompt-file hash, code hashes, pixi.lock hash,
     versions and time, using `code_hashes()` and `provenance()` from `src/wu/gt/train.py:80-101`.
     The earlier confidence outputs lack this; the new ones will not.
4. **`src/wu/probs_report.py`** (new, CPU only): measures, intervals, figures. Reuses `_interval`,
   `pca_scores`, `probe_scores` from `confidence_report.py`, `auc` from `gt/stats.py`, the tracking
   index from `scripts/tracking.py`, and `frame`, `theme`, `family_colour`, `save` from
   `gt/poster.py`.
5. **Queue and tasks:** `scripts/probs_queue.sh` on the existing `step()` pattern (one GPU job at a
   time, log to `~/models/wu-runs/probs.log`). Pixi tasks `probs-smoke`, `probs`, `probs-4b`,
   `probs-report`, `probs-export`.
6. **`tests/test_probs.py`:** `token_logps` agrees with `score`, padding invariance, prefix
   assertion, output schema.

## Order of work

1. **Checks before any run.**
   - Re-render a sample of prompts with our pinned tokenizer (`Qwen/Qwen3-8B` @ b968826d,
     `enable_thinking=False`) and confirm the strings match the dataset's.
   - Confirm no other GPU job is running.
2. **Write the predictions into `PLAN.md`** before running (see below).
3. **Smoke:** `--n 40` on base and wanda-2:4. Check that a greedy answer that is also a candidate
   gets the same log-prob from `short` and from `scoring`, and time it to size the queue.
4. **8B queue:** `short` for all 13 settings first, then `scoring` + `knows`.
   - After `short`: per setting, the share of finished and bare-name replies and accuracy. Settings
     with mostly broken text are flagged in the report, and still scored (it is cheap).
5. **Export** `short.jsonl`, `scores.jsonl`, `position_stats.jsonl` concatenated over settings with
   a `setting` column, to `~/models/wu-runs/probs-v1/export/`, with a note answering the README's
   "please confirm" items. These go to the dataset's author; nothing is pushed.
6. **Report and figures** for 8B.
7. **4B:** first check its tokenizer and chat template reproduce the rendered prompts exactly. If
   they do, run its 7 settings; if not, stop and report.
8. **v2:** when it arrives, `pixi run probs -- --tag v2` reruns scoring only.

## Measures

All within a fact, within a language, paired with base; ranks, never raw log-probs across languages.

- **K:** gold ranked first (occupation: best of `gold` and `true_alt`), share of wrong candidates
  beaten, margin to the best wrong candidate; each also null-corrected. KL(base ‖ setting) and
  entropy at the answer position.
- **M:** `knows` log-odds; its floor from the 100 controls (P(Yes) on invented subjects) and the
  mass on Yes + No, to flag unresponsive readouts.
- **Coupling:** AUROC of `knows` for gold-first vs not; tracking index (does `knows` drop more on
  lost facts than on kept ones); "lost but unaware" share; displaced vs diffuse loss from the short
  answer and the entropy.
- **Noise floor:** disagreement between exemplar sets A and B, and question vs paraphrase, in base.
- **Intervals:** bootstrap over facts (a fact's variants and exemplar sets resample together),
  B = 1,000, seed 0, paired with base. By popularity band and relation as well as overall.

## Relating it back to the earlier work

- **Same 13 settings, side by side:** the new coupling AUROC next to prediction AUROC, stated
  P(True) AUROC and the probe from `confidence-report.json`.
- **Predictions to record first:**
  - K-side ranking signals stay close to base in every setting, as prediction AUROC did (−0.03 to
    +0.01 around 0.88).
  - `knows` loses track under WANDA 2:4, as P(True) did (−0.11 [−0.15, −0.08]).
  - A probe on the pre-answer state, refitted per setting, keeps its AUROC where `knows` does not.
- **Shared facts:** 96 of the 400 facts are in PopQA-3000 (checked). Join per fact: new gold
  log-prob vs stored `logp_gold`; pre-answer `knows` vs post-answer P(True); new vs old short-answer
  correctness. 96 gives wide intervals; reported as a consistency check only.
- **Open items this closes:**
  - `KNOWING.md` "Not done: a control for the answer string itself": the null prompts are that.
  - Popularity confound: the new set is balanced over bands by design.
  - One prompt wording per read: paraphrases and two exemplar sets now give a wording floor.
- **Known weakness carried over:** `knows` is asked one way only, so a lean towards Yes is not
  cancelled as it was for P(True). The controls are the reference for it.
- **Docs:** new section in `PLAN.md` (predictions, runs, results), a section in `KNOWING.md`, rows in
  `model/README.md`. No `.tex` wording is touched.

## To raise with the dataset's author (in the export note)

- Exemplar set B has 2 worked examples for `director` and `father`, not 3.
- The README's six settings are not listed; we ran 13 and she can subset.
- Our compression details: GPTQ is our own block-wise implementation, group size 128, 128×512 C4
  windows, seed 42, `lm_head` and embeddings full precision; WANDA is unstructured, 32×512 C4
  windows; 2:4 is semi-structured. MLX, not torch.

## Verification

- `pixi run test` passes, including `tests/test_probs.py`.
- Smoke: `short` and `scoring` log-probs agree on shared answers; row counts match the README
  (1,994 / 3,988 / 2,000); every `run.json` has hashes.
- Base sanity: gold-first rate in English is in line with short-answer accuracy; controls get lower
  `knows` than head-band facts.
- Every number in the report and figures has an interval, and each figure states its setup, signals
  and result in numbers.
