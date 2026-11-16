# Plan, predictions and results

Copied from the development repository's `model/PLAN.md`, section P1 (15 Nov 2026). L1–L3 are the earlier confidence
experiments on PopQA-3000 in that file; `KNOWING.md` there holds the working notes.

## P1: token probabilities on the fact set (built 2026-11-15, afternoon)

Data and spec: `../epistemic-probabilities/` (its README is the spec). 400 PopQA facts
balanced over three popularity bands and 10 relations, plus 100 invented subjects with no answer, each
in English and Russian, as 7,982 prompts already rendered with Qwen3's chat template (thinking off,
three worked examples, prefill `Answer:` / `Ответ:`). Code: `src/wu/probs.py` (GPU),
`src/wu/probs_report.py` (CPU). Commands: `wu probs`, `wu probs-report`; queue `pixi run probs`.

**Three conditions, per setting.**
- `short`: greedy answer, at most 24 new tokens (1,994 prompts).
- `scoring`: log p of every candidate answer followed by the end-of-turn token (3,988 prompts, about
  76,000 continuations), and the same continuations under a *null prompt* that asks the same question
  about an invented subject (1,594 more prompts). Roles: gold, other true answers, plausible wrong
  answers, invented answers. At the answer position: entropy, top 20 tokens, KL from the base model.
- `knows`: log p of Yes and No to "Do you know the answer to this question?" (2,000 prompts), asked
  before any answer is given; and the residual stream at the last prompt token.

**In our terms.** K (content): does the gold answer outrank the wrong candidates. M (meta): does the
model say it knows. Coupling: does M follow K, and does it fall when K is lost.

**Decisions (15 Nov).**
- Settings: all 13 built for qwen3-8b, then the 7 built for qwen3-4b. The README's "6 settings" are not
  listed; a subset can be taken. WANDA 70% is not built.
- The prompt file as received is v1. Each setting's wrong answers will be added to the candidates
  (v2); `scoring` is then rerun with `--tag v2`. `short` and `knows` do not depend on the candidates.
- Each prompt is read once and its key-value cache shared by its continuations; prompts of one
  relation share the cache of their worked examples. A forward pass through the 8B model costs about
  0.1 s however short it is, so prompts of equal length are read together.
- Every output folder has a `run-*.json`: model revision, prompt-file hash, code hashes, versions.

**What this adds to L1–L3, and how the two are joined.**
- The same 13 settings: each new measure is set beside prediction AUROC, stated (P(True)) AUROC and
  the probe AUROC from `confidence-report.json`.
- 96 of the 400 facts are also in PopQA-3000 (matched on PopQA id). On those: the new gold log-prob
  against `logp_gold`, `knows` (asked before answering) against P(True) (asked about its own answer),
  and both short-answer gradings. 96 facts give wide intervals; this is a consistency check.
- The null prompt is the control for the answer string itself that `KNOWING.md` lists as not done.
- Popularity is balanced by design here, and there are two wordings and two example sets per
  question, so a wording floor can be stated.
- A weakness carried over: `knows` is asked one way only, so a lean towards Yes is not cancelled as
  it is for P(True). The invented subjects are the reference for it.

**Predictions, written before the runs** (each can fail):
1. K by rank moves little: the AUROC with which the gold answer's margin separates right from wrong
   short answers stays within 0.05 of base in every setting whose text is not broken (L1: prediction
   AUROC moved by −0.03 to +0.01).
2. `knows` is a weaker signal than the margin in the base model (L1: stated 0.81 against 0.88), and
   WANDA 2:4 is the setting where it loses most (L1: −0.11).
3. On facts whose gold answer drops out of first place, `knows` falls by less than the fall in K
   would predict: most lost facts still get P(Yes) within 0.1 of the base model's.
4. A probe on the pre-answer state, refitted on each setting, separates gold-first from not at least
   as well as `knows` does, in every setting (L3, with the caveat from the probe control that it may
   mostly re-read the answer probability).
5. The invented subjects get lower P(Yes) than tail facts in the base model, and compression narrows
   that gap.
6. No prediction for Russian against English; recorded as exploratory.

**Smoke (15 Nov, base, about 40 facts and controls).**
- The pinned model's own chat template reproduces all rendered `knows` prompts
  (`template_matches: true`); every prompt's tokens are a prefix of prompt + continuation.
- Where the greedy answer is also a candidate (45 cases), its log-prob from `scoring` equals the one
  from generation to within the end token's log-prob (median difference −0.001).
- Batch composition changes a log-prob by up to 0.34 nats on likely candidates (95% of them under
  0.09) and more on unlikely ones. This is bf16 arithmetic: the float32 test model agrees to 0.005.
  It is the numerical floor for every difference reported below.
- After "Yes" or "No" the base model rarely ends the turn (mean log p of the end token −6.5 and
  −8.3), so the README's score, which includes the end token, carries a term unrelated to knowing.
  Both are kept; the report's `knows` is the log-odds without the end token (correlation between
  the two 0.97). In Russian «Нет» is two tokens and «Да» one, and `**` takes some probability.
- Speed: about 0.15 s per scoring prompt, so about 18 minutes per setting.

**Runs (15 Nov).** Qwen3-8B, 13 settings, prompt file v1, all three conditions, 14:30–20:57, no
failures (`~/models/wu-runs/probs.log`). About 20 minutes per setting. Qwen3-4B (7 settings) queued
after it. Report: `<RUNS>/qwen3-8b/probs-v1-report.json`; figures `probs-v1-{coupling,lost,tracking}-{en,ru}.png`;
export in `<RUNS>/qwen3-8b/probs-v1-export/`.
Everything except weights and the raw candidate scores is also in `../epistemic-probabilities/analysis/`:
this plan, a copy of the code with `pyproject.toml` and `pixi.lock`, and `results/v1/` (summary CSVs,
`per-question.csv`, report JSON, figures, short answers, `runs.json`); `pixi run probs-results --out DIR`
writes the results folder. Its `README.md` says how to reproduce.
Plain-language versions for the poster (`probs_poster.py`): `probs-v1-plain-{1-forgetting,2-shows-vs-says,
3-kept-vs-lost,4-yes}.png`, English and Russian side by side, AUROC shown as pairs ranked correctly out
of 100. They are on the poster canvas under "Interim (15 Nov)" (`gt poster` writes the canvas).
Each plain figure states what the model is asked, what is read from it and what is counted. The canvas also
carries one real question taken through every step for the uncompressed 8B model (`probs_report.EXAMPLE`,
`probs_poster.example_card`): "Who is the author of The Bicentennial Man?", chosen for a margin and a P(Yes)
near the medians of correctly answered English questions. The detailed figures are on the canvas under "Details".

**Results, Qwen3-8B.** 400 facts × 2 wordings per language; brackets are 95% intervals over facts,
differences paired with the base model. Chosen rows; every setting is in the report.

| | short accuracy | gold first | AUROC margin → correct | AUROC `knows` → correct | AUROC real vs invented | `knows` order vs base (Spearman) | lean: change in mean log-odds |
|---|---|---|---|---|---|---|---|
| en base | 0.45 | 0.78 [0.74, 0.82] | 0.85 [0.81, 0.88] | 0.73 [0.68, 0.77] | 0.80 [0.76, 0.84] | 1 | 0 |
| en GPTQ 3-bit | 0.31 | 0.68 | 0.88 | 0.72 | 0.76 | 0.78 | −1.71 |
| en RTN 3-bit | 0.29 | 0.66 | 0.87 | 0.68 | 0.69 [0.64, 0.74] | 0.66 | −2.52 |
| en WANDA 50% | 0.32 | 0.65 | 0.87 | 0.70 | 0.76 | 0.70 | +0.73 |
| en WANDA 2:4 | 0.20 | 0.52 [0.47, 0.57] | 0.88 [0.84, 0.92] | 0.69 [0.63, 0.74] | 0.69 [0.64, 0.74] | 0.39 [0.32, 0.46] | −1.72 |
| en 10 blocks removed | 0.21 | 0.55 | 0.88 | 0.68 | 0.80 | 0.79 | +6.85 |
| ru base | 0.23 | 0.56 [0.51, 0.60] | 0.87 [0.84, 0.90] | 0.75 [0.70, 0.79] | 0.67 [0.63, 0.72] | 1 | 0 |
| ru GPTQ 3-bit | 0.17 | 0.46 | 0.90 | 0.69 | 0.63 | 0.62 | −3.38 |
| ru RTN 3-bit | 0.15 | 0.49 | 0.89 | 0.62 | 0.58 | 0.54 | −2.62 |
| ru WANDA 50% | 0.14 | 0.48 | 0.90 | 0.73 | 0.62 | 0.50 | +3.32 |
| ru WANDA 2:4 | 0.06 | 0.36 [0.31, 0.40] | 0.94 [0.90, 0.97] | 0.74 [0.64, 0.82] | 0.58 [0.52, 0.64] | 0.09 [0.01, 0.17] | +1.60 |
| ru 10 blocks removed | 0.06 | 0.43 | 0.83 | 0.59 | 0.60 | 0.24 | +11.31 |

Against the predictions:
1. **Held.** The margin's AUROC for a correct short answer changes by −0.00 to +0.04 in English and
   −0.04 to +0.07 in Russian (WANDA 2:4 in Russian rises, +0.07 [+0.02, +0.10]). Same as L1.
2. **First half held, second half did not.** `knows` is weaker than the margin in the base model
   (0.73 against 0.85). But its AUROC for a correct answer falls little anywhere (English −0.05 to
   +0.02; the interval excludes zero only for WANDA 30%), and WANDA 2:4 is not the largest fall
   (−0.04 [−0.11, +0.03]). In Russian the largest falls are 10 blocks removed (−0.15 [−0.27, −0.04])
   and RTN 3-bit (−0.12 [−0.19, −0.05]). WANDA 2:4 is the worst on two other measures: the order of
   facts by `knows` against the base model (0.39 English, 0.09 Russian) and, with RTN 3-bit, real
   against invented subjects.
3. **Wrong as stated, and the reason matters.** Only 17–68% (English) and 6–76% (Russian) of lost
   facts keep P(Yes) within 0.1 of the base model's, because compression moves P(Yes) as a whole:
   the mean log-odds shifts by −2.5 to +6.9 nats in English and −3.4 to +11.3 in Russian, on lost and
   kept facts alike. The sign depends on the method (3-bit quantization towards No, WANDA 50% and
   block removal towards Yes; WANDA 2:4 towards No in English and towards the middle in Russian,
   where the spread halves and 39% of replies fall between 0.4 and 0.6).
   What the prediction was after does hold, on measures a constant shift cannot move:
   - tracking (change in P(Yes) on lost minus kept facts) is within ±0.05 of zero in 9 of 12
     settings in English; where it is not, it is positive (RTN 3-bit +0.09, block removal +0.08 and
     +0.11), the wrong direction for awareness;
   - the AUROC with which `knows` separates kept from lost facts is the same for the compressed
     model as for the base model before the loss: paired difference −0.08 to +0.02 in English,
     interval includes zero in 10 of 11 settings with at least 20 lost facts (RTN 3-bit is lower).
     In Russian −0.05 to +0.08, includes zero in 9 of 12; higher for WANDA 30% (+0.04 [0.00, 0.08])
     and WANDA 50% (+0.08 [0.01, 0.15]), lower for AWQ 4-bit.
4. **Did not hold.** The probe for gold-first is not better than `knows`: base English 0.65
   [0.59, 0.71] against 0.72; WANDA 2:4 0.59 against 0.62 (Russian 0.51 against 0.62). For a correct
   short answer the probe is level with `knows` (0.70–0.77 against 0.68–0.75 in English) and below
   the margin everywhere. With 800 rows and 256 components this probe is far weaker than L3's
   (0.91 on 3,000 questions); not separated: sample size, or the prompt it reads.
5. **First half held; the second depends on the method.** Base: invented subjects get P(Yes) 0.20
   against 0.47 for tail facts in English (0.13 against 0.19 in Russian). The AUROC for real against
   invented falls from 0.80 to 0.69 for RTN 3-bit and WANDA 2:4 and stays at 0.76–0.81 elsewhere in
   English; in Russian from 0.67 to 0.58–0.63 for the 3-bit, WANDA 50%, 2:4 and block-removal settings.
6. Russian (exploratory): lower in content (gold first 0.56 against 0.78) and a weaker report in the
   base model (spread of log-odds 1.6 against 3.1; question and paraphrase agree at r = 0.46
   against 0.85). Margin AUROC is as high as in English.

Other numbers:
- Wording floor, base: gold-first flips between example sets A and B on 3.6% [2.4, 5.3] of English
  and 7.4% [5.4, 9.9] of Russian units; between question and paraphrase on 3.0% and 6.0%.
- Retention by popularity, WANDA 2:4, English: head 0.75, middle 0.50, tail 0.52.
- Null-prompt corrected gold-first is within 0.03 of the plain one in the base model (0.76 against
  0.78), so candidate priors are not what makes gold rank first.
- Shared with L1 (92 facts): short-answer grading agrees on 80–96%; margin against L1's gold
  log-prob, Spearman 0.60–0.72; `knows` against L1's stated P(True) 0.37 [0.18, 0.52] in base and
  lower after compression. Asking before answering and judging one's own answer are different reads.

**Qwen3-4B (7 settings, 20:58–22:48, no failures).** Report `<RUNS>/qwen3-4b/probs-v1-report.json`, export
`probs-v1-export/`. The same three results:
- Margin AUROC for a correct answer: 0.87–0.90 in English (base 0.87), 0.82–0.94 in Russian (base 0.91).
- `knows` for a correct answer: 0.57–0.65 in English (base 0.65); in Russian 0.65 in base, 0.39–0.70 after
  (below 0.5 for AWQ 3-bit, 0.43 [0.34, 0.52], and 10 blocks removed, 0.39 [0.28, 0.50]).
- Kept against lost: the paired difference includes zero in all 12 setting × language cases (−0.07 to +0.03).
- The lean moves more than on 8B (−10.2 to +4.6 nats in English), and its sign is not a property of the
  method: WANDA 50% moves 8B towards Yes (+0.7 English, +3.3 Russian) and 4B towards No (−10.2, −7.7);
  GPTQ 4-bit moves 4B by −6.9 and GPTQ 3-bit by +0.1.
- Base 4B says Yes to invented subjects more than 8B does (0.45 against 0.20 in English; 0.66 against 0.13
  in Russian), and `knows` in base is unrelated to L1's stated P(True) on the shared facts (0.05 [−0.16, 0.25]).

Limits:
- `knows` is asked one way, so its lean cannot be cancelled; shift-free measures are the ones to read.
- In some settings Yes and No hold a minority of the probability at that position (English: magnitude
  30% 0.46, RTN 3-bit 0.56; Russian: 5 blocks removed 0.26, 10 blocks 0.10); there the log-odds is a
  read of a reply the model would mostly not give.
- Unexplained: under WANDA 2:4 in Russian the two wordings agree better than in base (r = 0.78
  against 0.46) while the order no longer matches the base model's.
- "Lost" is defined against a fixed candidate list (v1); rerun `scoring` on v2 when it arrives.
- One model per size; probe intervals are for fitted scores.
