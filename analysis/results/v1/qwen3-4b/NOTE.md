# Token probabilities, Qwen3-4B, prompt file v1 (15 Nov 2026)

Run on Apple silicon (MLX, bf16), from `prompts_rendered.jsonl` as it was on 15 Nov
(sha256 in `runs.json`). Code: `analysis/src/wu/probs.py`, `probs_report.py`.

## Files

`short.jsonl` is beside this note. `scores.jsonl` and `position_stats.jsonl` are in `raw/<setting>/`,
gzipped, one folder per setting (`wanda-2of4` is WANDA 2:4); every row carries `setting`.

| file | rows | fields |
|---|---|---|
| `short.jsonl` | 1,994 × 7 settings | `id, lang, setting, variant, exemplar_set, text, token_ids, token_logprobs, finish_reason` |
| `scores.jsonl` | scoring and `knows` continuations × 7 settings | `id, lang, setting, variant, exemplar_set, condition, is_null, role, qid, form` (scoring) or `label` (knows), `logprob, first_logprob, n_tokens, token_logprobs` |
| `position_stats.jsonl` | one per prompt × 7 settings | `entropy`, `top20`, `kl_from_base` at the answer position |
| `runs.json` | per setting | model revision, prompt-file hash, token hash, code hashes, versions, minutes |

`logprob` is the sum of `token_logprobs` and includes the end-of-turn token, as the README asks.

## The README's "please confirm" items

- **Model:** `Qwen/Qwen3-4B` at revision `1cfa9a72`, `enable_thinking=False`. Its tokenizer and chat
  template give the same tokens as Qwen3-8B's for every prompt (same token hash), so the 8B-rendered file is used as is. Prompts are tokenized from your raw strings with
  no special tokens added.
- **Prefix property:** holds for every prompt + continuation pair (asserted at run time).
- **Counts:** 1,994 short, 3,988 scoring (1,594 of them null prompts), 2,000 `knows`.
- **Check:** where the greedy short answer is also a candidate, its log-prob from generation and from
  scoring agree (median difference −0.001).

## Broken text in `short`

One setting is partly broken: **10 blocks removed, in Russian**. About a quarter of its answers (234 of
994) are not names: mixed scripts, Chinese text, markdown. Its short-answer accuracy there is 4%. It was
scored like the others; treat its Russian numbers with care.

The other settings are fine. All end at least 91% of answers at `<|im_end|>` within 24 tokens. WANDA 50%
usually adds a full stop after the name (722 of 994 Russian answers, 283 of 1,000 English ones), which the
grading ignores; GPTQ 3-bit sometimes answers in a sentence (about 5% in Russian).

## Settings (7)

`base`, `gptq-3`, `gptq-4`, `awq-3`, `awq-4`, `wanda-50`, `shortgpt-10`.

- Quantization is weight-only, group size 128; `lm_head` and embeddings stay at full precision.
- GPTQ is our own block-wise implementation, calibrated on 128 × 512-token C4 windows, seed 42.
- WANDA is unstructured (30%, 50%) or 2:4 semi-structured, calibrated on 32 × 512-token C4 windows.
- ShortGPT removes 5 or 10 blocks.
- MLX, not torch: numbers will differ slightly from a CUDA run.

## Things to know before using the numbers

- **Numerical floor.** In bf16, the batch a prompt is scored in changes a log-prob by up to 0.34
  nats (measured on Qwen3-8B) on likely candidates (95% under 0.09), more on unlikely ones. Differences smaller than that
  are not signal.
- **`knows` and the end token.** After "Yes"/"No" the model rarely ends its turn (measured on Qwen3-8B: mean log p
  of the end token −6.5 and −8.3 in the base model), so `logprob` carries a term unrelated to knowing. Our
  report uses `sum(token_logprobs[:-1])` for Yes minus No; the two versions correlate at 0.97.
- **«Нет» is two tokens, «Да» one.** The second token of «Нет» is certain, so this does not bias
  the log-odds.
- **Yes + No do not always hold the mass.** In some settings other tokens take most of the
  probability at that position (e.g. 10 blocks removed, Russian: Yes + No = 0.21). `top20` in
  `position_stats.jsonl` shows what does.
- **The level of P(Yes) moves with the method**, by several nats, on all facts alike. Compare
  within a setting (ranks, AUROC, lost against kept), not mean P(Yes) across settings.

## Things in the prompt file you may want to change for v2

- Example set B has two worked examples for `director` and `father`, three elsewhere.
- `facts_en.jsonl` gold lists include PopQA answers (e.g. "Chris Claremont") that are not among the
  gold candidates in scoring.
- README line 39 is unfinished.

When v2 arrives only `scoring` needs rerunning (about 16 minutes per setting).
