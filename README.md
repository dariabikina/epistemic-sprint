# epistemic-probabilities

Data for a study of what happens when an LLM loses facts under compression. **Main measure: token probabilities.**
For each fact we measure how strongly the model prefers the correct answer over plausible wrong ones, what it answers
itself, and whether it says it knows the answer. Those measures are compared across various compression settings.

- **Model:** Qwen3-8B, thinking off, no system prompt
- **Languages:** English and Russian, the same facts appear in both, with the same IDs

## Contents

| file | what it is | needed to run? |
|---|---|---|
| `prompts_rendered.jsonl` | **every prompt**, already rendered with Qwen3-8B's chat template, plus the continuations to score | **yes, this is what you run** |
| `facts_en.jsonl`, `facts_ru.jsonl` | 400 facts × 2 languages: questions, gold answers, metadata | reference / joins |
| `controls_en.jsonl`, `controls_ru.jsonl` | 100 invented subjects × 2 languages (no true answer) | reference |
| `candidates.jsonl` | candidate answers per fact (by Wikidata ID); already expanded into `prompts_rendered.jsonl` | reference |
| `prompts.json` | prompt wording and few-shot examples the rendering was built from | documentation |

Every row in `prompts_rendered.jsonl` has `id`, `lang` (en | ru), `relation`, `is_control`, `variant` (question |
paraphrase), `condition` and `prompt`.
- **Feed `prompt` to the model as a raw string.** It already contains the chat template, the empty `<think>\n\n</think>`
  block and the assistant prefill (`Answer:` / `Ответ:`). Don't apply the template again.
- **Tokenise with `add_special_tokens=False`.** Keep special tokens (`<|im_end|>`) as single tokens, which is the
  default. For every row, the tokens of `prompt` are a prefix of the tokens of `prompt + continuation`; we checked this
  on 772 sampled pairs.

Each prompt (for `short` and `scoring`) is:
- **an instruction for the relation**, e.g. «Ответь одним словом.» for capitals, «Ответь только именем этого человека, без
  пояснений.» for people;
- **3 short worked examples** of the same relation ("Capital of Japan? Answer: Tokyo");
- **the question,** then the prefill.

## The dataset
- **Facts:** 400 from [PopQA](https://huggingface.co/datasets/akariasai/PopQA) (Mallen et al., 2023), across 10
  relations: capital, capital of, country, place of birth, occupation, author, composer, director, father, mother.
- **Popularity bands:** balanced over three bands of PopQA `s_pop` (English Wikipedia monthly page views of the subject);
  The tail is < 3,052, the head is ≥ 33,854, and the bands hold 134 / 133 / 133 facts
  - **Selection effect:** only facts whose subject has both a Russian Wikipedia article are kept
- **Gold answers:**
  - **`gold`:** PopQA answers plus Wikidata labels and aliases of the answer entities, in each language
  - **`gold_wikidata`:** adds every current Wikidata value of the relation (e.g. Atlanta → Georgia).
- **Plausible-wrong distractors:** occupation and country have dedicated plausible-wrong distractors: current neighbouring countries; occupations
  from the same field that are not true, broader or narrower forms of, or commonly paired with, the person's real
  occupations (feminine Russian forms for women). Other true answers are separate `true_alt` candidates
- **Nonce answers (invented answers)**: every real fact should be scored against 2 invented answers of the right type
- **Controls (questions with no answer):** checked against Wikidata, Wikipedia (en / ru / pt) and the web, so that no name matches or closely
  resembles a real entity

## The three conditions

### 1. `short`: the model's own short answer (1,994 prompts)
**What to run:** greedy generation, `max_new_tokens = 24`, stopping at `<|im_end|>`. In our test, 96% (English) and 99%
(Russian) of answers were a bare name

**Why it's needed:** it shows what the model actually answers in each setting. We use those answers to:
- **tell the two kinds of loss apart:**
  - **displaced:** a wrong answer took over;
  - **diffuse:** the probability just spread out;
- **add the model's wrong answers and any spellings we missed** to the candidate list before the final scoring.

**Send back** `short.jsonl`: `{id, lang, setting, variant, text, token_ids, token_logprobs, finish_reason}`.

### 2. `scoring`: probability of each candidate answer (3,988 prompts, no generation)
**What a row holds:** `continuations` lists every candidate surface form as `text` = `" " + form + "<|im_end|>"`, with
its `qid`, its `role` and the `form` itself.

**Roles:**
- `gold`: the correct answer;
- `true_alt`: another answer that is also true per Wikidata, e.g. Georgia for "What is Atlanta the capital of?";
- `hard` / `popmatched`: plausible wrong answers;
- `distractor`: candidates for the invented controls
- `nonce`: an invented answer of the right type, 2 per real fact (qid is null).

**What to do:** one forward pass over prompt + continuation, then store per continuation:
- `logprob`: the sum of the continuation tokens' log-probs;
- `first_logprob`: the first token's log-prob;
- `n_tokens`.

Reuse the prompt's KV cache across its continuations.

**Variants:**
- **Example sets:** `exemplar_set` A is primary, B a noise floor. Run both.
- **Null prompts:** rows with `variant = question` for facts also have `null_prompt`, the same prompt with an invented
  subject of the same relation. Score **the same continuations** under it too (`is_null = true`); this corrects for
  answer priors (e.g. "United States" is likely for any country question).
- **Per prompt, at the answer position:** the entropy of the full next-token distribution and the top-20 tokens with
  log-probs. For dense only, also save the full log-softmax vector (float16) so we can compute KL(dense ‖ compressed).

**Send back:**
- `scores.jsonl`: `{id, lang, setting, condition: "scoring", variant, exemplar_set, is_null, qid, role, form, logprob, first_logprob, n_tokens}`
- `position_stats.jsonl`: `{id, lang, setting, variant, exemplar_set, is_null, entropy, top20}`

### 3. `knows`: does the model say it knows? (2,000 prompts)
"Do you know the answer to this question? {question} Answer with just one word, Yes or No." / «Знаешь ли ты ответ на этот
вопрос? {question} Ответь только одним словом, Да или Нет.», then the prefill. Score the two continuations
(`" Yes<|im_end|>"` / `" No<|im_end|>"`, Да / Нет) exactly as in scoring.

**Send back** these rows in `scores.jsonl` with `condition: "knows"`.

## Order of work
1. **Run `short` in all 6 settings,** send `short.jsonl` (minutes per setting).
2. **We add each setting's wrong answers and missed spellings** to the candidates, freeze them, and send an updated
   `prompts_rendered.jsonl` (v2).
3. **Run `scoring` and `knows` in all 6 settings on v2.** Scoring before step 2 would have to be redone.
4. **Report broken settings:** if a setting (likely Wanda 70%) produces mostly broken text in `short`, tell us before
   scoring it.

## Please confirm before starting
- **thinking:** off;
- **model:** the exact model revision and its own chat template;
- **compression details:** the GPTQ calibration data and the Wanda sparsity pattern (unstructured?), so we can report
  them.

## Field reference

**`facts_en.jsonl`:**
- **IDs:** `id` (`<subject QID>_<relation>`), `relation`, `s_qid`, `o_qid`, `o_qids`, `o_qids_extra`,
  `wikidata_values`, `popqa_ids`.
- **Subject:** `subj`, `subj_ru`, `subj_pt` (unused), `sex`.
- **Popularity:** `s_pop`, `pop_en`, `pop_ru`, `pop_pt`, `n_sitelinks`, `band`.
- **Questions:** `question_en`, `paraphrase_en`.
- **Gold:** `gold{en,ru,pt}`, `gold_wikidata{en,ru,pt}`. Portuguese fields are present but unused and unreviewed.

**`facts_ru.jsonl`:**
- **IDs:** `id`, `relation`, `s_qid`, `o_qid`, `o_qids`.
- **Subject:** `subj_ru`, `subj_en`, `sex`.
- **Popularity:** `s_pop`, `pop_en`, `pop_ru`, `pop_pt`, `n_sitelinks`, `band` (English band, primary), `band_ru`
  (secondary).
- **Questions:** `question_ru`, `paraphrase_ru`, `question_en`.
- **Gold:** `gold{ru, ru_lenient, ru_wikidata, en}`.
- **Checks:** `flags`, `same_name_ru`, `same_name_en`.

**`controls_*.jsonl`:** `id`, `relation`, `subj` / `subj_ru`, `sex`, `is_control`, `question_*`, `paraphrase_*`, `gold`
(empty).

**`candidates.jsonl`:**
- **Fact:** `id`, `relation`, `band`, `is_control`, `scoring_eligible`, `scoring_excluded_langs`.
- **Candidates:** `candidates[{qid, role (gold | true_alt | hard | popmatched | distractor | nonce), label_en, forms{en,ru,pt}}]`.
- **Other:** `from_runs` (filled from the `short` runs), `null_control_id`.

## Reference code (sketch, not run here: no GPU in our environment)
```python
import copy, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="auto").eval()
END = tok.convert_tokens_to_ids("<|im_end|>")

def short(prompt):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=24, do_sample=False, eos_token_id=END,
                         output_scores=True, return_dict_in_generate=True)
    gen = out.sequences[0, ids.shape[1]:]
    lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, gen)]
    return tok.decode(gen, skip_special_tokens=True), gen.tolist(), lps

def score(prompt, conts):
    p = tok(prompt, add_special_tokens=False).input_ids
    with torch.no_grad():
        out = model(torch.tensor([p], device=model.device), use_cache=True)
    first = out.logits[0, -1].float().log_softmax(-1)
    res = []
    for c in conts:
        full = tok(prompt + c["text"], add_special_tokens=False).input_ids
        assert full[:len(p)] == p
        cid = torch.tensor([full[len(p):]], device=model.device)
        with torch.no_grad():
            o = model(cid, past_key_values=copy.deepcopy(out.past_key_values), use_cache=False)
        lp = torch.cat([first[None], o.logits[0, :-1].float().log_softmax(-1)])
        t = lp.gather(1, cid[0][:, None])[:, 0]
        res.append({**c, "logprob": t.sum().item(), "first_logprob": t[0].item(), "n_tokens": len(t)})
    return res, first   # `first`: the next-token distribution at the answer position (entropy, top-20, KL)
```

## Analysis notes
- **Compare within a fact:** each setting against dense on the same fact (gold ranked first? share of distractors
  beaten? margin), and use ranks. Never compare raw log-probs across languages: Cyrillic splits into many more tokens.
- **Occupation:** the best of `gold` and `true_alt` counts as correct.
- **"Lost but unaware":** the gold drops out of first place while `knows` P(Yes) stays high.

## Sources
PopQA (Mallen et al., 2023); Wikidata; Wikimedia Pageviews API (2025-01 to 2025-10).

## Running it

The code that runs these prompts on compressed Qwen3 models is in [`analysis/`](analysis/); its
[`README`](analysis/README.md) describes the outputs and the results already in `analysis/results/v1/`.

**You need:** an Apple-silicon Mac (the code uses [MLX](https://github.com/ml-explore/mlx); about 48 GB of
memory for Qwen3-8B), [pixi](https://pixi.sh), and network access to Hugging Face for the model weights
(about 16 GB for Qwen3-8B) and the calibration text.

```sh
git clone <this repository> && cd epistemic-probabilities/analysis
pixi install --locked          # the exact environment in pixi.lock
pixi run test                  # 58 tests, about 20 s, no model needed
pixi run probs-smoke           # about 40 facts on uncompressed Qwen3-8B, about 4 min after the download
```

Outputs go to `~/models/wu-runs/<model>/<setting>/` . Set `WU_RUNS` to put them elsewhere:
`WU_RUNS=/some/folder pixi run probs-smoke`. The prompts are read from `../prompts_rendered.jsonl`.

To replicate the full run:

```sh
# 1. weights (pinned revisions) and the compressed models; built once, then cached under the run folder
pixi run wu download --model qwen3-8b
for s in gptq-3 gptq-4 awq-3 awq-4 wanda-30 wanda-50 wanda-2:4 shortgpt-5 shortgpt-10; do
  pixi run wu compress --model qwen3-8b --setting $s      # rtn-3, rtn-4 and magnitude-30 need no build
done

# 2. short answers, candidate scores and "do you know?" for all 13 settings (about 4.5 h)
pixi run probs
pixi run probs-4b              # the 7 settings of Qwen3-4B (about 2 h), after the same two steps for qwen3-4b

# 3. measures, intervals and figures (no GPU)
pixi run probs-report --model qwen3-8b

# 4. the files as they are in analysis/results/v1
pixi run probs-export --model qwen3-8b
pixi run probs-results --out results/v1
```

`pixi task list` shows every task with a one-line description. Without a Mac you can still work from the
files in `analysis/results/v1/`: summary and per-question CSVs, the short answers, and the gzipped scores.

## License

MIT; see [`LICENSE`](LICENSE). Copyright (c) 2026 Daria Bikina.
The facts are drawn from PopQA (Mallen et al., 2023), which has its own terms.
