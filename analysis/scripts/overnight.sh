#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# Overnight queue, 2026-11-13. One GPU job at a time; every step logged; a failure does not stop the queue.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/overnight.log
step() { echo "== $(date +%H:%M) $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
S8="awq-3,gptq-3,wanda-50,shortgpt-5,awq-4,gptq-4,wanda-2:4,wanda-30,shortgpt-10,magnitude-30"

# 1. Stage 2 material: conversational answers that allow "I don't know" (8B)
for s in base awq-3 gptq-3 wanda-50 shortgpt-5 awq-4; do step wu eval --model qwen3-8b --setting $s --mode chat-idk --subset known+400 --batch-size 128; done
# 2. Inside the 8B model: states, probe, self-report for every setting
for s in ${S8//,/ }; do step wu represent --model qwen3-8b --setting $s; done
# 3. Where the damage happens: splicing and logit lens (8B: answer format is stable there)
for s in awq-3 gptq-3 wanda-50 shortgpt-5; do step wu splice --model qwen3-8b --setting $s --per-group 150 --step 4; done
for s in awq-3 gptq-3 wanda-50 shortgpt-5 wanda-2:4; do step wu lens --model qwen3-8b --setting $s --per-group 150; done
# 4. Perplexity over 32 windows (paired ratios with intervals), 8B
step python -I scripts/ppl_windows.py qwen3-8b base,$S8
# 5. Size series: same data, different sizes
for m in qwen3-0.6b qwen3-4b qwen3-14b; do
  step wu sweep --model $m --settings base,gptq-4,gptq-3,awq-4,awq-3,wanda-50,shortgpt-10
  step python -I scripts/ppl_windows.py $m base,gptq-4,gptq-3,awq-4,awq-3,wanda-50,shortgpt-10
  for s in awq-3 gptq-3; do step wu represent --model $m --setting $s; done
done
step wu splice --model qwen3-14b --setting awq-3 --per-group 150 --step 4
step wu lens --model qwen3-14b --setting awq-3 --per-group 150
# 6. CPU-only summaries
for m in qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b; do step wu uncertainty --model $m; step wu ppl-spread --model $m; done
step wu wrongtype --models qwen3-0.6b,qwen3-1.7b,qwen3-4b,qwen3-8b,qwen3-14b
step wu geometry --model qwen3-8b --settings awq-3,gptq-3,wanda-50,shortgpt-5
step wu geometry --model qwen3-8b --settings awq-3,gptq-3,wanda-50,shortgpt-5 --layer 19
echo "== $(date +%H:%M) done" >> $L
