#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# Overnight queue, 2026-11-15: facts and marbles (wu.confidence) for qwen3-8b and qwen3-1.7b, then two more
# seeds of the ground-truth report model (E1), then the CPU-only reports. One GPU job at a time; every step
# is logged; a failed step does not stop the queue; a step whose output exists is skipped by the command.
# Experiments L1 (facts), L2 (marbles), L3 (inside) in PLAN.md.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/confidence.log
step() {
  free=$(df -g "$R" | awk 'NR==2 {print $4}')
  if [ "$free" -lt 8 ]; then echo "!! SKIPPED (only ${free} GB free): $*" >> $L; return; fi
  echo "== $(date '+%m-%d %H:%M') $*" >> $L
  pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L
}
S8="base gptq-3 awq-3 wanda-50 wanda-2:4 gptq-4 awq-4 wanda-30 magnitude-30 shortgpt-5 shortgpt-10"
S17="base rtn-4 rtn-3 rtn-2 gptq-4 gptq-3 awq-4 awq-3 wanda-50 wanda-2:4 magnitude-30 shortgpt-10"

# 1. 8B, the settings that already have graded answers (about 30 min each)
for s in $S8; do step wu confidence --model qwen3-8b --setting $s; done
# 2. 8B with round-to-nearest, the setting that raised stated confidence in the ground-truth world:
#    perplexity and graded answers first (wu sweep), then the same reads
step wu sweep --model qwen3-8b --settings rtn-4,rtn-3
for s in rtn-4 rtn-3; do step wu confidence --model qwen3-8b --setting $s; done
# 3. 1.7B, every setting (about 8 min each)
for s in $S17; do step wu confidence --model qwen3-1.7b --setting $s; done
# 4. Ground-truth report model (E1): seeds 2 and 3, trained and scored as seeds 0 and 1 were
for seed in 2 3; do
  [ -f $R/gt/runs/zipf-report-gt-6m-q-s$seed-long/summary.json ] || step gt report-train --seed $seed
  step gt report zipf-report-gt-6m-q-s$seed-long
done
# 5. Reports and figures (no GPU)
for m in qwen3-8b qwen3-1.7b; do step wu confidence-report --model $m; done
step gt-poster
echo "== $(date '+%m-%d %H:%M') confidence done" >> $L
