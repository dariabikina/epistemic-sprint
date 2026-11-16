#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# P1, 2026-11-15: the epistemic-probabilities prompts (wu.probs) on every built setting of one model.
# Usage: bash scripts/probs_queue.sh MODEL TAG "SETTINGS"    (base must come first: the others need its
# next-token distributions for the KL). Short answers for every setting first, then scoring and knows.
# One GPU job at a time; a failed step does not stop the queue. Log: ~/models/wu-runs/probs.log
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/probs.log
M=${1:-qwen3-8b}; TAG=${2:-v1}
S=${3:-base wanda-2:4 gptq-3 wanda-50 awq-3 gptq-4 awq-4 shortgpt-10 rtn-3 rtn-4 wanda-30 magnitude-30 shortgpt-5}
while pgrep -f "wu probs --model" > /dev/null; do sleep 30; done
step() {
  free=$(df -g "$R" | awk 'NR==2 {print $4}')
  if [ "$free" -lt 8 ]; then echo "!! SKIPPED (only ${free} GB free): $*" >> $L; return; fi
  echo "== $(date '+%m-%d %H:%M') $*" >> $L
  pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L
}
for s in $S; do step wu probs --model $M --setting $s --tag $TAG --what short; done
echo "== $(date '+%m-%d %H:%M') $M $TAG short answers done" >> $L
for s in $S; do step wu probs --model $M --setting $s --tag $TAG --what scoring,knows; done
echo "== $(date '+%m-%d %H:%M') $M $TAG probs done" >> $L
