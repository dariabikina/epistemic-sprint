#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# Third queue, 2026-11-15: the same reads (wu.confidence) for qwen3-14b, the settings that already have graded
# answers. Waits for the first two queues (one GPU job at a time). Same log.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/confidence.log
while pgrep -f "scripts/confidence_(overnight|sizes).sh" > /dev/null; do sleep 120; done
step() {
  free=$(df -g "$R" | awk 'NR==2 {print $4}')
  if [ "$free" -lt 8 ]; then echo "!! SKIPPED (only ${free} GB free): $*" >> $L; return; fi
  echo "== $(date '+%m-%d %H:%M') $*" >> $L
  pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L
}
for s in base gptq-3 awq-3 wanda-50 gptq-4 awq-4 shortgpt-10; do step wu confidence --model qwen3-14b --setting $s; done
step wu confidence-report --model qwen3-14b
echo "== $(date '+%m-%d %H:%M') confidence 14b done" >> $L
