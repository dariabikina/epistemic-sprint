#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# E1: models trained to report confidence in their own guess (seeds 0, 1), then compressed; does the report follow
# the compressed model's correctness? Waits for gt_refit_worlds.sh (one GPU job at a time).
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/gt/queue.log
while pgrep -f "scripts/gt_refit_worlds.sh" > /dev/null; do sleep 60; done
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
step gt-test-analysis
for seed in 0 1; do
  [ -f $R/gt/runs/zipf-report-gt-6m-q-s$seed-long/summary.json ] || step gt report-train --seed $seed
  step gt report zipf-report-gt-6m-q-s$seed-long
done
echo "== $(date '+%m-%d %H:%M') report done" >> $L
