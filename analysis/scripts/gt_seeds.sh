#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# C1, seeds 2 and 3. Waits for gt_zipf_sweep.sh. Trains and compresses MAIN with seeds 2 and 3, then the
# across-seed table for seeds 0-3.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/gt/queue.log
while pgrep -f "scripts/gt_zipf_sweep.sh" > /dev/null; do sleep 60; done
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
for seed in 2 3; do
  step gt train zipf --shape gt-6m-q --seed $seed --steps 32000 --eval-every 2000 --tag long
  step gt compress zipf-gt-6m-q-s$seed-long
done
R=zipf-gt-6m-q-s0-long,zipf-gt-6m-q-s1-long,zipf-gt-6m-q-s2-long,zipf-gt-6m-q-s3-long
step gt seeds $R
step gt-notes
echo "== $(date '+%m-%d %H:%M') seeds done" >> $L
