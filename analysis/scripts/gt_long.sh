#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# G3, C1. Waits for gt_after_overnight.sh and fixups_after_gt.sh. zipf on gt-6m-q, 32k steps, seeds 0 and 1,
# each compressed with all 18 settings; mix and gp-hyper on gt-6m, 32k steps.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/gt/queue.log
while pgrep -f "scripts/(gt_after_overnight|fixups_after_gt).sh" > /dev/null; do sleep 60; done
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
for seed in 0 1; do
  step gt train zipf --shape gt-6m-q --seed $seed --steps 32000 --eval-every 2000 --tag long
  step gt compress zipf-gt-6m-q-s$seed-long
done
for w in mix gp-hyper; do step gt train $w --steps 32000 --eval-every 2000 --tag long; done
step gt-notes
echo "== $(date '+%m-%d %H:%M') long runs done" >> $L
