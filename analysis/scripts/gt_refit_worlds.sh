#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# tp and warped with smaller amplitudes (values stay inside [-6, 6]; see GT.md, "Approximations and
# changes"), and mix, which contains both: old pools and runs archived, pools rebuilt, every run that used
# them retrained with its original settings. Waits for gt_inside_figures.sh.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
G=$R/gt
L=$G/queue.log
while pgrep -f "scripts/gt_(review_fixes|inside_figures).sh" > /dev/null; do sleep 60; done
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
A=$G/archive/2026-11-14-end-bins
mkdir -p $A/data $A/runs
for w in tp warped mix; do
  [ -d $G/data/$w ] && mv $G/data/$w $A/data/
  for r in $G/runs/$w-gt-*; do [ -d "$r" ] && mv "$r" $A/runs/; done
done
step gt-test
step gt data tp warped mix
for seed in 0 1; do for w in tp warped mix; do step gt train $w --seed $seed; done; done
step gt train mix --shape gt-1m
step gt train mix --shape gt-20m
step gt train mix --steps 32000 --eval-every 2000 --tag long
step gt-notes
step gt-plot
echo "== $(date '+%m-%d %H:%M') refit worlds done" >> $L
