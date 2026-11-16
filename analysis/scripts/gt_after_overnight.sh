#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# G1, G2. Waits for scripts/overnight.sh. Tests, smoke, the ladder (gt-6m, seeds 0 and 1), then gt-1m and
# gt-20m on four worlds. Log: <GT>/queue.log.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/gt/queue.log
mkdir -p $R/gt
echo "== $(date '+%m-%d %H:%M') waiting for the overnight queue" >> $L
while pgrep -f "scripts/overnight.sh" > /dev/null; do sleep 60; done
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }

step gt-ladder               # runs gt-test and gt-smoke first
step gt ladder --seed 1
for w in gp-hyper regimes zipf mix; do
  step gt train $w --shape gt-1m
  step gt train $w --shape gt-20m
done
step gt-notes
step gt-plot
echo "== $(date '+%m-%d %H:%M') done" >> $L
