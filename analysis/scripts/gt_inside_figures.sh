#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# I1 again, for the layer figure (layers.png, from 2,000 saved points). Waits for gt_review_fixes.sh.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/gt/queue.log
while pgrep -f "scripts/gt_review_fixes.sh" > /dev/null; do sleep 60; done
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
step gt inside zipf-gt-6m-q-s0-long --settings gptq-2,wanda-50,rtn-3+emb
for r in zipf-gt-6m-q-s0-d128k zipf-iid-gt-6m-q-s0-long zipf-bursty-gt-6m-q-s0-long; do step gt inside $r --settings gptq-2,wanda-50; done
step gt-notes
echo "== $(date '+%m-%d %H:%M') inside figures done" >> $L
