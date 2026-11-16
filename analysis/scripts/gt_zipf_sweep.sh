#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# C2-C5, I1. Waits for gt_long.sh. Finished runs are skipped and evaluations cached, so a re-run resumes.
#   A. gt-1m-q and gt-6m-q at 8k / 32k / 128k steps
#   B. zipf variants on gt-6m-q, 32k steps
#   C. CORE settings on every run of A and B; single-block settings on MAIN
#   D. gt inside on MAIN, the 128k gt-6m-q run, zipf-iid and zipf-bursty
#   E. gt control over all runs
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/gt/queue.log
while pgrep -f "scripts/gt_long.sh" > /dev/null; do sleep 60; done
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
CORE="rtn-4,rtn-3,rtn-2,gptq-3,gptq-2,wanda-50,magnitude-50,rtn-3+emb"
MAIN=zipf-gt-6m-q-s0-long

# A. tokens per parameter
step gt train zipf --shape gt-6m-q --steps 8000 --tag d8k
step gt train zipf --shape gt-1m-q --steps 8000 --tag d8k
step gt train zipf --shape gt-1m-q --steps 32000 --eval-every 2000 --tag long
step gt train zipf --shape gt-1m-q --steps 128000 --eval-every 8000 --tag d128k
step gt train zipf --shape gt-6m-q --steps 128000 --eval-every 8000 --tag d128k
# B. the input distribution (same model and budget as MAIN)
for w in zipf-s0.7 zipf-s1.5 zipf-bursty zipf-fresh zipf-iid; do
  step gt data $w
  step gt train $w --shape gt-6m-q --steps 32000 --eval-every 2000 --tag long
done
# C. compression
RUNS="zipf-gt-6m-q-s0-d8k zipf-gt-1m-q-s0-d8k zipf-gt-1m-q-s0-long zipf-gt-1m-q-s0-d128k zipf-gt-6m-q-s0-d128k"
for w in zipf-s0.7 zipf-s1.5 zipf-bursty zipf-fresh zipf-iid; do RUNS="$RUNS $w-gt-6m-q-s0-long"; done
for r in $RUNS; do step gt compress $r --settings $CORE; done
SPLICE=""; for b in 0 1 2 3 4 5; do SPLICE="$SPLICE,gptq-2@$b,wanda-50@$b"; done
step gt compress $MAIN --settings ${SPLICE#,}
# D. inside
step gt inside $MAIN --settings gptq-2,wanda-50,rtn-3+emb
for r in zipf-gt-6m-q-s0-d128k zipf-iid-gt-6m-q-s0-long zipf-bursty-gt-6m-q-s0-long; do step gt inside $r --settings gptq-2,wanda-50; done
# E. control
ALL="$MAIN,zipf-gt-6m-q-s1-long,${RUNS// /,}"
step gt control $ALL --settings rtn-3,gptq-2,wanda-50,rtn-3+emb
step gt-notes
echo "== $(date '+%m-%d %H:%M') zipf sweep done" >> $L
