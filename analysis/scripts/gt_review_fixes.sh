#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# After the 14 Nov code review. (1) Single-block settings re-run: block L is now calibrated on base inputs.
# (2) Every compress report recomputed from the caches (AUC intervals now B = 1,000; d_excess, d_excess_which
# added). (3) control (bootstrap over runs and symbols), seeds, notes.
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/gt/queue.log
step() { echo "== $(date '+%m-%d %H:%M') $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
MAIN=zipf-gt-6m-q-s0-long
SPLICE=""; for b in 0 1 2 3 4 5; do SPLICE="$SPLICE,gptq-2@$b,wanda-50@$b"; done
step gt compress $MAIN --settings ${SPLICE#,}
for d in $R/gt/compress/*/; do r=$(basename $d); [ $r = $MAIN ] || step gt compress $r --settings base; done
RUNS=$(find $R/gt/compress -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | grep -v -- '-s[123]-long' | grep -v "^$MAIN$" | sort | tr '\n' ',')
step gt control $MAIN,zipf-gt-6m-q-s1-long,${RUNS%,} --settings rtn-3,gptq-2,wanda-50,rtn-3+emb
step gt seeds zipf-gt-6m-q-s0-long,zipf-gt-6m-q-s1-long,zipf-gt-6m-q-s2-long,zipf-gt-6m-q-s3-long
step gt-notes
echo "== $(date '+%m-%d %H:%M') review fixes done" >> $L
