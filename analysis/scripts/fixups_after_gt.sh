#!/bin/bash
# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT
# 2026-11-14: re-record ShortGPT states aligned by original block index (represent.py fix), then redo what used them.
# Waits for the ground-truth queue (one GPU job at a time).
cd "$(dirname "$0")/.."
R=${WU_RUNS:-$HOME/models/wu-runs}; mkdir -p "$R/gt"  # run directories and logs; set WU_RUNS to move them
L=$R/overnight.log
while pgrep -f "scripts/gt_after_overnight.sh" > /dev/null; do sleep 60; done
step() { echo "== $(date +%H:%M) $*" >> $L; pixi run "$@" >> $L 2>&1 || echo "!! FAILED: $*" >> $L; }
echo "== $(date +%H:%M) fixups: ShortGPT states by original block index" >> $L
for s in shortgpt-5 shortgpt-10; do step wu represent --model qwen3-8b --setting $s; done
step wu geometry --model qwen3-8b --settings awq-3,gptq-3,wanda-50,shortgpt-5
step wu geometry --model qwen3-8b --settings awq-3,gptq-3,wanda-50,shortgpt-5 --layer 19
echo "== $(date +%H:%M) fixups done" >> $L
