#!/usr/bin/env bash
# LR pilot queue: six runs in sequence, arms alternating so both share the
# variant-encoding cost of the first runs. One console log per run next
# to its run directory. Continues past a failed run; prints each exit code.
# Run from the project root.
set -u
PY="G:/Projects/venvs/tcg-rocm/Scripts/python.exe"
EXP=experiments/2026-11-11_teacher_vs_real_pairs
OUT=data/runs/2026-11-11_teacher_vs_real_pairs
for lr in 1e-04 3e-04 1e-03; do
  for arm in R T; do
    name="pilot_${arm}_lr${lr}"
    start=$(date +%H:%M:%S)
    PYTHONPATH=. "$PY" scripts/run_training.py "$EXP/configs/$name.yaml" > "$OUT/$name.console.log" 2>&1; rc=$?
    echo "$name start=$start end=$(date +%H:%M:%S) exit=$rc"
  done
done
