#!/usr/bin/env bash
cd /g/Projects/TradingCardGameEmbeddings
export PYTHONPATH=. PYTHONUNBUFFERED=1
/g/Projects/venvs/tcg-rocm/Scripts/python.exe -u scripts/run_training.py scratch/preflight/17lands_merged.yaml --check > scratch/preflight/17lands_merged.log 2>&1
code=$?
echo "$(date -Is) merged exit=$code" >> scratch/preflight/summary.log
