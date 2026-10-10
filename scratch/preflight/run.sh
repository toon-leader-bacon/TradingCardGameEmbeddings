#!/usr/bin/env bash
cd /g/Projects/TradingCardGameEmbeddings
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=/g/Projects/venvs/tcg-rocm/Scripts/python.exe
for g in small pack pool; do
  /usr/bin/time -v true >/dev/null 2>&1
  $PY -u scripts/run_training.py scratch/preflight/17lands_$g.yaml --check > scratch/preflight/17lands_$g.log 2>&1
  echo "$(date -Is) $g exit=$?" >> scratch/preflight/summary.log
done
