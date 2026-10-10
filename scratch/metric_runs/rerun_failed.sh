#!/usr/bin/env bash
# Reruns the 10 CSVs that failed on 2026-10-04 (old 2021 layouts), one at a time.
cd /g/Projects/TradingCardGameEmbeddings
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=/g/Projects/venvs/tcg-rocm/Scripts/python.exe
LOG=scratch/metric_runs/rerun_failed.log
: > $LOG
for f in AFR.PremierDraft MID.PremierDraft STX.PremierDraft STX.TradDraft VOW.PremierDraft VOW.QuickDraft; do
  $PY -u scripts/run_metrics.py --source seventeenlands_draft_data --raw-path data/raw/17lands/draft_data/$f.csv >> $LOG 2>&1
  echo "$(date -Is) draft $f exit=$?" >> scratch/metric_runs/rerun_summary.log
done
for f in AFR.PremierDraft AFR.TradDraft STX.PremierDraft STX.TradDraft; do
  $PY -u scripts/run_metrics.py --source seventeenlands_replay_data --raw-path data/raw/17lands/replay_data/$f.csv >> $LOG 2>&1
  echo "$(date -Is) replay $f exit=$?" >> scratch/metric_runs/rerun_summary.log
done
echo "$(date -Is) ALL DONE" >> scratch/metric_runs/rerun_summary.log
