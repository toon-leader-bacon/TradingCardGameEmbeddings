#!/bin/bash
# Run each ablation until its round-0 encoder is written, then stop it (tree kill by Windows pid).
cd /g/Projects/TradingCardGameEmbeddings
RUNS=data/runs/2026-11-11_teacher_vs_real_pairs
# (The original waited here for the previous GPU job in the session to exit.)
mkdir -p $RUNS/logs
for name in ablation_T_noaug ablation_T_noclip; do
  echo "$(date +%T) start $name"
  PYTHONPATH=. G:/Projects/venvs/tcg-rocm/Scripts/python.exe scripts/run_training.py experiments/2026-11-11_teacher_vs_real_pairs/configs/$name.yaml > $RUNS/logs/$name.log 2>&1 &
  pid=$!
  winpid=$(cat /proc/$pid/winpid)
  until [ -f $RUNS/$name/round0000/encoder.pt ] && [ -f $RUNS/$name/round0000/manifest.json ]; do
    if ! kill -0 $pid 2>/dev/null; then echo "$(date +%T) $name exited early"; break; fi
    sleep 20
  done
  sleep 10
  taskkill //T //F //PID $winpid > /dev/null 2>&1
  wait $pid 2>/dev/null
  echo "$(date +%T) stopped $name after round 0"
done
