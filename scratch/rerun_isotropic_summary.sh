#!/usr/bin/env bash
# Re-run isotropic_summary after its rc=139 (segfault) in the 2026-10-08 regen.
# faulthandler prints the native stack on a crash; a sampler logs memory every 30 s.
cd /g/Projects/TradingCardGameEmbeddings || exit 1
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
LOG_DIR=logs/regen_2026-10-08
LOG=$LOG_DIR/25b_metrics_isotropic_summary_rerun.log
MEM=$LOG_DIR/25b_metrics_isotropic_summary_rerun.mem.log
SUMMARY=$LOG_DIR/summary.log

echo "$(date '+%F %T') START 25b isotropic_summary_rerun (faulthandler)" >> "$SUMMARY"
t0=$(date +%s)
venv/Scripts/python.exe -X faulthandler scripts/run_metrics.py --source isotropic_summary > "$LOG" 2>&1 &
PID=$!
WINPID=$(cat /proc/$PID/winpid 2>/dev/null)
while kill -0 "$PID" 2>/dev/null; do
  echo "$(date '+%T') $(tasklist //FI "PID eq $WINPID" //FO CSV //NH 2>/dev/null | tail -1)" >> "$MEM"
  sleep 30
done
wait "$PID"
rc=$?
echo "$(date '+%F %T') END   25b isotropic_summary_rerun rc=$rc secs=$(( $(date +%s) - t0 )) tracebacks=$(grep -c Traceback "$LOG")" >> "$SUMMARY"
