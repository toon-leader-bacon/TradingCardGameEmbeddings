#!/usr/bin/env bash
# Run every queued run in queue.csv order, with a RAM/disk pre-flight before each.
# Usage (project root): bash experiments/2026-10-24_scale_profile/analysis/run_queue.sh
R=data/runs/2026-10-24_scale_profile
while true; do
  RUN=$(awk -F, 'NR>1 && $3=="queued"{print $1; exit}' $R/queue.csv)
  [ -z "$RUN" ] && { echo "queue empty $(date '+%F %T')"; exit 0; }
  FREE=$(powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '\r')
  echo "preflight $RUN: free RAM ${FREE} GB $(date '+%F %T')"
  if [ "$FREE" -lt 12 ]; then echo "HALT: free RAM ${FREE} GB < 12"; exit 2; fi
  bash experiments/2026-10-24_scale_profile/analysis/run_one.sh "$RUN"
  echo "finished $RUN exit=$? $(date '+%F %T')"
done
