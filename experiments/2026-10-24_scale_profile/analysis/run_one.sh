#!/usr/bin/env bash
# Launch one queued run with --profile: provenance, console capture, Windows
# counter samplers, queue.csv status.
# Usage (from project root): bash experiments/2026-10-24_scale_profile/analysis/run_one.sh <run> [attempt]
# queue.csv columns: run,config,status,attempts,started,ended,exit
# The body is one function, parsed in full before it runs.
main() {
set -u
RUN="$1"; ATTEMPT="${2:-1}"
R=data/runs/2026-10-24_scale_profile
A=experiments/2026-10-24_scale_profile/analysis
PY=/g/Projects/venvs/tcg-rocm/Scripts/python.exe
CFG=$(awk -F, -v r="$RUN" '$1==r{print $2}' $R/queue.csv)
LOG=$R/logs/$RUN.log; [ "$ATTEMPT" -gt 1 ] && LOG=$R/logs/$RUN.attempt$ATTEMPT.log
P=$R/provenance/$RUN; mkdir -p "$P" $R/logs
set_row() {
  "$PY" - "$R/queue.csv" "$RUN" "$@" <<'PYEOF'
import csv, sys
path, run, *kv = sys.argv[1:]
rows = list(csv.DictReader(open(path, newline="")))
for row in rows:
    if row["run"] == run:
        for item in kv:
            k, v = item.split("=", 1); row[k] = v
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
PYEOF
}
START=$(date '+%Y-%m-%d %H:%M:%S')
set_row status=running attempts=$ATTEMPT started="$START" ended= exit=
{
  echo "started: $START"; echo "attempt: $ATTEMPT"; echo "config: $CFG"
  echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "git status:"; git status --short
} > "$P/provenance.txt"
git diff HEAD > "$P/uncommitted.patch"
"$PY" -m pip freeze > "$P/pip_freeze.txt" 2>&1
"$PY" -c "import sys,torch,platform; print('python', sys.version); print('torch', torch.__version__); print('hip', torch.version.hip); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('platform', platform.platform()); print('cpu', platform.processor())" > "$P/hardware.txt" 2>&1
rm -f $R/logs/$RUN.finished
for S in sample_resources sample_utilization; do
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File $A/$S.ps1 "$RUN" > /dev/null 2>&1 &
done
PYTHONPATH=. "$PY" scripts/run_training.py "$CFG" --profile > "$LOG" 2>&1
EXIT=$?
END=$(date '+%Y-%m-%d %H:%M:%S')
echo "ended: $END"$'\n'"exit: $EXIT" >> "$P/provenance.txt"
if [ $EXIT -eq 0 ]; then ST=done; else ST=failed; fi
set_row status=$ST ended="$END" exit=$EXIT
touch $R/logs/$RUN.finished
echo "run=$RUN exit=$EXIT start=$START end=$END"
exit $EXIT
}
main "$@"; exit $?
