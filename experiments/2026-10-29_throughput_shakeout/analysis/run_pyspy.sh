#!/usr/bin/env bash
# Run one arm under py-spy record (raw collapsed stacks, thread ids, 50 Hz,
# non-blocking), with OMP_NUM_THREADS=1, provenance and the Windows samplers.
# Usage (project root): bash experiments/2026-10-29_throughput_shakeout/analysis/run_pyspy.sh <arm> <py-spy flag>
#   e.g. run_pyspy.sh pyspy_all --idle ; run_pyspy.sh pyspy_gil --gil
main() {
set -u
RUN="$1"; MODE="$2"
R=data/runs/2026-10-29_throughput_shakeout
A=experiments/2026-10-29_throughput_shakeout/analysis
PY=/g/Projects/venvs/tcg-rocm/Scripts/python.exe
SPY=/g/Projects/venvs/tcg-rocm/Scripts/py-spy.exe
CFG=experiments/2026-10-29_throughput_shakeout/configs/$RUN.yaml
P=$R/provenance/$RUN; mkdir -p "$P" $R/logs $R/pyspy
START=$(date '+%Y-%m-%d %H:%M:%S')
{
  echo "started: $START"; echo "config: $CFG"; echo "py-spy: $($SPY --version) record $MODE --subprocesses --threads --rate 50 --nonblocking --format raw"
  echo "env: OMP_NUM_THREADS=1"
  echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "git status:"; git status --short
} > "$P/provenance.txt"
rm -f $R/logs/$RUN.finished
for S in sample_resources sample_utilization; do
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File $A/$S.ps1 "$RUN" > /dev/null 2>&1 &
done
OMP_NUM_THREADS=1 PYTHONPATH=. "$SPY" record $MODE --subprocesses --threads --rate 50 --nonblocking --format raw \
  -o $R/pyspy/$RUN.txt -- "$PY" scripts/run_training.py "$CFG" > $R/logs/$RUN.log 2>&1
EXIT=$?
echo "ended: $(date '+%Y-%m-%d %H:%M:%S')"$'\n'"exit: $EXIT" >> "$P/provenance.txt"
touch $R/logs/$RUN.finished
echo "run=$RUN exit=$EXIT"
exit $EXIT
}
main "$@"; exit $?
