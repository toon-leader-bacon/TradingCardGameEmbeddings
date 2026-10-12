# 2026-10-12 15:00 · incident: launcher re-ran training after multi_contrastive finished

**What happened:**
- At about 09:05, while `multi_contrastive` was running, I edited the
  ignored launcher `data/runs/2026-10-11_model_by_diet_r2/run_one.sh`, so
  that it would start the resource samplers itself.
- Bash reads a script lazily, from a byte offset. When training finished
  normally (14:57:01; `run.log` ends with the usual final summary), bash
  resumed at its old offset inside the edited file:
  - it printed `run_one.sh: line 35: run: No such file or directory`;
  - it started both samplers again for `multi_contrastive`;
  - it ran `run_training.py` a second time. That exited 1 at 14:57:29 with
    `ValueError: run directory ... multi_contrastive already exists`, so
    no data was overwritten.

**Damage:**
- The console capture `logs/multi_contrastive.log` was overwritten by the
  second run. It now holds only that run's traceback.
  - The full project log survives: `multi_contrastive/run.log` (DEBUG),
    and the script's own INFO log `logs/run_training/2026-10-12_081838.log`,
    copied to `logs/multi_contrastive.script_log_copy.log`.
  - Lost: console-only output (raw stderr, progress bars).
- The resource and utilization samples for `multi_contrastive` were
  overwritten when the restarted samplers wrote fresh headers. Lost
  (about 07:00 h of RAM/VRAM samples, about 06:00 h of GPU/CPU utilization
  samples.
- The training's real exit code wasn't recorded. `queue.csv` first
  showed `failed`, exit 1, from the second run. It now shows `done` with
  an empty exit and a note.
- Training results are intact: `rounds.csv`, all checkpoints and
  `run.log` were written by the original process, which finished first.

**Classed as:** small issue (1 of 5 in this experiment): a bug in a
one-off script, caused by editing it mid-run.

**Action:** no retry needed, because the run completed. Fixes, all in
the ignored launcher files:
- `run_one.sh`: the body is wrapped in `main() { ... }; main "$@"; exit $?`,
  so bash parses the whole file before it runs anything, and an edit
  during a run can no longer change that run.
- `sample_resources.ps1` and `sample_utilization.ps1`: if the run's
  sample file already exists, they write to a new, timestamped file
  instead.
- I won't edit a running script again.

**Confound:** none. The launcher doesn't touch the training code. The
human's commits during the run (`86a0ecc`, `80877a5`) change only
docstrings, comments and TODO files in `src/` and `scripts/`, so the
next run (`multi_metric`, commit `80877a5`) runs the same code as
`multi_mixed` (`467d0e1`) and `multi_contrastive` (`4d97bdf`).
