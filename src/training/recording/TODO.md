# TODO

## `RunSampler`: rename, check modularity, review it properly

`run_sampler.py` (and its `csv_table.py`) came from the 2026-10-24
scale_profile experiment, where it ran alongside a timing profiler that
stayed on that experiment's branch. It was kept because per-run resource
tracking is useful generally, but it was carried over rather than
designed here.

- **Name:** "sampler" is vague, and close to the diet's samplers
  (`diet_sampler_for`). Pick a name that says what it records (VRAM and
  feature cache use over a run).
- **Modularity:** it reads VRAM through `torch.cuda` directly and the
  cache through `FeatureCacheStats`, with its column set fixed in one
  module. Decide whether the sources should be pluggable (system RAM, GPU
  utilization and CPU, which the experiments' `sample_*.ps1` scripts
  record today, since the venv has no psutil), and whether `CsvTable`
  belongs in `recording/` or should replace `CsvRunListener`'s own CSV
  writing.
- **Review:** give it a full read and a file-critic pass, as new code
  would get: the thread lifecycle, `scripts/run_training.py`'s wiring,
  and the tests.
