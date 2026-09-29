# TODO

- Use the test loss to adjust the loss function temperature in an
  intelligent way.
- Explore a "just-in-time" metric-generation architecture where a dojo
  drives its associated metric generator directly against the raw data
  at split time, instead of always going through an intermediate
  per-metric output file (parquet/JSONL) written by a separate scan
  step. A possible future alternative, not a planned change (raised in
  plans/deck_outcome_dojo.md).
- [x] Move 17lands data retrieval onto the shared `Downloader` base
  class (2026-09-29). `SeventeenLandsDownloader` now subclasses
  `Downloader`; `_run_phase_1()` is a thin no-arg wrapper around
  `download()` (which already meant "every known-valid file" when
  called with no args), returning `raw_data_dir`. `phase_2()` is not
  overridden (nothing further to fetch), so it inherits the base's
  no-op default. The constructor now matches every other source's
  `(rate_limiter=None, raw_data_dir=None)` shape - `rate_limiter` was
  previously required and keyword-only. `download()`/`download_one()`
  stay as the richer, filterable API for direct/scripted use, same
  relationship `HearthstoneJsonDownloader` has with its own phase-1
  helpers. `scripts/run_data_retrieval.py`'s special-cased
  `run_seventeenlands()` is now `_run_standard(SeventeenLandsDownloader())`
  like every other entry.
- `SeventeenLandsDownloader.download_one()` never checks whether a
  ref's destination file already exists before re-downloading and
  re-extracting it, unlike every other `Downloader` subclass
  (`HearthstoneJsonDownloader.download_missing_builds()`,
  `PokemonTcgDataDownloader`, etc.), which skip files already on disk.
  `scripts/run_data_retrieval.py`'s own module docstring claims "every
  downloader is independently resumable," but 17lands currently isn't
  - a re-run redownloads everything, despite 17lands runs being called
  out there as "10+ hour crawls." Fix: skip a ref in `download()`/
  `download_one()` when `_destination_path(ref)` already exists,
  mirroring `download_missing_builds()`'s check.
