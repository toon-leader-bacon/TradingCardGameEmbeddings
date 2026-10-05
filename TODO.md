# TODO

- [ ] **`Label = Any` in `src/schema/type_hints.py` crosses every
  dojo's label boundary untyped** (`TrainingDatum = Tuple[TrainingInput,
  Label]`). The module docstring and an inline comment already flag it
  as a known looseness ("typically a single scalar value, but could be
  a list of values"), which is itself the §1 (data & types: no
  untyped `Any` past a boundary) violation PRINCIPLES.md asks to avoid.
  A discriminated union or a per-dojo generic would make a mismatched
  label shape an illegal (unrepresentable) state instead of a runtime
  surprise - but it fans out to every dojo file, so it's a real design
  task, not a local fix. Flagged during a 2026-10-05 project-wide
  cleanup scan, not yet fixed.
- Use the test loss to adjust the loss function temperature in an
  intelligent way.
- Explore a "just-in-time" metric-generation architecture where a dojo
  drives its associated metric generator directly against the raw data
  at split time, instead of always going through an intermediate
  per-metric output file (parquet/JSONL) written by a separate scan
  step. A possible future alternative, not a planned change.
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
- [x] `SeventeenLandsDownloader` is now resumable (2026-09-29).
  `download()`'s batch loop skips a ref whose destination file already
  exists (reporting it as a success with that existing path) instead
  of re-downloading and re-extracting it, matching every other
  `Downloader` subclass's convention
  (`HearthstoneJsonDownloader.download_missing_builds()`,
  `PokemonTcgDataDownloader`, etc.) and making
  `scripts/run_data_retrieval.py`'s "every downloader is independently
  resumable" claim actually true for 17lands. `download_one()`, called
  directly, is unchanged and still always (re-)downloads and
  overwrites - the skip is `download()`'s own batch-loop behavior, same
  relationship `download_build()` has to
  `download_missing_builds()`.
