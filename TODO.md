# TODO

- Use the test loss to adjust the loss function temperature in an
  intelligent way.
- Explore a "just-in-time" metric-generation architecture where a dojo
  drives its associated metric generator directly against the raw data
  at split time, instead of always going through an intermediate
  per-metric output file (parquet/JSONL) written by a separate scan
  step. A possible future alternative, not a planned change (raised in
  plans/deck_outcome_dojo.md).
- Move 17lands data retrieval onto the shared `Downloader` base class
  (`src/data_retrieval/downloader.py`) with the phase 1 / phase 2
  interface every other source uses; `SeventeenLandsDownloader` still
  stands alone.
