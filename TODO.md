# TODO

- Implement better training loops that consider the test loss to determine when to stop.
- Use the test loss to adjust the loss function temperature in an intelligent way.
- Refactor CardIngestionStage.ingest to remove source_game as an argument. Concrete implementations should know which game they ingest already (typically a one-to-one mapping with a downloader), so a call like `MtgCardIngestionStage.ingest(..., GameId.POKEMON)` should be an illegal statement, not just an unenforced convention.
- Explore a "just-in-time" metric-generation architecture where a dojo drives its associated metric generator directly against the raw data at training/prepare_splits time, instead of always going through an intermediate per-metric output file (parquet/JSONL) written by a separate scan step. Raised while designing DeckOutcomeScanner (plans/deck_outcome_dojo.md) — the current architecture (a metric/scanner produces a file or collection of files, its associated dojo reads and mixes that data at training time) is being kept for now; this is a possible future alternative, not a planned change.
