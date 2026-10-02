# seventeenlands

Metric classes over 17lands' public MTG draft/game/replay data exports
(`data/raw/17lands/`). Each raw export gets its own subdirectory here,
independent of the others - none share state or a scanner.

`rowwise_metric.py` - `RowwiseMetric(inner, frame_of)`, an Adapter that
runs a not-yet-vectorized row metric (`Metric[dict]`) inside a chunk
scan. It feeds each chunk's rows to the inner metric one dict at a time.
A failing row is skipped; once the chunk is done, one `RowFailures`
error reports how many rows failed, chained to the first row's
exception. The scanner logs and counts that error once per chunk. A
family uses the adapter only while it still has row metrics; no family
does today (game_data is fully vectorized, draft_data and replay_data
still scan rows).

## Containers

- **`draft_data/`** - six `Metric[dict]` metrics over per-pick draft
  CSVs (`data/raw/17lands/draft_data/<Set>.<EventType>.csv`). See
  [`draft_data/README.md`](draft_data/README.md).
- **`game_data/`** - eleven vectorized `Metric[GameDataChunk]` metrics
  over per-game CSVs (`data/raw/17lands/game_data/<Set>.<EventType>.csv`),
  scanned in typed numpy chunks (`GameDataChunk`). See
  [`game_data/README.md`](game_data/README.md).
- **`replay_data/`** - nine `Metric[dict]` metrics over per-game replay
  CSVs (`data/raw/17lands/replay_data/<Set>.<EventType>.csv`). See
  [`replay_data/README.md`](replay_data/README.md).

`BRAINSTORM.md` (this directory) is the original, now-superseded
brainstorm pass across all three raw exports - each subdirectory's own
`BRAINSTORM.md` has since sharpened its own section directly against
that export's raw data.
