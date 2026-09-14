# seventeenlands

Metric classes over 17lands' public MTG draft/game/replay data exports
(`data/raw/17lands/`). Each raw export gets its own subdirectory here,
independent of the others - none share state or a scanner.

## Containers

- **`draft_data/`** - six `Metric[dict]` metrics over per-pick draft
  CSVs (`data/raw/17lands/draft_data/<Set>.<EventType>.csv`). See
  [`draft_data/README.md`](draft_data/README.md).
- **`game_data/`** - eleven `Metric[dict]` metrics over per-game CSVs
  (`data/raw/17lands/game_data/<Set>.<EventType>.csv`). See
  [`game_data/README.md`](game_data/README.md).
- **`replay_data/`** - nine `Metric[dict]` metrics over per-game replay
  CSVs (`data/raw/17lands/replay_data/<Set>.<EventType>.csv`). See
  [`replay_data/README.md`](replay_data/README.md).

`BRAINSTORM.md` (this directory) is the original, now-superseded
brainstorm pass across all three raw exports - each subdirectory's own
`BRAINSTORM.md` has since sharpened its own section directly against
that export's raw data.
