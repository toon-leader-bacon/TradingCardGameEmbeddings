# seventeenlands

Metric classes over 17lands' public MTG draft/game/replay data exports
(`data/raw/17lands/`). Each raw export gets its own subdirectory here,
independent of the others: none share state, though they share the
chunk-scan machinery below. The typed chunk pieces every family builds
on (`ZoneCounts`, batch column readers, `GameKeys`/`ChunkDecks`, and the
whole game_data parser) live in
[`../../seventeenlands/`](../../seventeenlands/README.md), shared with
the deck box extraction stage.

Every family scans its CSVs in typed chunks, sharing:

- `chunk_scanner.py` - the `ChunkParser` Protocol and
  `scan_chunked_csv()`: one read pass per CSV, each pyarrow batch parsed
  once and handed to every metric, failures isolated per (metric,
  chunk) and logged as `METRIC FAILURE` lines. A parser factory raises
  `UnsupportedCsvLayout` for a CSV layout its family deliberately does
  not read; `scripts/run_metrics.py` logs that CSV as `CSV SKIPPED`,
  not as a failure.
- `deck_box_path.py` - `REPLAY_DATA_DECK_BOX_PATH`: replay_data's
  private deck box, written by `run_metrics.py` and read by its deck
  dojo. game_data's deck ids are the canonical MTG box's
  (`data/final/decks/mtg.db`), so game_data keeps no box.
- `card_column_tallies.py` - `CardColumnTallies`: per-column tallies
  summed per card.
- `card_count_table_metric.py` - `CardCountTableMetric[ChunkT]`
  (Template Method, abstract): a per-card count table tallied over a
  chunk's card columns. A subclass names the columns
  (`_column_card_uuids(chunk)`) and their per-column increments
  (`_increments(chunk)`). game_data's `GameCardCountTableMetric` and
  replay_data's `DeckEventRateMetric` build on it.

## Partitions and slices

Every 17lands metric writes one partition file per source CSV:

    data/metrics/seventeenlands/<family>/<OUTPUT_STEM>/<SET>/<Format>.parquet

The path is the only record of a file's set and format.
`partition.py`'s `SeventeenLandsPartition` builds it (`path()`), reads
it back (`from_path()`), and `find_partitions()` lists a metric's files.

A dojo trains on a **slice**: `data_slice.py`'s
`SeventeenLandsSlice(expansions=None, formats=None)`, a set filter and a
format filter (None means every one). A slice is a filter, never a
grouping: it always yields one row per key, so training a metric per
format means one dojo per format slice.

A metric is one of two kinds (`sliced_metric.py`; `is_count_table()`
tells them apart):

- **`CountTableMetric`** - each partition holds raw, summable counts
  (`COUNT_COLUMNS`) per key (`KEY_COLUMNS`, e.g. `nocab_uuid` or
  `deck_uuid`), written by `count_table.py`'s `write_count_table()`,
  which checks the columns against the metric's declaration. A slice
  sums the counts per key, then the metric's `output_from_counts()`
  labels them (`count_table.ratio_output()` for the common ratio). The
  finished table is keys, label and `sample_count`, except where a
  metric regroups its keys into the shape its dojo reads
  (`PickNumberDecayCurveMetric`: one row per card, per-pick-number
  lists).
  Summing then dividing is exact for any slice. A metric with
  `HAS_BASELINE` also writes one null-key baseline row per partition;
  the slice sums those into its own baseline (a card's game length
  minus the slice's mean game length).
- **`RowStreamMetric`** - one row per game (or pick, or half-turn); a
  slice stacks its partitions and adds `set` and `format` columns.

`slice_file.py`'s `SeventeenLandsSliceFile(metric).build(slice)` writes
the slice's file, `<family>/slices/<OUTPUT_STEM>.<slice name>.parquet`,
and returns its path; it is an ordinary metric parquet, carrying the
partitions' binder version (they must all agree). It is rebuilt only
when its fingerprint changes: each partition's path, size and
modification time, the metric's `LABEL_VERSION` and the module's
`SLICE_BUILD_VERSION` (bump the first on any label change, the second
on any change to how slices are built). Writes go through a temporary
file, so a failed build never leaves a half-written slice.
`finished_count_table(metric, tables)` is the sum-and-label step on its
own.

Dojos: `src/dojos/seventeenlands/sliced_dojos.py`'s
`seventeenlands_training_path(metric, data_slice, path)` returns an
explicit path if given, else the built slice file. Its two bases,
`SeventeenLandsCardLabelDojo` and `SeventeenLandsDeckRegressionDojo`,
take `data_slice` (default: everything), so a dojo's default name and
split prefix is the slice file's stem (`win_rate_when_in_deck.all`).
`FileManagerParquet` and the generic dojo bases are unchanged.

Every 17lands metric is sliced: game_data (9 count tables, 3 row
streams), draft_data (4 count tables, 2 row streams) and replay_data (7
count tables, 2 row streams).

## Containers

- **`draft_data/`** - six vectorized `Metric[DraftDataChunk]` metrics
  (four count tables, two row streams) over per-pick draft CSVs
  (`data/raw/17lands/draft_data/<Set>.<EventType>.csv`). See
  [`draft_data/README.md`](draft_data/README.md).
- **`game_data/`** - twelve vectorized `Metric[GameDataChunk]` metrics (nine
  count tables, three row streams)
  over per-game CSVs (`data/raw/17lands/game_data/<Set>.<EventType>.csv`),
  scanned in typed numpy chunks (`GameDataChunk`). See
  [`game_data/README.md`](game_data/README.md).
- **`replay_data/`** - nine vectorized `Metric[ReplayDataChunk]`
  metrics (seven count tables, two row streams) over per-game replay
  CSVs (`data/raw/17lands/replay_data/<Set>.<EventType>.csv`), each
  per-half-turn field read as long-form `TurnEvents`. See
  [`replay_data/README.md`](replay_data/README.md).

`BRAINSTORM.md` (this directory) is the original, now-superseded
brainstorm pass across all three raw exports - each subdirectory's own
`BRAINSTORM.md` has since sharpened its own section directly against
that export's raw data.
