# draft_data

Converts 17lands' raw per-pick draft data
(`data/raw/17lands/draft_data/<Set>.<EventType>.csv`) into one partition
file per metric per CSV under `data/metrics/seventeenlands/draft_data/`
(see [`../README.md`](../README.md), "Partitions and slices"), matching
each CSV's `pack_card_<name>`/`pool_<name>` column suffixes and each
row's `pick` (and `pick_2`) cell against a `CardBinder` already
populated for `GameId.MTG` (see
[`../../../card_binder/README.md`](../../../card_binder/README.md)).

One row is one pick: the pack the drafter saw, the pool drafted so far
and the card(s) taken. No metric needs another row of the same draft, so
a chunk boundary inside a draft changes nothing.

The scan is chunked and typed, like game_data's. The shared
`../chunk_scanner.py` streams a CSV in pyarrow record batches;
`DraftDataChunkParser` turns each batch into one `DraftDataChunk`, and
every metric's `accumulate()` receives that chunk. All six metrics are
vectorized `Metric[DraftDataChunk]`s: four count tables and two row
streams.

## Files

### Chunk, parser, scanner

- `draft_data_chunk.py` — the chunk's data types:
  - `DraftStratum`: a value a take rate can be stratified by
    (`pack_number`, `pick_number`, `rank`).
  - `DraftPicks`: each row's taken card(s). `is_pick_two` is read from
    the `pick_2` cell itself (non-empty: a PickTwo row, matched or not).
    `first_codes`/`second_codes` code the picks against the pack's card
    code table (`NO_PICK` = -1 for an empty or unmatched pick), and
    `first_uuids` holds the first pick's uuid as a str (None when
    unmatched).
  - `DraftDataChunk`: `pack` and `pool` (`../zone_counts.py`'s
    `ZoneCounts`), `pack_column_codes`, `picks`, `draft_id`,
    `pack_number`, `pick_number` and `rank` (`""` when unranked). It
    checks every per-row field's length at construction. `picked()` is
    the (rows, pack columns) mask of taken cells: present and coded as
    the row's first or second pick, one comparison over the chunk.
- `draft_data_chunk_parser.py` — `DraftDataChunkParser.from_header(
  header, card_binder, source_game)`, the only place that knows the
  CSV's column names. Card matching is `DraftCardColumns`' (below),
  done once per CSV; the pack card code table and each pack column's
  code are built once too. Card counts are read as float32 (some
  exports write `1.0`) and narrowed to int16 by `../batch_columns.py`.
  Each distinct pick name is matched once per chunk. A missing scalar
  column fails the CSV; a null scalar raises naming the column and row.
  The rank is read from `rank`, else `user_rank` (older exports, MID
  and VOW); a CSV with neither (AFR, STX) reads every row as unranked
  (`""`).
- `pack_pool_columns.py` — `DraftCardColumns`: every matched
  `pack_card_`/`pool_` column as `(column, uuid)` pairs, plus
  `uuid_for_name(name)` (cached) and `unmatched_names`.
- `scanner.py` — `scan_draft_csv(raw_csv_path, metrics, parser,
  block_size)`: the shared `scan_chunked_csv()`, typed for draft_data.
  Failures are isolated per (metric, chunk) and logged as `METRIC
  FAILURE` lines.

### Count tables (take rates)

- `keyed_card_tallies.py` — `KeyedCardTallies(owner, tally_count)`:
  one float64 `CardColumnTallies` (`../card_column_tallies.py`) per
  stratum key. Per chunk, rows are stably sorted by key (each stratum
  factorized separately, so int and str strata mix) and each key's run
  is summed per column with one `np.add.reduceat`. `count_columns(
  key_names, count_names, keep)` returns the count-table columns.
- `pack_card_tally_metric.py` — `PackCardTallyMetric` (Template Method,
  abstract): per (card, stratum key), `(in_pack, picked)`; the label is
  `take_rate = picked / in_pack`, `sample_count = in_pack`. `KEY_COLUMNS`
  (`nocab_uuid` then the stratum names) is the single source of a
  subclass's stratification. Hooks: `_eligible_rows(chunk)` (default
  every row) and `_stratum_values(chunk, stratum)`. A PickTwo row's
  second pick counts as picked too.
- `pack_card_tally_metrics.py` — `CardTakeRateMetric` (by pack and pick
  number), `FirstPickRateMetric` (pack 0 pick 0 rows only; keyed by the
  card alone) and `RankStratifiedTakeRateMetric` (by pack, pick and
  rank; ranked rows only).
- `pick_number_decay_curve_metric.py` — `PickNumberDecayCurveMetric`:
  partitions hold `(in_pack, picked)` per (card, pick_number), the pick
  number clamped to `MAX_BUCKET_COUNT - 1`; `output_from_counts`
  regroups a slice into one row per card with
  `take_rate_by_pick_number` (null where a bucket is unseen) and
  `sample_count_by_pick_number`, each `max(pick_number) + 1` long.

### Row streams

- `draft_choice_stream_metric.py` — `DraftChoiceStreamMetric`
  (Template Method): one output row per single pick, `draft_id`,
  `pack_number`, `pick_number`, `pool_uuids` (when `INCLUDES_POOL`),
  `pack_option_uuids` and `pick_uuid` (null when unmatched), written per
  chunk with `ParquetBuilder.write_columns()`. List columns are built
  from a zone's present matrix with `np.nonzero`, in header order. PickTwo
  rows are skipped (a two-card pick is not a one-of-N choice).
- `pack_to_pick_choice_set_metric.py` — `PackToPickChoiceSetMetric`:
  the pack's options and the pick.
- `pool_conditioned_pick_metric.py` — `PoolConditionedPickMetric`: the
  same plus the pool so far (`INCLUDES_POOL`); a card held twice is
  listed once.

- `BRAINSTORM.md` — candidate metrics from this raw source not yet
  built.

## Card-name matching

`card_lookup.uuid_for_name_or_front_face()`
(`src/data_refinement/card_binder/card_lookup.py`) matches a bare card
name: a unique exact `get_by_name()` match, else a unique split/MDFC
front-face match (17lands' column names use only a card's front face;
Scryfall names it `"A // B"`), else unmatched. Ambiguity is never
guessed at. A pick cell's value is drawn from the same per-set card pool
as the header suffixes, so the one `DraftCardColumns` cache serves both.
An unmatched column is absent from every zone.

## How to run

From the project root (ROCm venv):

```
PYTHONPATH=. python scripts/run_metrics.py --source seventeenlands_draft_data \
    --raw-path data/raw/17lands/draft_data/TMT.TradDraft.csv
```

Each metric writes its partition,
`data/metrics/seventeenlands/draft_data/<OUTPUT_STEM>/<SET>/<Format>.parquet`;
dojos build their slice files from the partitions on first use.

On `TMT.TradDraft.csv` (156 MB, 160k picks) the family runs in about
26 s including the binder load; the row implementation scanned it in
41 s. Its outputs match the row implementation's exactly.
