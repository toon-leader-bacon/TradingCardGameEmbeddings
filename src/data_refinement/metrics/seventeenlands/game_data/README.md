# game_data

Converts 17lands' raw per-game data
(`data/raw/17lands/game_data/<Set>.<EventType>.csv`) into training-data
parquet files under `data/metrics/seventeenlands/game_data/`, matching
each CSV's `opening_hand_<name>`/`drawn_<name>`/`tutored_<name>`/
`deck_<name>`/`sideboard_<name>` column suffixes against a `CardBinder`
already populated for `GameId.MTG` (see
[`../../../card_binder/README.md`](../../../card_binder/README.md)).

The scan is chunked and typed. `scan_game_csv` streams a CSV in pyarrow
record batches. `GameDataChunkParser` turns each batch into one
`GameDataChunk` (numpy arrays), and every metric's `accumulate()`
receives that chunk. The card-average metrics are vectorized
`Metric[GameDataChunk]`s. The other metrics here are still row metrics
(`Metric[dict]`, one row per call); the driver runs them inside the
chunk scan through `RowwiseMetric`
([`../rowwise_metric.py`](../rowwise_metric.py)). Porting them is
tracked in `plans/seventeenlands_chunk_scan.md`.

## Files

- `game_data_chunk.py` — `GameZone` (the five card-column families,
  valued by header prefix), `ZoneCounts` (one zone's card uuids, one
  per matched header column and possibly repeating, plus an int16
  `(rows, columns)` count matrix; `present()` is `counts > 0`), and
  `GameDataChunk` (every zone's `ZoneCounts`, typed `won`/`on_play`/
  `num_turns` arrays, and an optional pandas `source_frame` for wrapped
  row metrics). The chunk checks at construction that every zone is
  present and every field has the same row count. `source_frame_of`
  returns the frame, and raises if the parser was built without one.
- `game_data_chunk_parser.py` — `GameDataChunkParser.from_header(header,
  card_binder, source_game, keep_source_frame)`. It is the only place
  that knows the CSV's column names. Card matching is
  `GameCardColumns.from_header()`'s, unchanged. `needed_columns()` and
  `column_types()` tell the scanner what to read, with card counts read
  as int16. `parse(batch)` builds a `GameDataChunk`: a null count cell
  becomes 0, and a null `won`/`on_play`/`num_turns` raises `ValueError`
  naming the column and row. It keeps the batch as `source_frame` only
  when `keep_source_frame` is set (the family still has row metrics).
- `game_card_columns.py` — `GameCardColumns`: built once per metric
  instance from that metric's own `(card_binder, header, source_game)`.
  Parses every `opening_hand_<name>`/`drawn_<name>`/`tutored_<name>`/
  `deck_<name>`/`sideboard_<name>` header column and matches its
  `<name>` suffix against `card_binder` (see "Card-name matching"
  below), exposing the results as `opening_hand_columns`/
  `drawn_columns`/`tutored_columns`/`deck_columns`/`sideboard_columns`
  (`list[tuple[str, UUID]]`) plus `uuid_for_name(name)` and
  `present_uuids(row, columns)`. Every column here is a per-game copy
  **count** (`deck_<name>` sums to 40, `opening_hand_<name>` to 7), not
  a per-copy list entry — `present_uuids()` samples on presence
  (count > 0) exactly once per qualifying card, never once per copy.
- `scanner.py` — `scan_game_csv(raw_csv_path, metrics, parser,
  block_size)` streams the CSV with `pyarrow.csv.open_csv`, reading only
  `parser.needed_columns()`. It parses each batch once and hands the
  chunk to every metric, then finalizes them all. It never touches the
  `CardBinder` or a `DeckBox`.
  - Failures are isolated per (metric, chunk) through `call_isolated`
    ([`../../isolated_call.py`](../../isolated_call.py)). Each failure
    logs an ERROR line starting `METRIC FAILURE` that names the metric,
    the CSV and the chunk, with its traceback. The scan ends with one
    `METRIC FAILURE summary` line per failing metric.
  - A metric that fails partway through `accumulate()` may be left
    half-tallied, so its output for that CSV must not be trusted.
- `game_card_average_metric.py` — `GameCardAverageMetric` (Template
  Method, abstract, accumulation, `Metric[GameDataChunk]`). It keeps a
  per-card average of one per-game value over every game the card was
  present in, in the subclass's `ZONE`. It is built from
  `(version_metadata, output_path=None)`; the driver computes the binder
  version once per run.
  - Per chunk it tallies `value_sum` and `count` per matched column, in
    one array operation over `zone.present()`. A chunk whose column
    layout differs from the first raises `ValueError`.
  - `finalize()` sums the columns per card uuid, so two columns naming
    one card both count. It writes `nocab_uuid`, `LABEL_COLUMN` and
    `sample_count` for every card seen at least once. A CSV with no rows
    writes a zero-row file with that full schema.
  - Subclasses fix `LABEL_COLUMN`, `DEFAULT_OUTPUT_PATH` and `ZONE`, and
    implement `_values(chunk)` (one value per row). They may override
    `_extra_accumulate(chunk)` (no-op by default) and
    `_label(value_sum, count)` (default: the average).
- `game_card_average_metrics.py` — `GameCardWinRateMetric` (abstract;
  `_values` is `won` as 1.0/0.0) and its three concretes:
  - `WinRateWhenInDeckMetric`: `P(won | card in deck_<name>)`;
  - `OpeningHandWinRateMetric`: `P(won | card in opening_hand_<name>)`;
  - `DrawnWinRateMetric`: `P(won | card in drawn_<name>)`, a card seen
    at any point in the game, opening hand or not.
- `game_length_association_metric.py` — `GameLengthAssociationMetric`,
  a `GameCardAverageMetric` over `deck_<name>` that averages
  `num_turns`.
  - `_extra_accumulate()` keeps a format-wide (turn sum, game count)
    baseline over every row.
  - `_label()` subtracts that baseline from each card's own average.
  - It overrides nothing else.
- `on_play_win_rate_delta_metric.py` — `OnPlayWinRateDeltaMetric`
  (standalone accumulation): per card, `P(won | in deck, on_play) -
  P(won | in deck, on_draw)`. Not a `GameCardAverageMetric` subclass —
  its tally is two-dimensional per card (keyed by `(card_uuid,
  on_play)`), not that base's single running sum/count. Nullable
  output: a side never seen for a card writes `None` for that card's
  delta rather than guessing.
- `tutor_target_rate_metric.py` — `TutorTargetRateMetric` (standalone
  accumulation): per card, `P(tutored | in deck)` — among games where a
  card was in the deck, how often a tutor effect actually fetched it.
  `sample_count` is required output, not optional, since most cards'
  true rate is near zero.
- `game_deck_label_metric.py` — `GameDeckLabelMetric` (Template Method,
  abstract, streaming): one game's constructed deck (`deck_<name>`,
  referenced by `deck_uuid`, never embedded) paired with a single
  scalar label already present on that row. A subclass fixes
  `LABEL_COLUMN`/`LABEL_TYPE`/`DEFAULT_OUTPUT_PATH` and implements
  `_label_for_row()`; every other step (deck identification, hashing
  via `deck_uuid_from_cards()`, writing into a shared `DeckBox`, output
  row write) is shared. `deck_box` is a required constructor parameter
  here — unlike `sts_gg/card_average_metric.py`'s `CardAverageMetric`,
  a metric with no use for a deck box in this container simply doesn't
  declare the parameter at all (see every class above, none of which
  take `deck_box`).
- `game_deck_label_metrics.py` — three concrete `GameDeckLabelMetric`
  subclasses: `DeckWinPredictionMetric` (`won`, `pa.bool_()`),
  `DeckGameLengthPredictionMetric` (`num_turns`, `pa.int64()`),
  `DeckRankTierPredictionMetric` (`rank`, `pa.string()` — known tier
  vocabulary `bronze`/`silver`/`gold`/`platinum`/`diamond`/`mythic` plus
  `OTHER_LABEL` as a safety net for an unseen value).
- `on_play_win_rate_sensitivity_by_deck_metric.py` —
  `OnPlayWinRateSensitivityByDeckMetric`: the deck-level mirror of
  `OnPlayWinRateDeltaMetric`, aggregated per `deck_uuid` instead of per
  card. Accumulation, not streaming (the label needs cross-row
  aggregation across every game sharing an identical deck), so it does
  **not** subclass `GameDeckLabelMetric` — it duplicates that class's
  deck-identification-and-hashing steps directly instead of sharing
  them across the streaming/accumulation split (the same call
  `draft_data`'s `pack_to_pick_choice_set_metric.py`/
  `pool_conditioned_pick_metric.py` already made for a similar pair).
  Same nullable-output convention as `OnPlayWinRateDeltaMetric`.
- `tutor_target_pool_metric.py` — `TutorTargetPoolMetric` (streaming,
  fan-out): given one game's full draft pool (`deck_<name>` ∪
  `sideboard_<name>`), writes one output row per pool card labelling
  whether it appears in `tutored_<name>` that game — the first metric
  in this codebase where one `accumulate()` call writes zero or more
  output rows rather than exactly one. Not a deck-input metric: its
  identity is the per-game triple plus a `pool_card_uuid`, never a
  `deck_uuid` — the pool here (deck ∪ sideboard) is a different card
  multiset than any `GenericDeck` this container mints elsewhere, so it
  never calls `deck_uuid_from_cards()` and takes no `deck_box`.
- `BRAINSTORM.md` — candidate metrics from this raw source not yet
  built (this container currently implements the eleven ideas on its
  "Human Review Short List"; other candidates from its longer lists
  remain future work).

## Card-name matching

`card_lookup.uuid_for_name_or_front_face()` (`src/data_refinement/card_binder/card_lookup.py`) matches a bare card name: a unique exact
`get_by_name()` match, else a unique split/MDFC front-face match (17lands'
column names use only a card's front face; Scryfall names it `"A // B"`),
else unmatched. Ambiguity is never guessed at. `GameCardColumns` applies it to every card column suffix.

`uuid_for_name()` caches every lookup (hit or miss) so the same name is
never queried against `card_binder` twice; `unmatched_names` exposes
every name this instance's cache has no uuid for, for a caller to log.
Unlike `draft_data`, `game_data` has no per-row cell value analogous to
`pick` — every name `GameCardColumns` ever looks up comes from a header
column suffix, matched once at construction time; an unmatched column
is simply absent from all five `*_columns` lists.

## Card-binder and deck-box access shape

The vectorized metrics (`GameCardAverageMetric` and its subclasses)
never see the binder or the header. Card matching lives in
`GameDataChunkParser`, which the driver builds once per CSV, and their
constructor takes the run's `MetricVersionMetadata`.

The row metrics still take `(card_binder, header, source_game,
output_path=None)`. The four deck-input metrics also take a required
`deck_box`: `game_deck_label_metrics.py`'s three concretes and
`OnPlayWinRateSensitivityByDeckMetric`. Each builds its own private
`GameCardColumns` via `GameCardColumns.from_header()`.

The driver (`scripts/run_metrics.py`) owns everything shared:
- it reads each CSV's header once;
- it passes one metrics-private `DeckBox` to every deck-input metric,
  so identical decks dedupe, and saves it after the last CSV;
- it wraps the row metrics in `RowwiseMetric`.

The per-game identifier this container settles on, since no single raw
column is a unique key: the composite `(draft_id: str, match_number:
int, game_number: int)`, read directly off each row as three separate
output columns — mirroring `draft_data`'s own `draft_id`/`pack_number`/
`pick_number` convention rather than one joined string.

## How it works

`GameCardAverageMetric` and its four subclasses share one chunk-level
`accumulate()`:
1. tally this subclass's `ZONE` presence against `_values(chunk)`, per
   column;
2. call the optional `_extra_accumulate()` hook.

`finalize()` groups the tallies by card and applies `_label()`.
`GameLengthAssociationMetric` changes only the hooks, never
`accumulate()` or `finalize()`.

On `KTK.TradDraft.csv` (151 MB, 56k games), the four card-average
metrics scan in 2.8 s against 27.0 s for the earlier row implementation.
Their outputs are identical to that implementation's: the parity check
is `scripts/compare_metric_outputs.py`.

`GameDeckLabelMetric` and its three `game_deck_label_metrics.py`
subclasses are streaming instead — one row already carries a complete
example (the deck's cards plus a scalar label), so `accumulate()`
buffers it into an open [`ParquetBuilder`](../../parquet_builder.py)
and `finalize()` only closes that builder, mirroring
[`../../sts_gg/deck_label_metric.py`](../../sts_gg/deck_label_metric.py)'s
shape. `OnPlayWinRateDeltaMetric`/`TutorTargetRateMetric` are standalone
accumulation metrics with their own two-dimensional or ratio-shaped
tallies; `OnPlayWinRateSensitivityByDeckMetric` is `OnPlayWinRateDeltaMetric`'s
accumulation-shaped deck-level mirror; `TutorTargetPoolMetric` is
streaming but fans a single input row out to zero or more output rows
(one per pool card), calling `write_row()` once per fanned-out row
rather than the one-row-per-call shape every other streaming metric in
this codebase uses.

## How to run

From the project root (ROCm venv):

```
PYTHONPATH=. python scripts/run_metrics.py --source seventeenlands_game_data \
    --raw-path data/raw/17lands/game_data/KTK.TradDraft.csv
```

Each metric writes `data/metrics/seventeenlands/game_data/<SET>/<Format>/<name>.parquet`.
The family deck box is `data/metrics/seventeenlands/game_data/deck_box.db`.

Parity check: write the outputs to a scratch root with `--output-root`,
then diff them against the existing outputs:

```
PYTHONPATH=. python scripts/run_metrics.py --source seventeenlands_game_data \
    --raw-path data/raw/17lands/game_data/KTK.TradDraft.csv --output-root scratch/parity
PYTHONPATH=. python scripts/compare_metric_outputs.py \
    --reference data/metrics/seventeenlands/game_data/KTK/TradDraft \
    --candidate scratch/parity/seventeenlands/game_data/KTK/TradDraft
```
