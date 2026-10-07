# replay_data

Converts 17lands' raw per-game replay data
(`data/raw/17lands/replay_data/<Set>.<EventType>.csv`) into one partition
file per metric per CSV under `data/metrics/seventeenlands/replay_data/`
(see [`../README.md`](../README.md), "Partitions and slices"). Each CSV's
`deck_<name>` column suffixes and every Arena id in its per-half-turn
cells are matched against a `CardBinder` already populated for
`GameId.MTG`, Arena aliases included (see
[`../../../card_binder/README.md`](../../../card_binder/README.md)).

One row is one game. Besides the deck and the keys, a row has about
2,500 per-half-turn columns, `f"{actor}_turn_{N}_{field}"` (actor
`user` or `oppo`, N the actor's own turn counter), each holding a
`"|"`-delimited list of Arena card ids. The metrics read nine of those
fields; every other column is never read.

The scan is chunked and typed, like game_data's and draft_data's. The
shared `../chunk_scanner.py` streams a CSV in pyarrow record batches;
`ReplayDataChunkParser` turns each batch into one `ReplayDataChunk`, and
every metric's `accumulate()` receives that chunk. All nine metrics are
vectorized `Metric[ReplayDataChunk]`s: seven count tables and two row
streams. No metric splits a cell or visits a row in Python.

## Files

### Chunk, parser, scanner

- `replay_data_chunk.py` — the chunk's data types:
  - `Actor` (`USER` = 0, `OPPO` = 1; `label` is the column prefix) and
    `ReplayField`, the nine fields the metrics read, valued by column
    suffix: `creatures_cast`, `non_creatures_cast`, `creatures_attacked`,
    `creatures_blocking`, `creatures_unblocked`,
    `user_creatures_killed_combat`, `oppo_creatures_killed_combat`,
    `cards_discarded`, `cards_tutored`. `CAST_FIELDS` is the first two.
  - `TurnEvents`: one field's card occurrences over a chunk, long form.
    One entry per (row, actor, turn, card occurrence), so two copies in
    one cell are two entries. `codes` index the chunk's card table; an
    Arena id no card matches is kept with code `UNMATCHED` (-1), since
    the kill metric needs "something died" even for unknown cards.
    `matched()`, `for_actor()` and `select()` filter entries;
    `half_turn_ids()` encodes each entry's half-turn as one int,
    `(row * 2 + actor) * turn_span + turn`, which sorts by row, then
    user before oppo, then turn; `split_half_turn_ids()` decodes it.
    `rows_naming(codes, row_count)` lines the entries up against any
    card list (rows × list positions), and `combine()` concatenates
    fields.
  - `ReplayDataChunk`: `deck` (`../../../seventeenlands/zone_counts.py`'s `ZoneCounts`),
    `deck_codes` (each deck column's card code), `events` (a read-only
    mapping with a `TurnEvents` for every `ReplayField`; empty where the
    CSV lacks the field), `card_uuids` (the card table as of this
    chunk), `keys`, `num_turns` and `decks` (each row's deck, from
    `../../../seventeenlands/chunk_decks.py`). Construction checks row counts, that every
    field is present, that event rows and codes are in range, and that
    `deck_codes` name `deck.card_uuids`. `events_for(fields)` combines
    several fields.
- `replay_data_chunk_parser.py` — `ReplayDataChunkParser.from_header(
  header, card_binder, source_game)`, the only place that knows the
  column grammar. A field column's suffix must equal a `ReplayField`
  value exactly. Every field column is read as a string; per column,
  pyarrow splits each cell on `"|"` and flattens it, and empty tokens
  (empty cells) are dropped. Each distinct token is normalized to the
  Arena alias ledger's form (`str(int(float(token)))`, so `"104936"`
  and `"104936.0"` match alike) and matched once per CSV; a non-numeric
  token fails the CSV. `CardCodeTable` is the CSV's append-only card
  table, seeded with the deck columns' cards, so a code means the same
  card in every chunk. Deck counts are read as float32 and narrowed to
  int16 by `../../../seventeenlands/batch_columns.py`. A missing scalar column (`draft_id`,
  `match_number`, `game_number`, `num_turns`) fails the CSV; a null
  scalar raises naming the column and row. The oldest layout (AFR, STX:
  `game_index`, no `deck_` columns, no `match_number`) raises
  `UnsupportedCsvLayout`, so the run skips those CSVs.
- `replay_card_columns.py` — `ReplayCardColumns`: every matched
  `deck_`/`sideboard_` column as `(column, uuid)` pairs, plus
  `uuid_for_name(name)` and `uuid_for_arena_id(arena_id)`, each cached
  separately, and their `unmatched_names`/`unmatched_arena_ids`.
- `scanner.py` — `scan_replay_csv(raw_csv_path, metrics, parser,
  block_size)`: the shared `scan_chunked_csv()`, typed for
  replay_data. Failures are isolated per (metric, chunk) and logged as
  `METRIC FAILURE` lines.

### Count tables over card codes

- `code_tallies.py` — `CodeTallies(owner, tally_count)`: float64
  tallies per card code, grown as new codes appear. `add(tally, codes,
  weights=None)` is one `np.bincount`; `count_columns(card_uuids,
  names, keep)` returns the count-table columns.
- `code_count_table_metric.py` — `CodeCountTableMetric` (Template
  Method, abstract): owns the tallies, keeps the latest card table and
  writes the partition, one row per card whose first count is above
  zero. A subclass implements `_tally(chunk)` and
  `output_from_counts()`.
- `replay_turn_event_rate_metric.py` — `ReplayTurnEventRateMetric`:
  per card, `(total, hits)` over half-turns; the label is `hits /
  total`, `sample_count = total`. A subclass implements
  `_denominator_and_hit_codes(chunk)`.
- `replay_turn_event_rate_metrics.py` — its two concretes:
  - `CombatKillInvolvementRateMetric`: P(a creature died in combat
    that half-turn | the card fought that half-turn). The denominator
    is each half-turn's distinct matched cards in `creatures_attacked`
    ∪ `creatures_blocking`. A half-turn is a hit for all of them when
    either side's `creatures_killed_combat` names anything, matched or
    not.
  - `CombatDamagePushThroughRateMetric`: P(card in
    `creatures_unblocked` | card in `creatures_attacked`, same
    half-turn). Every matched attack entry counts in the denominator
    (two copies attacking count twice); each distinct (half-turn, card)
    both attacked and unblocked is one hit.
- `average_turn_cast_metric.py` — `AverageTurnCastMetric`: per card,
  `(count, turn_sum)` over every matched cast entry, either actor; the
  label is the average cast turn.
- `turns_to_game_end_after_cast_metric.py` —
  `TurnsToGameEndAfterCastMetric`: per (game, card cast that game),
  `num_turns` minus the card's first cast turn, the smallest turn on
  either actor's counter; partitions hold `(count, delta_sum)`.
  `num_turns` is on the same per-actor scale as the turn columns
  (checked against MSH.PremierDraft), so no conversion is needed.

### Count tables over deck columns

- `deck_event_rate_metric.py` — `DeckEventRateMetric`: a
  `../card_count_table_metric.py` `CardCountTableMetric` over the deck
  columns. Per card, `(in_deck, hit)`: games it was in the user's deck,
  and of those, games the user's `FIELDS` entries named it. Only the
  user's half-turns are read: a card in the user's own deck can only be
  cast, discarded or tutored by the user, and there is no
  `oppo_turn_N_cards_tutored` column at all.
- `cast_rate_metric.py` (`CAST_FIELDS`), `discard_rate_metric.py`
  (`cards_discarded`) and `tutor_target_rate_metric.py`
  (`cards_tutored`) — its three concretes. `DiscardRateMetric` assumes
  every user discard is self-inflicted (unverified).
  `TutorTargetRateMetric` is distinct from game_data's class of the
  same name.

### Row streams

- `combat_aggression_profile_metric.py` —
  `CombatAggressionProfileMetric`: one row per game, `draft_id`,
  `match_number`, `game_number`, `deck_uuid` (the user's deck, stored in
  the family deck box) and `combat_aggression_profile`: the mean number
  of matched attackers over the user's half-turns with at least one
  matched attacker (0.0 when there were none).
- `attacker_blocker_combat_outcome_metric.py` —
  `AttackerBlockerCombatOutcomeMetric`: one row per half-turn with at
  least one matched attacker, in (row, user before oppo, turn) order:
  `draft_id`, `match_number`, `game_number`, `actor`, `turn`,
  `attacker_uuids` and `blocker_uuids` (matched cards in cell order)
  and `net_kill_delta`: the defending side's matched combat kills minus
  the attacking side's own, so positive means the attacker traded up.

- `BRAINSTORM.md` — candidate metrics from this raw source not yet
  built.

## Card matching

Deck columns are matched once per CSV with
`card_lookup.uuid_for_name_or_front_face()`, the policy draft_data and
game_data use: a unique exact name match, else a unique split/MDFC
front-face match, else unmatched. An unmatched deck column is absent
from the deck.

Arena ids are matched with `card_binder.get_by_alias(source_game,
DataSource.ARENA, arena_id)`. `ScryfallCardIngestionStage` registers
each alias as `str(int)` with no decimal, which is why tokens are
normalized first. Sets whose cards lack Arena aliases in the binder
(PIO, SIR) match almost no event cells, so their event outputs are
nearly empty; their deck outputs are unaffected.

## How to run

From the project root (ROCm venv):

```
PYTHONPATH=. python scripts/run_metrics.py --source seventeenlands_replay_data \
    --raw-path data/raw/17lands/replay_data/LTR.TradSealed.csv
```

Each metric writes its partition,
`data/metrics/seventeenlands/replay_data/<OUTPUT_STEM>/<SET>/<Format>.parquet`,
and the family deck box is
`data/metrics/seventeenlands/replay_data/deck_box.db`; dojos build their
slice files from the partitions on first use.

On `LTR.TradSealed.csv` (32 MB, 5,001 games) the family runs in about
14 s including the binder load; the row implementation took 49 s. Its
outputs and deck box match the row implementation's exactly.
