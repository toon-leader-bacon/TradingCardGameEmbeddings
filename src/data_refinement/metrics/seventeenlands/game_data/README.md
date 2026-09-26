# game_data

Converts 17lands' raw per-game data
(`data/raw/17lands/game_data/<Set>.<EventType>.csv`) into training-data
parquet files under `data/metrics/seventeenlands/game_data/`, matching
each CSV's `opening_hand_<name>`/`drawn_<name>`/`tutored_<name>`/
`deck_<name>`/`sideboard_<name>` column suffixes against a `CardBinder`
already populated for `GameId.MTG` (see
[`../../../card_binder/README.md`](../../../card_binder/README.md)).
Every metric here satisfies the shared `Metric[dict]` Protocol
([`../../metric.py`](../../metric.py)).

## Files

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
- `scanner.py` — `scan_game_csv(raw_csv_path, metrics, chunk_size)`
  drives every metric in a list over one chunked read pass of a
  game_data CSV, handing each metric one row at a time and isolating
  one metric's `accumulate()`/`finalize()` failure (logged, not raised)
  from every other metric in the list. Never touches `GameCardColumns`
  or `DeckBox` itself — each metric already built its own
  `GameCardColumns` before reaching this function, and a deck-input
  metric already received its `DeckBox` directly.
- `game_card_average_metric.py` — `GameCardAverageMetric` (Template
  Method, abstract, accumulation): shared per-card running
  `(value_sum, total_count)` average, driven off whichever
  `GameCardColumns` column set a subclass calls "present" on a row. A
  subclass fixes `LABEL_COLUMN`/`DEFAULT_OUTPUT_PATH` and implements
  `_present_card_uuids()`/`_value_for_row()`; an optional
  `_extra_accumulate(row)` hook (no-op by default) lets a subclass add
  extra per-row bookkeeping without overriding `accumulate()` itself.
- `game_card_average_metrics.py` — three concrete
  `GameCardAverageMetric` subclasses: `WinRateWhenInDeckMetric`
  (`P(won | card in deck_<name>)`), `OpeningHandWinRateMetric`
  (`P(won | card in opening_hand_<name>)`), `DrawnWinRateMetric`
  (`P(won | card in drawn_<name>)` — a card seen at any point in the
  game, opening hand or not).
- `game_length_association_metric.py` — `GameLengthAssociationMetric`,
  a fourth `GameCardAverageMetric` subclass kept in its own file since
  it needs a format-wide baseline the base class doesn't track: it
  overrides `_extra_accumulate()` to update a running
  `(_global_turn_sum, _global_game_count)` unconditionally on every
  row, then overrides `finalize()` to write each card's own average
  `num_turns` minus that global baseline, instead of the base class's
  plain average.
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

Every metric's constructor takes `(card_binder, header, source_game,
output_path=None)` directly (plus a required `deck_box` for the four
deck-input metrics — `game_deck_label_metrics.py`'s three concretes and
`OnPlayWinRateSensitivityByDeckMetric`) and builds its own private
`GameCardColumns` internally via `GameCardColumns.from_header()`. Each
metric re-parsing the same CSV header independently is a header-sized
cost (thousands of columns for `game_data`), not a row-sized one, so
paying it once per metric is negligible next to the scan itself. A
driver scanning multiple metrics over the same CSV reads the header
once itself and passes the same `header` object to every metric's
constructor; a driver wanting decks deduped across several deck-input
metrics passes the *same* `DeckBox` instance to each of their
constructors and saves it itself once scanning finishes — no shared
`GameCardColumns`/`DeckBox` instance is ever built or injected by
`scanner.py` itself.

The per-game identifier this container settles on, since no single raw
column is a unique key: the composite `(draft_id: str, match_number:
int, game_number: int)`, read directly off each row as three separate
output columns — mirroring `draft_data`'s own `draft_id`/`pack_number`/
`pick_number` convention rather than one joined string.

## How it works

`GameCardAverageMetric` and its three `game_card_average_metrics.py`
subclasses share one `accumulate()` sequence: tally every card this
subclass's column set has present on a row toward a running
`(value_sum, total_count)`, then call the optional `_extra_accumulate()`
hook. `finalize()` writes one row per card seen at least once
(`nocab_uuid`, `LABEL_COLUMN`, `sample_count`).
`GameLengthAssociationMetric` reuses that same `accumulate()` unchanged
but overrides `_extra_accumulate()`/`finalize()` to subtract a
format-wide baseline from each card's own average — the same kind of
documented Template Method exception
[`../draft_data/pick_number_decay_curve_metric.py`](../draft_data/pick_number_decay_curve_metric.py)'s
`PickNumberDecayCurveMetric` already established.

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

```python
from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    DrawnWinRateMetric,
    OpeningHandWinRateMetric,
    WinRateWhenInDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (
    OnPlayWinRateSensitivityByDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_pool_metric import (
    TutorTargetPoolMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import (
    scan_game_csv,
)
from src.schema.game_id import GameId

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
raw_csv_path = Path("data/raw/17lands/game_data/MSH.PremierDraft.csv")
header = pd.read_csv(raw_csv_path, nrows=0).columns
deck_box = DeckBox()  # metrics-private - never the published deck box,
# shared across every deck-input metric below so identical decks dedupe

metrics = [
    WinRateWhenInDeckMetric(binder, header, GameId.MTG),
    OpeningHandWinRateMetric(binder, header, GameId.MTG),
    DrawnWinRateMetric(binder, header, GameId.MTG),
    GameLengthAssociationMetric(binder, header, GameId.MTG),
    OnPlayWinRateDeltaMetric(binder, header, GameId.MTG),
    TutorTargetRateMetric(binder, header, GameId.MTG),
    DeckWinPredictionMetric(binder, header, GameId.MTG, deck_box),
    DeckGameLengthPredictionMetric(binder, header, GameId.MTG, deck_box),
    DeckRankTierPredictionMetric(binder, header, GameId.MTG, deck_box),
    OnPlayWinRateSensitivityByDeckMetric(binder, header, GameId.MTG, deck_box),
    TutorTargetPoolMetric(binder, header, GameId.MTG),
]
scan_game_csv(raw_csv_path, metrics)
# each metric's DEFAULT_OUTPUT_PATH now exists

deck_box.save(
    Path("data/metrics/seventeenlands/game_data/deck_box.jsonl"),
    GameId.MTG,
    binder.version_for(GameId.MTG),
)
```
