# replay_data

Converts 17lands' raw per-game replay data
(`data/raw/17lands/replay_data/<Set>.<EventType>.csv`) into training-data
parquet files under `data/metrics/seventeenlands/replay_data/`, matching
each CSV's `deck_<name>`/`sideboard_<name>` column suffixes against a
`CardBinder` already populated for `GameId.MTG` (see
[`../../../card_binder/README.md`](../../../card_binder/README.md)), and
resolving every per-turn Arena-ID cell (`creatures_cast`,
`creatures_attacked`, `creatures_killed_combat`, ...) against the same
`CardBinder` via its Arena-alias index. Every metric here satisfies the
shared `Metric[dict]` Protocol ([`../../metric.py`](../../metric.py)).

## Files

- `replay_card_columns.py` — `ReplayCardColumns`: built once per metric
  instance from that metric's own `(card_binder, header, source_game)`.
  Two independent matching mechanisms live here (see "Card matching"
  below): `deck_columns`/`sideboard_columns` (header-parsed, like the
  sibling containers), and `uuid_for_arena_id()`/`arena_uuids()`
  (per-cell, Arena-ID based - this container's only card-shaped columns
  beyond `deck_`/`sideboard_`, since replay_data has no
  `opening_hand_<name>` family the way `game_data` does).
  `user_turn_numbers`/`oppo_turn_numbers` are derived from the header at
  construction (never hardcoded), and `turn_column(actor, turn, field)`
  builds the literal `f"{actor}_turn_{turn}_{field}"` column name every
  metric below needs.
- `scanner.py` — `scan_replay_csv(raw_csv_path, metrics, chunk_size)`
  drives every metric in a list over one chunked read pass of a
  replay_data CSV, handing each metric one row at a time and isolating
  one metric's `accumulate()`/`finalize()` failure (logged, not raised)
  from every other metric in the list. Reads with `low_memory=False` -
  a deliberate deviation from `draft_data`/`game_data`'s scanners,
  since replay_data's ~2500 mostly-sparse per-turn columns otherwise
  get inconsistent per-internal-chunk dtype inference from pandas' C
  parser (see "Card matching" below for why `ReplayCardColumns` already
  tolerates either dtype regardless).
- `replay_turn_event_rate_metric.py` — `ReplayTurnEventRateMetric`
  (Template Method, abstract, accumulation): the turn-indexed
  generalization of `game_data.GameCardAverageMetric`'s shape - instead
  of one value applied to every card present on a row, this walks every
  `(actor, turn)` half-turn on a row and tallies a `(hit_count,
  total_count)` pair per card across every occurrence. A subclass fixes
  `LABEL_COLUMN`/`DEFAULT_OUTPUT_PATH` and implements
  `_denominator_cards_for_turn()`/`_numerator_cards_for_turn()`.
- `replay_turn_event_rate_metrics.py` — two concrete
  `ReplayTurnEventRateMetric` subclasses: `CombatKillInvolvementRateMetric`
  (`P(a creature died in combat that half-turn | card fought that
  half-turn)` - denominator is `creatures_attacked ∪ creatures_blocking`,
  numerator is a turn-wide boolean checked against the RAW
  `user_creatures_killed_combat`/`oppo_creatures_killed_combat` cells'
  presence, not `arena_uuids()`'s matched output, since a creature that
  died but doesn't match a known card should still count as "something
  died") and `CombatDamagePushThroughRateMetric` (`P(card in
  creatures_unblocked | card in creatures_attacked)` - a genuine
  per-card numerator, not turn-wide).
- `average_turn_cast_metric.py` — `AverageTurnCastMetric` (standalone,
  accumulation): per card, the average turn number it's cast on across
  every `(actor, turn)` occurrence in every game scanned - both actors'
  occurrences count, no deck-membership conditioning.
- `cast_rate_metric.py` — `CastRateMetric` (standalone, accumulation):
  `P(cast at least once in the game | card in deck_<name>)`.
  User-deck-only, and scans only `user_turn_*` half-turns (a card in
  the user's own deck can only ever be cast by the user).
- `turns_to_game_end_after_cast_metric.py` —
  `TurnsToGameEndAfterCastMetric` (standalone, accumulation): per card,
  average `row["num_turns"] - first_cast_turn` (first occurrence,
  either actor). Verified directly against
  `data/raw/17lands/replay_data/MSH.PremierDraft.csv` that `num_turns`
  is on the same per-actor turn-number scale as
  `user_turn_N`/`oppo_turn_N` (roughly "the highest turn number either
  player reached before the game ended"), not a combined elapsed-turn
  count across both players - so this formula needs no unit
  conversion.
- `discard_rate_metric.py` / `tutor_target_rate_metric.py` — two
  standalone accumulation metrics, same user-deck-only,
  `user_turn_*`-only shape as `CastRateMetric`. `DiscardRateMetric`
  carries a documented, unverified assumption that every
  `user_turn_N_cards_discarded` hit is self-inflicted, never a forced
  discard from an opposing effect. `TutorTargetRateMetric` here is a
  distinct class from
  [`../game_data/tutor_target_rate_metric.py`](../game_data/tutor_target_rate_metric.py)'s
  same-named class - each lives in its own source-specific package,
  the same convention every repeated concept name across
  `draft_data`/`game_data` already follows. There is no
  `oppo_turn_N_cards_tutored` column at all (opponent tutoring is
  hidden), so this one's `user_turn_*`-only scope isn't a choice.
- `combat_aggression_profile_metric.py` —
  `CombatAggressionProfileMetric` (streaming): the user's full
  `deck_<name>` list (referenced by `deck_uuid`, written into a shared
  `DeckBox`) paired with a derived combat-tempo scalar - the average
  `len(creatures_attacked)` per user half-turn that had any attack at
  all. `deck_box` is a required constructor parameter.
- `attacker_blocker_combat_outcome_metric.py` —
  `AttackerBlockerCombatOutcomeMetric` (streaming, fan-out): the second
  fan-out-shaped metric in this codebase (after
  [`../game_data/tutor_target_pool_metric.py`](../game_data/tutor_target_pool_metric.py)'s
  `TutorTargetPoolMetric`) - one row fans out to one output example per
  half-turn with at least one attacker. `net_kill_delta` is
  attacker-favorable-positive: `len(defending side's
  creatures_killed_combat)` minus `len(attacking side's own
  creatures_killed_combat)`, this half-turn. No `deck_box` - this
  metric's identity is `(draft_id, match_number, game_number, actor,
  turn)`, never a `deck_uuid`.
- `BRAINSTORM.md` — candidate metrics from this raw source not yet
  built (this container currently implements the nine ideas covered
  above; several ideas - the hardest temporal-state metrics, the
  freeze-turn-blocked multi-group idea, the structurally-incomplete
  opponent-deck idea, and a couple trimmed for scope discipline - remain
  future work, see that file's own "Deferred to future work" note).

## Card matching

Two independent mechanisms, since replay_data exposes cards two
different ways - unlike `draft_data`/`game_data`, which only ever match
cards by header column suffix:

1. **Name-suffixed header columns** (`deck_<name>`/`sideboard_<name>`
   only): `ReplayCardColumns._match_uuid()` (private - the only place
   this policy lives) matches a bare card name against `card_binder`:
   an exact `card_binder.get_by_name(source_game, name)` match first;
   on 0 or 2+ matches, a regex fallback via
   `card_binder.get_by_name_regex(source_game,
   f"^{re.escape(name)}( //.*)?$")`; on 0 or 2+ matches from that
   fallback, the name is unmatched. This is the same 17lands-wide
   policy [`../draft_data/pack_pool_columns.py`](../draft_data/pack_pool_columns.py)'s
   `DraftCardColumns`/[`../game_data/game_card_columns.py`](../game_data/game_card_columns.py)'s
   `GameCardColumns` already implement for their own sources,
   independently re-typed here rather than shared via inheritance or
   import - see [`../../TODO.md`](../../TODO.md) for the logged
   cross-source dedup consideration this adds a third instance of.
2. **Arena-ID pipe-delimited cells** (every per-turn event column, plus
   `opening_hand`/`candidate_hand_N`/`eot_{side}_*_in_play`):
   `uuid_for_arena_id(arena_id)` matches an already-normalized Arena id
   string via `card_binder.get_by_alias(source_game, DataSource.ARENA,
   arena_id)`, caching hit or miss in a cache separate from
   `uuid_for_name()`'s. `arena_uuids(cell)` is the caller-facing entry
   point: it handles a `NaN`/`None` cell (`[]`), a bare `float`/`int`
   cell (pandas' single-id-column dtype inference), and a
   `"|"`-delimited `str` cell uniformly, normalizing every token via
   `str(int(float(token)))` before the lookup - **this normalization is
   load-bearing, not cosmetic**: a per-turn Arena-ID column where every
   populated cell in a chunk happens to hold exactly one id is inferred
   as `float64` by `pandas.read_csv` (e.g. a cell arrives as the Python
   float `104936.0`), while `ScryfallCardIngestionStage` registers each
   Arena alias as `str(int)` with no decimal (`"104936"`) - a naive
   `str()` call would silently miss every such cell. Unmatched ids are
   dropped into `unmatched_arena_ids` rather than raising, mirroring
   `_match_uuid()`'s own unmatched handling.

## Card-binder and deck-box access shape

Every metric's constructor takes `(card_binder, header, source_game,
output_path=None)` directly (plus a required `deck_box` for
`CombatAggressionProfileMetric`, the only deck-input metric this round)
and builds its own private `ReplayCardColumns` internally via
`ReplayCardColumns.from_header()`. Each metric re-parsing the same CSV
header independently is a header-sized cost, not a row-sized one, so
paying it once per metric is negligible next to the scan itself.

The per-game identifier this container settles on, since no single raw
column is a unique key: the composite `(draft_id: str, match_number:
int, game_number: int)`, read directly off each row as three separate
output columns - the same convention `game_data` already settled on.
`AttackerBlockerCombatOutcomeMetric` extends this with two more
separate columns, `actor: str` (written from a `Literal["user",
"oppo"]` value) and `turn: int`, for its per-half-turn identity.

## How it works

`ReplayTurnEventRateMetric` and its two
`replay_turn_event_rate_metrics.py` subclasses share one `accumulate()`
sequence: for each actor in `("user", "oppo")`, for each turn in that
actor's own dynamically-discovered turn-number range, call
`_denominator_cards_for_turn()` and (if non-empty)
`_numerator_cards_for_turn()`, tallying a running `(hit_count,
total_count)` per card. `finalize()` writes one row per card seen at
least once (`nocab_uuid`, `LABEL_COLUMN`, `sample_count`).

`AverageTurnCastMetric`/`TurnsToGameEndAfterCastMetric` are standalone
accumulators with their own per-card tallies (a `(turn_sum,
occurrence_count)` running average, and a `(delta_sum,
occurrence_count)` running average respectively) - genuinely different
shapes from the hit/total ratio above, so neither subclasses
`ReplayTurnEventRateMetric`. `CastRateMetric`/`DiscardRateMetric`/
`TutorTargetRateMetric` are three more standalone accumulators sharing
a simpler per-game `(hit_count, total_count)` shape (one set
intersection against `present_uuids(row, deck_columns)` per game, not a
per-turn double loop) - kept separate from each other and from the
`ReplayTurnEventRateMetric` family, since the shared logic in each case
is small enough that forcing an abstraction would cost more than the
duplication it removes (see [`../../TODO.md`](../../TODO.md)).

`CombatAggressionProfileMetric` is streaming - one row already carries
a complete example (the deck plus a scalar this class computes via its
own turn loop), so `accumulate()` writes it immediately via an open
`pyarrow.parquet.ParquetWriter` and `finalize()` only closes that
writer, mirroring
[`../../sts_gg/deck_label_metric.py`](../../sts_gg/deck_label_metric.py)'s
shape. `AttackerBlockerCombatOutcomeMetric` is streaming but fans a
single input row out to zero or more output rows (one per qualifying
half-turn) via a single `pa.Table.from_pydict()` call built from a
wider dict, the same fan-out convention
[`../game_data/tutor_target_pool_metric.py`](../game_data/tutor_target_pool_metric.py)'s
`TutorTargetPoolMetric` established first, applied here over turns
within a game rather than over pool members.

## How to run

```python
from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.replay_data.average_turn_cast_metric import (
    AverageTurnCastMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.cast_rate_metric import (
    CastRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.turns_to_game_end_after_cast_metric import (
    TurnsToGameEndAfterCastMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.discard_rate_metric import (
    DiscardRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metrics import (
    CombatDamagePushThroughRateMetric,
    CombatKillInvolvementRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.combat_aggression_profile_metric import (
    CombatAggressionProfileMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.attacker_blocker_combat_outcome_metric import (
    AttackerBlockerCombatOutcomeMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.scanner import (
    scan_replay_csv,
)
from src.schema.game_id import GameId

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
raw_csv_path = Path("data/raw/17lands/replay_data/MSH.PremierDraft.csv")
header = pd.read_csv(raw_csv_path, nrows=0).columns
deck_box = DeckBox()  # metrics-private - never the published deck box

metrics = [
    AverageTurnCastMetric(binder, header, GameId.MTG),
    CastRateMetric(binder, header, GameId.MTG),
    TurnsToGameEndAfterCastMetric(binder, header, GameId.MTG),
    DiscardRateMetric(binder, header, GameId.MTG),
    TutorTargetRateMetric(binder, header, GameId.MTG),
    CombatKillInvolvementRateMetric(binder, header, GameId.MTG),
    CombatDamagePushThroughRateMetric(binder, header, GameId.MTG),
    CombatAggressionProfileMetric(binder, header, GameId.MTG, deck_box),
    AttackerBlockerCombatOutcomeMetric(binder, header, GameId.MTG),
]
scan_replay_csv(raw_csv_path, metrics)
# each metric's DEFAULT_OUTPUT_PATH now exists

deck_box.save(
    Path("data/metrics/seventeenlands/replay_data/deck_box.jsonl"), GameId.MTG
)
```
