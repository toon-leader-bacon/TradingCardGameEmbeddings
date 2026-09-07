# sts_gg

Converts sts_gg's raw Slay the Spire 2 run data
(`data/raw/sts_gg/runs.jsonl`) into training-data parquet files,
resolving each card reference against the same `CardBinder`
`spire_codex`'s ingestion stage populates (see
[`../../card_binder/spire_codex/README.md`](../../card_binder/spire_codex/README.md)).
Every metric here satisfies the shared `Metric[dict]` Protocol
([`../metric.py`](../metric.py)): twelve per-card accumulation
metrics (ten of which share one Template Method base,
`card_average_metric.py`), and twelve per-run streaming metrics that
all share a separate Template Method base (`deck_label_metric.py` —
see "Deck-to-scalar metrics" below). Each stats-block scalar this
container reads (`relicCount`, `totalDamageTaken`, `deckSize`, etc.)
gets both a per-deck and a per-card metric — see "Per-card average
metrics" below for how the two families relate.

## Files

- `scanner.py` — `scan_runs_jsonl(raw_path, metrics)` drives every
  metric in a list over one read pass of `runs.jsonl`, isolating one
  metric's `accumulate()`/`finalize()` failure (logged, not raised)
  from every other metric in the list.
- `card_upgrade_rate_metric.py` — `CardUpgradeRateMetric`
  (accumulation): `P(card upgraded by run end | card in final deck)`.
- `card_win_rate_at_act2_metric.py` — `CardWinRateAtAct2Metric`
  (accumulation): `P(win | card in deck_at_start_of_act_2)`, where
  "start of act 2" is derived per-run from `hpPerFloor`'s `actIdx`
  field, never a hardcoded floor cutoff — the boundary varies run to
  run (Neow bonuses/skipped floors shift it).
- `deck_label_metric.py` — `DeckLabelMetric` (streaming, abstract):
  Template Method base for "one run's final deck -> one scalar label
  already on that row" metrics — see "Deck-to-scalar metrics" below.
- `deck_label_metrics.py` — eleven concrete `DeckLabelMetric`
  subclasses: `RelicCountMetric`, `CharacterPredictionMetric`,
  `TotalDamageTakenMetric`, `TotalCardsPickedMetric`,
  `TotalCardsSkippedMetric`, `TotalTurnsMetric`, `ElitesKilledMetric`,
  `FloorsClearedMetric`, `TotalCombatsMetric`, `KilledByMetric`
  (nullable — null on a win), `WinMetric`.
- `ascension_prediction_metric.py` — `AscensionPredictionMetric`, a
  twelfth `DeckLabelMetric` subclass kept in its own file since it
  predates `deck_label_metrics.py` and other code already imports it
  from this path.
- `card_average_metric.py` — `CardAverageMetric` (accumulation,
  abstract): Template Method base for "average this run-level scalar
  per card, across every run that card appeared in" metrics — see
  "Per-card average metrics" below.
- `card_average_metrics.py` — nine concrete `CardAverageMetric`
  subclasses: `CardRelicCountMetric`, `CardTotalDamageTakenMetric`,
  `CardDeckSizeMetric`, `CardTotalCardsPickedMetric`,
  `CardTotalTurnsMetric`, `CardElitesKilledMetric`,
  `CardFloorsClearedMetric`, `CardTotalCombatsMetric`,
  `CardWinRateMetric` (the naive, per-card, unconditioned
  `P(win | card in final deck)` — distinct from
  `CardWinRateAtAct2Metric`'s act-2-conditioned version above).
- `card_character_prediction_metric.py` — `CardCharacterPredictionMetric`
  (accumulation): the per-card frequency-table version of
  `CharacterPredictionMetric` — one row per (card, character) pair
  actually seen, not one scalar per card, so it does NOT subclass
  `CardAverageMetric` (see its own module docstring).
- `BRAINSTORM.md` — candidate metrics not yet built from this raw
  source.

## Deck references

sts_gg's raw `"deck"` field only ever lists a run's FINAL deck (each
entry tagged with the floor it was *acquired* at, not floors it was
present for) — a card removed mid-run leaves no trace anywhere in the
row. There is no way to reconstruct "deck at floor N" from this data,
so every multi-card metric in this container has the same input: the
run's final deck.

Rather than embedding that deck's full card list in every such
metric's own output, a multi-card metric takes a
`DeckBox` ([`../../deck_box/deck_box.py`](../../deck_box/deck_box.py))
in its constructor — a **private instance, never the published deck
box** `deck_box/sts_gg/` builds — and writes the deck into it via
`DeckBox.create_if_absent()`, keyed by
[`../hash_utils.py`](../hash_utils.py)'s `deck_uuid_from_cards()` (a
content-addressed hash of the resolved card multiset). The metric's
own output row then carries that `deck_uuid` instead of the card list.
Every metric in one scan pass that's handed the *same* `DeckBox`
instance dedupes identical final decks against each other for free,
since they all mint the same id from the same content — this only
works because `deck_uuid_from_cards()` is one shared function, not
each metric hashing independently.

Every metric in this container takes a `deck_box` constructor
parameter in the same position for a consistent shape. The two
per-card metrics (`CardUpgradeRateMetric`/`CardWinRateAtAct2Metric`)
never reference a deck, so theirs defaults to `None` and is stored
unused; every `DeckLabelMetric` subclass genuinely needs one, so
theirs is required with no default.

## Deck-to-scalar metrics

`DeckLabelMetric` (`deck_label_metric.py`) is the Template Method base
for every metric whose training input is one run's final deck and
whose label is a single scalar already present on that same run's raw
row (`deck_label_metrics.py`'s twelve subclasses, counting
`AscensionPredictionMetric`). Every step of `accumulate()` — resolving
the deck's cards, hashing them via `deck_uuid_from_cards()`, writing
the deck into `deck_box` via `create_if_absent()`, writing the output
row — is defined once in the base class; a subclass fixes
`LABEL_COLUMN`/`LABEL_TYPE`/`DEFAULT_OUTPUT_PATH` and implements
`_label_for_run(row)`, the one step that actually varies between them.
This exists because eleven of these metrics are otherwise identical
except for which one field of the row they read — see
`~/.claude/docs/PATTERNS.md`'s Template Method entry.

`scan_runs_jsonl` has no involvement in any of this — it only calls
`accumulate()`/`finalize()` on already-constructed `Metric[dict]`
instances, so a metric's own `DeckBox` writes are invisible to it. The
calling driver owns constructing the private `DeckBox`, injecting it
into whichever metrics want it, and saving it once
`scan_runs_jsonl` returns (see "How to run" below).

## Per-card average metrics

`CardAverageMetric` (`card_average_metric.py`) is the per-card mirror
of `DeckLabelMetric`, one level down: instead of referencing a run's
whole final deck once, it tallies a running average of the same kind
of run-level scalar **per card**, across every run that card ever
appeared in — e.g. `CardRelicCountMetric` answers "averaged over every
run this card was in the final deck of, what was `relicCount`?", where
`RelicCountMetric` (above) answers "for this one run's final deck,
what was `relicCount`?". Nine of `deck_label_metrics.py`'s eleven
scalars (every one except `character` and `killed_by`) have a
`card_average_metrics.py` counterpart.

Unlike `DeckLabelMetric`, this is accumulation, not streaming — a
card's average is only knowable after scanning every run it appears
in, the same shape as `CardUpgradeRateMetric`/`CardWinRateAtAct2Metric`
above. `CardWinRateMetric` folds win/loss through this same averaging
machinery by treating a win as `1.0` and a loss as `0.0` — a rate is
just the average of an indicator variable, so no separate
accumulation shape is needed for it.

`CardCharacterPredictionMetric` (`card_character_prediction_metric.py`)
is the per-card mirror of `CharacterPredictionMetric`, but does NOT
subclass `CardAverageMetric` — a card's output here is a whole
frequency table (one row per `(card, character)` pair actually seen,
each carrying its own `probability`), not a single running average, so
reusing `CardAverageMetric`'s single-sum/count shape would force an
abstraction over what's only a superficially similar problem. It
duplicates the per-copy iteration and card-id resolution rule directly
instead, consistent with this container's existing convention of
duplicating that one small rule per file (see "How it works" below)
rather than centralizing it.

Every metric in this family takes `deck_box: DeckBox | None = None`
in its constructor for the same cross-container consistency reason
`CardUpgradeRateMetric`/`CardWinRateAtAct2Metric` do — none of them
reference a deck, so it's accepted and stored unused.

## How it works

`CardUpgradeRateMetric`/`CardWinRateAtAct2Metric` and every
`CardAverageMetric` subclass share the same accumulation shape:
`accumulate()` only tallies in-memory per-card `(numerator_sum,
total_count)` pairs from one run's raw JSON dict — no output exists
until `finalize()` divides those tallies and writes one parquet row
per card (`nocab_uuid`, a rate/average column, `sample_count`).
`CardCharacterPredictionMetric` tallies the same way, but keyed by
`(card, character)` instead of `card` alone, so `finalize()` writes
one row per pair rather than one row per card. Across all of these, a
card copy present multiple times in one run counts as that many
independent samples, not deduplicated to one per run.

Every `DeckLabelMetric` subclass is streaming instead: a run's deck and
its label are both already present on that one row, so `accumulate()`
writes its one output row (`run_id`, `deck_uuid`, `LABEL_COLUMN`)
immediately via an open `pyarrow.parquet.ParquetWriter` (after writing
that run's final deck into its `DeckBox` — see "Deck references"
above), and `finalize()` only closes that writer — no cross-run
aggregation needed, unlike the two accumulation metrics above.

Across every metric here, a card that fails to resolve against the
`CardBinder` (e.g. spire_codex's card dump lagging a newer sts.gg
build) is logged and excluded from that one deck entry only — never
drops the run. `CardUpgradeRateMetric`, `CardWinRateAtAct2Metric`,
`DeckLabelMetric`, `CardAverageMetric`, and
`CardCharacterPredictionMetric` each independently duplicate the same
`"CARD."`-prefix-strip + `CardBinder.get_by_alias` resolution rule
(same rule as
[`../legacy/sts_gg/deck_outcome_metric.py`](../legacy/sts_gg/deck_outcome_metric.py))
rather than sharing a helper — deliberate, pending a later
cross-cutting dedup pass across every sts_gg/sts2runs site with this
same logic. Neither `DeckLabelMetric`'s twelve subclasses nor
`CardAverageMetric`'s nine repeat this duplication among themselves —
that's exactly what each family's own Template Method base collapses
to one copy.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts_gg.card_upgrade_rate_metric import (
    CardUpgradeRateMetric,
)
from src.data_refinement.metrics.sts_gg.card_win_rate_at_act2_metric import (
    CardWinRateAtAct2Metric,
)
from src.data_refinement.metrics.sts_gg.ascension_prediction_metric import (
    AscensionPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    RelicCountMetric,
    WinMetric,
    # ...and the rest of deck_label_metrics.py's DeckLabelMetric subclasses
)
from src.data_refinement.metrics.sts_gg.card_average_metrics import (
    CardRelicCountMetric,
    CardWinRateMetric,
    # ...and the rest of card_average_metrics.py's CardAverageMetric subclasses
)
from src.data_refinement.metrics.sts_gg.card_character_prediction_metric import (
    CardCharacterPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.scanner import scan_runs_jsonl
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.game_id import GameId

binder = CardBinder.load([Path("data/final/cards/slay_the_spire_2.jsonl")])
deck_box = DeckBox()  # metrics-private - never the published deck box
metrics = [
    CardUpgradeRateMetric(binder),  # accepts deck_box but never uses it
    CardWinRateAtAct2Metric(binder),  # accepts deck_box but never uses it
    AscensionPredictionMetric(binder, deck_box),
    RelicCountMetric(binder, deck_box),
    WinMetric(binder, deck_box),
    CardRelicCountMetric(binder),  # per-card: accepts deck_box but never uses it
    CardWinRateMetric(binder),
    CardCharacterPredictionMetric(binder),
]
scan_runs_jsonl(Path("data/raw/sts_gg/runs.jsonl"), metrics)
# each metric's DEFAULT_OUTPUT_PATH now exists
deck_box.save(Path("data/metrics/sts_gg/deck_box.jsonl"), GameId.SLAY_THE_SPIRE_2)
```
