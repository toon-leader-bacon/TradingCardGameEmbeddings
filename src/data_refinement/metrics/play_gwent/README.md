# play_gwent

Metrics driven directly by playgwent.com's community deck guides
(`data/raw/play_gwent/guides.jsonl`), rather than gwent.one's card
catalog (`../gwent_one/`). Two small utilities plus the first
deck-masking metric family live here.

## `DeckCardMaskMetric` — game-aware whole-card deck masking

[`../deck_card_mask_metric.py`](../deck_card_mask_metric.py) is a
`Metric[dict]`-shaped (`accumulate()`/`finalize()`, [`../metric.py`](../metric.py))
Template Method base for masking one whole card out of a deck, chosen
via game-specific logic — distinct from `../masked_field_metric.py`'s
`MaskedFieldMetric` family, which masks one *field* of one *card*, not
a whole card out of a deck.

**Raw-row-driven, not `DeckBox`-driven:** each metric's `accumulate()`
is called once per raw guide object from `guides.jsonl`. `GenericDeck`/
`DeckBox` are deliberately game-agnostic (a plain `card_nocab_uuids`
multiset, no role/slot information) — once a deck is stored, there's
no way to recover "which entry was the leader" from the deck alone.
So every metric reads its masking target directly off the raw row's
own fields (e.g. a guide's `leaderId`), never by re-deriving it from
an already-stored deck.

As a side effect of every row (whether or not that row also yields a
training example), the row's deck is ensured to exist in the
`DeckBox` passed to the metric's constructor — by delegating to
`../../deck_box/play_gwent/extraction_stage.py`'s
`PlayGwentDeckExtractionStage.extract_one()`/`deck_uuid_for_guide()`
(both public, reused by both the published-deck-box builder and every
metric here), never by re-implementing card resolution or uuid-minting
in the metric itself. The `DeckBox` passed in may be empty (cold
start) or the already-loaded published `data/final/decks/gwent.jsonl`
box (warm start) — either is safe, since `extract_one()`'s writes are
idempotent by `deck_uuid_for_guide()`'s deterministic id scheme.

**Output schema** (one row per raw guide with a target found):
`deck_uuid: str`, `target_card_uuid: str`, `label: str`. Deliberately
has no `masked_field` column, unlike `MaskedFieldMetric` — the whole
card is masked, not one field of it.

**Label semantics are deliberately per-metric**, unlike
`MaskedFieldMetric`'s shared `MASKED_FIELD`-path convention: each
concrete subclass decides independently what's worth predicting about
its target card. The plan is to build several of these metrics first
and look for patterns to de-duplicate later, rather than forcing a
shared shape prematurely.

A subclass fixes:

- `_deck_uuid_for_row(row, deck_box) -> UUID` — ensures `row`'s deck
  exists in `deck_box` (side effect) and returns its uuid.
- `_target_card_uuid_for_row(row, card_lookup) -> UUID | None` — the
  game-aware selection, read directly off `row`. `None` means "no
  valid target for this row" — a real, expected outcome, not an error.
- `_label_for_card(card) -> str` — the true label for the masked-out
  card.

### `LeaderMaskedFromDeckMetric`

[`leader_masked_from_deck_metric.py`](leader_masked_from_deck_metric.py) —
masks a Gwent deck's leader card and predicts its **identity** (which
of 42 leaders), not a field of the leader (e.g. not its faction, which
is near-trivially recoverable from the rest of a Gwent deck's cards,
making it an uninteresting prediction target). Mirrors `sts_gg`'s
`CharacterPredictionMetric` precedent (predicts character identity,
not a field of the character).

- `SOURCE_GAME = GameId.GWENT`
- `LABEL_VALUES = leader_labels.LEADER_NAMES` (42 frozen leader names)
- `DEFAULT_OUTPUT_PATH = Path("data/metrics/play_gwent/leader_masked_from_deck.parquet")`
- `_deck_uuid_for_row`: delegates to
  `PlayGwentDeckExtractionStage.extract_one()`/`deck_uuid_for_guide()`.
- `_target_card_uuid_for_row`: reads `row["leaderId"]` directly and
  looks it up via `card_lookup.get_by_alias(GameId.GWENT, DataSource.GWENT_ONE, str(leader_id))`
  — the same alias namespace `PlayGwentDeckExtractionStage`'s own card
  resolution uses. A missing/unresolvable `leaderId` is a data-quality
  edge case (logged, row skipped — its deck is still registered).
- `_label_for_card`: the leader card's `raw_content["name"]`, falling
  back to `masked_field_metric.OTHER_LABEL` for a name outside
  `LEADER_NAMES` (an expected failure mode as new leaders are added
  upstream — see `leader_labels.py`'s FRESHNESS caveat — not a bug).

## `scanner.py`

`scan_guides_jsonl(raw_path, metrics)` drives every `Metric[dict]` in
`metrics` over one streamed read of `guides.jsonl`, then finalizes all
of them — a per-container copy of `../sts_gg/scanner.py`'s shape
(deliberately not centralized; this project defers that kind of
cross-container dedup to a later, comprehensive pass). Isolates one
metric's `accumulate()`/`finalize()` failure from the rest of the
list, logging rather than raising. A line that isn't valid JSON is
silently skipped (matches `leader_deck_counts.py`'s convention for
this same raw source).

## `leader_deck_counts.py` / `leader_labels.py`

`leader_deck_counts.count_decks_per_leader(raw_path=None) -> dict[str, int]`
is metric-design tooling (not a metric itself): streams `guides.jsonl`
and counts guide decks per leader name, using each guide's own
`leader.name` field directly — no `CardBinder`/`card_lookup` join
needed. Used to generate `leader_labels.LEADER_NAMES`, a frozen tuple
of the 42 distinct leader names observed across all 60,138 guides as
of 2026-09-09 (well-balanced: 491–2555 decks per leader, no long tail)
— see that file's own FRESHNESS caveat for what to do if a future
expansion adds leaders.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.data_refinement.metrics.play_gwent.scanner import scan_guides_jsonl
from src.schema.game_id import GameId

card_binder = CardBinder.load([Path("data/final/cards/gwent.jsonl")])
deck_box = DeckBox.load([Path("data/final/decks/gwent.jsonl")])  # warm start
metrics = [LeaderMaskedFromDeckMetric(card_binder, deck_box)]

scan_guides_jsonl(Path("data/raw/play_gwent/guides.jsonl"), metrics)
deck_box.save(Path("data/final/decks/gwent.jsonl"), GameId.GWENT)
```

## Out of scope (deferred)

- **Multi-target masking** (e.g. "mask every artifact in the deck",
  "mask every stratagem") is a real, wanted future direction — see
  `BRAINSTORM.md`'s multi-card section — but needs its own output
  shape (a structured JSON masking rule, not a list-of-lists-of-
  strings) and its own design pass. `DeckCardMaskMetric` as built
  supports exactly one target card per row; do not stretch it to
  lists.
- **Downstream Dojo consumer**: no `Dojo` reads any `play_gwent` deck
  metric yet. A future `DataConstructor` would resolve `deck_uuid` via
  `DeckBox`, `target_card_uuid` via `CardBinder`/`CardLookup`, mask
  that whole card out of the deck's card list before building model
  input, and read `label` against the originating metric's
  `LABEL_VALUES` for decoder head sizing (same shared contract as
  `../gwent_one/README.md`'s equivalent note for `MaskedFieldMetric`).
