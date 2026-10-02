# play_gwent

Metrics driven directly by playgwent.com's community deck guides
(`data/raw/play_gwent/guides.jsonl`), rather than gwent.one's card
catalog (`../gwent_one/`). Two small utilities plus the first
deck-masking metric family live here.

## `DeckCardMaskMetric` — game-aware whole-card deck masking

[`../generic/deck_card_mask_metric.py`](../generic/deck_card_mask_metric.py) is a
`Metric[dict]`-shaped (`accumulate()`/`finalize()`, [`../metric.py`](../metric.py))
Template Method base for masking one whole card out of a deck, chosen
via game-specific logic — distinct from `../generic/masked_field_metric.py`'s
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

Every metric here reads the published Gwent box
(`data/final/decks/gwent.db`) and never writes or saves it. A row's
deck is looked up by `deck_uuid_for_guide(row["id"])` from
`../../deck_box/play_gwent/extraction_stage.py`, the same uuid deck
ingestion minted, and a guide the box lacks is skipped. Until
2026-10-02 the leader metric wrote decks through `extract_one()` and
its run re-saved the published box, which made every metric keyed to
the box stale.

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

- `_deck_uuid_for_row(row, deck_box) -> UUID | None` — `row`'s deck
  uuid (here: looked up in the published box, `None` if absent).
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
- `_deck_uuid_for_row`: `PlayGwentDeckExtractionStage.deck_uuid_for_guide()`,
  if the published box holds that deck.
- `_target_card_uuid_for_row`: reads `row["leaderId"]` directly and
  looks it up via `card_lookup.get_by_alias(GameId.GWENT, DataSource.GWENT_ONE, str(leader_id))`
  — the same alias namespace `PlayGwentDeckExtractionStage`'s own card
  resolution uses. A missing/unresolvable `leaderId` is a data-quality
  edge case (logged, row skipped).
- `_label_for_card`: the leader card's `raw_content["name"]`, falling
  back to `masked_field_metric.OTHER_LABEL` for a name outside
  `LEADER_NAMES` (an expected failure mode as new leaders are added
  upstream — see `leader_labels.py`'s FRESHNESS caveat — not a bug).

## Guide metrics over the published box (read-only)

Three newer metrics (2026-10-01) read each guide's deck from the
published box (`data/final/decks/gwent.db`) and never write any deck
box, like `LeaderMaskedFromDeckMetric`. `published_guide_decks.py`'s `PublishedGuideDecks` parses a raw
guide row into a typed `GuideDeck`: the deck is
`deck_uuid_for_guide(row["id"])` in the published box, and its faction is
the leader card's binder faction (looked up through `leaderId`, the
binder's spelling: `monster`, `northern_realms`). A card is legal for a
faction if it is neutral, of that faction, or a dual-faction
(`faction-duo`) syndicate card naming it (`legal_factions`).

| Metric | Input -> label | Rows (2026-10-01) |
|---|---|---|
| `CardInclusionRateMetric` (`card_inclusion_rate.parquet`) | card -> P(card in deck \| card legal for the deck's faction); leaders excluded, cards legal for fewer than 20 decks dropped | 1,218 |
| `FactionConditionedInclusionMetric` (`faction_conditioned_inclusion.parquet`) | card -> P(card in deck \| faction) per faction, as two parallel lists (`inclusion_rate_by_faction`, `legal_deck_count_by_faction`; count 0 = illegal, masked) | 1,218 |
| `GuideVotesMetric` (`guide_votes.parquet`) | guide deck -> sign(v) * ln(1 + \|v\|) of its net votes (v from -120 to 1,594; 5,268 guides negative, so plain log(1 + v) is undefined). Rows point into the published box | 60,197 |

The two inclusion metrics share a `FactionInclusionTally`
(`faction_inclusion_tally.py`) and a Template Method base in
`card_inclusion_metrics.py`, the Gwent twins of
`../fabtcg_decklists/`'s hero-conditioned metrics. Biases: guides span
2019-2026 and every balance patch (24k are flagged invalid today, kept);
votes also reflect the write-up, the author's reach and the guide's age.

Run them with `scripts/run_metrics.py --source play_gwent_guides` (about
2 minutes; writes only these three files).

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

`leader_deck_counts.count_decks_per_leader(card_lookup, raw_path=None)
-> dict[str, int]` is metric-design tooling (not a metric itself):
streams `guides.jsonl` and counts guide decks per leader, keyed by the
gwent.one `CardBinder`'s canonical name for each guide's `leaderId` —
the same name `LeaderMaskedFromDeckMetric._label_for_card` itself
checks `LEADER_NAMES` against, via the same
`card_lookup.get_by_alias(GameId.GWENT, DataSource.GWENT_ONE,
str(leader_id))` lookup. Deliberately *not* the guide's own embedded
`leader.name` text — playgwent.com's raw guide text and gwent.one's
card names have disagreed on at least two leaders' spelling
(`"Reckless Fury"` vs. `"Reckless Flurry"`), so counting by guide text
would produce a vocabulary that mismatches what the metric validates
against. Used to generate
`leader_labels.LEADER_NAMES`, a frozen tuple of the 42 distinct leaders
observed across 60,197 guides as of 2026-09-25 (well-balanced: 492–2559
decks per leader, no long tail) — see that file's own FRESHNESS caveat
for what to do if a future expansion adds leaders.

## How to run

Ingest the play_gwent deck box first
(`scripts/run_deck_box_ingestion.py --source play_gwent`), then:

```
PYTHONPATH=. python3 scripts/run_metrics.py --source play_gwent         # leader mask
PYTHONPATH=. python3 scripts/run_metrics.py --source play_gwent_guides  # the other three
```

Neither run writes the deck box.

## Out of scope (deferred)

- **Multi-target masking** (e.g. "mask every artifact in the deck",
  "mask every stratagem") is a real, wanted future direction — see
  `BRAINSTORM.md`'s multi-card section — but needs its own output
  shape (a structured JSON masking rule, not a list-of-lists-of-
  strings) and its own design pass. `DeckCardMaskMetric` as built
  supports exactly one target card per row; do not stretch it to
  lists.
