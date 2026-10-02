# fabtcg_decklists

Metrics over fabtcg.com's published tournament decklists
(`data/raw/fabtcg_decklists/decklists/<slug>.html`, 4,161 files). Ideas
not built yet are in [`BRAINSTORM.md`](BRAINSTORM.md).

**Read-only over the published deck box.** The deck box ingestion stage
(`../../deck_box/fabtcg_decklists/`) already resolved every decklist into
`data/final/decks/flesh_and_blood.db`. These metrics never re-resolve
card names and never write a deck box: `scanner.py` feeds one
`{"slug": <file stem>}` row per raw file, and `PublishedDecklists`
(`published_decklists.py`) finds that slug's deck in the published box
(a published deck's provenance `source_id` is its slug). Deck-level rows
therefore point into the published box (`requires_deck_box`), like
`../final_decks/` and `../sts2_runs/`.

**Hero.** The box keeps no slot structure, but a hero is card-intrinsic:
the deck's one card whose typebox names `Hero` (`hero_legality.is_hero`).
4,152 of the 4,161 decks have exactly one; the rest are skipped.

**Legality** (`hero_legality.is_legal_for_hero`): card talents must be the
hero's (Elemental also grants Earth, Ice, Lightning); card classes must
all be the hero's, or any one for a `/` hybrid; Revered and Reviled
exclude each other; a `**X Specialization**` card needs X in the hero's
name. Only 0.19% of real deck card slots break it.

## Metrics

| Metric | Input -> label | Rows (2026-10-01) |
|---|---|---|
| `HeroMaskedFromDeckMetric` (`hero_masked_from_deck.parquet`) | deck minus its hero -> hero name, over `hero_labels.HERO_NAMES` (48 heroes with 20+ decks) + `OTHER` (77 heroes, 454 decks). A `DeckCardMaskMetric`, the twin of `../play_gwent/`'s leader metric | 4,152 |
| `CardInclusionRateMetric` (`card_inclusion_rate.parquet`) | card -> P(card in deck \| card legal for the deck's hero); cards legal for fewer than 20 decks are dropped | 2,411 |
| `HeroConditionedInclusionMetric` (`hero_conditioned_inclusion.parquet`) | card -> P(card in deck \| hero) per `HERO_NAMES` hero, as two parallel lists (`inclusion_rate_by_hero`, `legal_deck_count_by_hero`; count 0 = illegal, masked) | 2,405 |

The two inclusion metrics share a `HeroInclusionTally`
(`hero_inclusion_tally.py`) and a Template Method base in
`card_inclusion_metrics.py`. Card universe: cards some deck holds (a card
no deck holds may postdate the corpus). Known bias: the deck box maps a
decklist name printed in several pitches to one printing, so a rate is
really per card name.

Not built: pitch-curve shape (BRAINSTORM #16). The task is "hero + gear ->
cards per pitch", but no generic cell maps a deck to a distribution, and
the hero-and-gear sub-deck it needs is not in the published box (it would
need a metric-private box or a new data constructor).

## How to run

```
PYTHONPATH=. python3 scripts/run_metrics.py --source fabtcg_decklists
```

About 30 s. Needs the FaB binder and the published FaB deck box; writes
only `data/metrics/fabtcg_decklists/`.
