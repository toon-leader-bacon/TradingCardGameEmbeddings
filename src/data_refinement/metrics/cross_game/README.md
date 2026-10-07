# cross_game

Metrics whose rows span several games, so they cannot live in any one
game's container. Today: `rarity/`, the cross-game rarity tier.

## rarity/

`RarityTierMetric` labels every card of six games (MTG, Pokemon,
Hearthstone, Gwent, Flesh and Blood, StS2) with a `RarityTier`
([`../../../schema/rarity_tier.py`](../../../schema/rarity_tier.py)): how
scarce the card is to get, on a four-step ladder shared by every game.
`TIER_1` is the most common and `TIER_4` the rarest. Two values sit off
the ladder: `SPECIAL` (scarce but not on the ladder: promos, event cards)
and `OTHER` (not a rarity: curses, statuses, tokens, quests). Dominion
has no rarity and gets no rows, as does any card with no rarity value
(including each game's Unknown sentinel).

It is a `CorpusScanMetric` ([`../generic/corpus_scan_metric.py`](../generic/corpus_scan_metric.py)):
the card store is already loaded, so `scan()` writes the whole file in
one call.

### Files

- `rarity/rarity_translator.py` - `RarityTranslator`, the Protocol for
  one game's rarity field (`rarity_tier_of(card)`, `raw_rarity_of(card)`,
  `masked_paths()`); `FieldRarityTranslator`, which covers a rarity held
  in one top-level `raw_content` string (all six games); and
  `UnmappedRarityError`, raised for a raw value its game's table does not
  list, so a new rarity in a re-download fails the next run instead of
  landing in a wrong class.
- `rarity/translator_tables.py` - each game's raw rarity -> tier table
  and `RARITY_TRANSLATORS` (game -> translator). Data only; tuning a
  mapping is an edit here.
- `rarity/rarity_tier_metric.py` - `RarityTierMetric(card_lookup,
  translators=RARITY_TRANSLATORS, output_path=None)`.

### Output

`data/metrics/cross_game/rarity_tier.parquet`, one file for all games:

| Column | Meaning |
| --- | --- |
| `nocab_uuid` | the card |
| `source_game` | `GameId` value |
| `raw_rarity` | the binder's value |
| `label` | `RarityTier` value (`OTHER` rows are written) |

The file carries one CardBinder version per game, stored as a JSON
object under the `card_binder_versions` schema key
([`../version_metadata.py`](../version_metadata.py),
`MultiGameVersionMetadata`). A single-game metric file keeps its
`game` / `card_binder_version` keys and does not carry this one.
Evaluation reads it with `MetricParquetLabels("rarity_tier", [path])`.

### Mapping

Counts are from the live binders on 2026-10-07. A card whose
binder entry has no rarity (295 Pokemon, 1 MTG) gets no row.

| Game | TIER_1 | TIER_2 | TIER_3 | TIER_4 | SPECIAL | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| MTG | 11,592 | 10,317 | 10,912 | 2,057 | 3 | 0 |
| Hearthstone | 2,448 | 1,714 | 1,011 | 1,014 | 0 | 0 |
| Gwent | 270 | 243 | 341 | 406 | 0 | 0 |
| Flesh and Blood | 2,592 | 1,304 | 976 | 114 | 44 | 157 |
| StS2 | 119 | 219 | 155 | 18 | 19 | 47 |
| Pokemon | 5,021 | 4,705 | 3,754 | 2,180 | 973 | 0 |

- MTG: common, uncommon, rare, mythic; special and bonus are `SPECIAL`.
- Hearthstone: FREE and COMMON, RARE, EPIC, LEGENDARY.
- Gwent: common, rare, epic, legendary.
- Flesh and Blood: basic and common, rare, super-rare and majestic,
  legendary and fabled; promo and marvel are `SPECIAL`; token is `OTHER`.
- StS2: Basic and Common, Uncommon, Rare, Ancient; Event is `SPECIAL`;
  Curse, Status, Token and Quest are `OTHER`.
- Pokemon: Common, Uncommon, Rare and Rare Holo, then 37 ultra, secret,
  illustration, shiny, ex, V, GX and ACE SPEC variants as one top tier;
  Promo, Classic Collection and Pikachu Rare are `SPECIAL`. This table
  is checked against the raw set files; the binder stores each card's
  lowest print rarity.

Games fill the tiers unevenly (Gwent is top-heavy, FaB's top tier is
2%); that is each game's design, not a mapping error.

### How to run

```bash
PYTHONPATH=. python scripts/run_metrics.py --source cross_game
```

It loads every translated game's binder into one `CardBinder`, so all six
binder files must exist.
