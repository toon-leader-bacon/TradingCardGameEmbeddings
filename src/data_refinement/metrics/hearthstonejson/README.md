# hearthstonejson

Eight single-card masking metrics over the Hearthstone `CardBinder`
(`../../card_binder/hearthstonejson/`, built from the newest
HearthstoneJSON build only), in `card_mask_metrics.py`, with paired dojos
in `src/dojos/hearthstonejson/card_mask_dojos.py` (see
[`../scryfall/README.md`](../scryfall/README.md) for the shape).

Counts are from the binder built from build 251951 (6,187 cards,
2026-10-01).

| Metric | Field | Label | Rows | Also masked by the dojo |
|---|---|---|---|---|
| `CostRegressionMetric` | `cost` | regression, 0-20 | 6,183 | - |
| `AttackRegressionMetric` | `attack` | regression, minions | 3,952 | - |
| `HealthRegressionMetric` | `health` | regression, minions | 3,950 | - |
| `ClassMaskMetric` | `cardClass` | 11 classes + NEUTRAL + MULTICLASS | 6,187 | `classes`, `runeCost` |
| `RarityMaskMetric` | `rarity` | FREE / COMMON / RARE / EPIC / LEGENDARY | 6,187 | - |
| `CardTypeMaskMetric` | `type` | MINION / SPELL / WEAPON / LOCATION / HERO | 6,187 | `attack`, `health`, `durability`, `armor`, `races`, `spellSchool` |
| `RacesMaskMetric` | `races` | multi-label over 12 tribes (none = empty; ALL = every tribe) | 3,952 | - |
| `SpellSchoolMaskMetric` | `spellSchool` | 7 schools + NONE | 1,945 | - |

Choices:

- Class is the deck-defining field (one class plus neutrals). A card
  with a `classes` list is MULTICLASS; `runeCost` exists only on Death
  Knight cards, so it is masked with the class.
- Card type masks every key that only some types have; otherwise the
  presence of `races` or `spellSchool` would name the type.
- A spell with no school is labeled NONE (about half of spells), and a
  minion with no tribe is the empty set: both are real answers.
- Costs 25, 30 and 100 (four event cards) are skipped.

## How to run

```bash
PYTHONPATH=. python scripts/run_card_binder_ingestion.py --source hearthstonejson
PYTHONPATH=. python scripts/run_metrics.py --source hearthstonejson
```

`BRAINSTORM.md` holds the remaining ideas (mechanics would need the raw
build: the binder drops `mechanics`).
