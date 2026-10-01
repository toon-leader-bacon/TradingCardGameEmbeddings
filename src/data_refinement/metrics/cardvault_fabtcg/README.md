# cardvault_fabtcg

Six single-card masking metrics over the Flesh and Blood `CardBinder`
(`../../card_binder/cardvault_fabtcg/`), in `card_mask_metrics.py`, with
paired dojos in `src/dojos/cardvault_fabtcg/card_mask_dojos.py` (see
[`../scryfall/README.md`](../scryfall/README.md) for the shape).

Counts are from the live binder (5,187 cards, checked 2026-10-01).

| Metric | Field | Label | Rows | Also masked by the dojo |
|---|---|---|---|---|
| `PitchMaskMetric` | `pitch` | "1" / "2" / "3" | 4,218 | `color` |
| `CostRegressionMetric` | `cost` | regression, whole numbers 0-13 | 4,106 | - |
| `PowerRegressionMetric` | `power` | regression, 0-14 | 2,239 | - |
| `DefenseRegressionMetric` | `defense` | regression, 0-10 | 4,228 | - |
| `ClassMaskMetric` | `typebox` | 13 classes + Classless + Multiclass + OTHER | 5,091 | - |
| `CardTypeMaskMetric` | `typebox` | 9 card types + OTHER | 5,091 | - |

Choices:

- The leaned card has no `classes`, `talents` or `types` keys: talent,
  class and card type are words in `typebox` ("Light Warrior Action -
  Attack"). Class and type both mask the whole typebox and read their
  label off its words. Classless covers talent-only cards ("Shadow
  Action") and events; Multiclass covers hybrids ("Assassin / Ninja",
  "Pirate Necromancer"); OTHER covers Bard, Adjudicator, Merchant,
  Shapeshifter and Thief (3-24 cards each).
- Pitch is a fixed class: three values. `color` is pitch in other words
  (red 1, yellow 2, blue 3 on every card with both), so it is masked
  too. 927 names come in all three pitches with the same text and
  scaled numbers.
- X costs, "*" stats and the one cost of 99 are skipped.
- The 94 two-faced cards are skipped by every metric: `back_face`
  repeats typebox, pitch, color and stats for the other face.

## How to run

```bash
PYTHONPATH=. python scripts/run_metrics.py --source cardvault_fabtcg
```
