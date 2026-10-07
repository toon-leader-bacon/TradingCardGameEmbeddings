# pokemon_tcg

Five single-card masking metrics over the Pokemon `CardBinder`
(`../../card_binder/pokemon_tcg/`), in `card_mask_metrics.py`, with
paired dojos in `src/dojos/pokemon_tcg/card_mask_dojos.py` (see
[`../scryfall/README.md`](../scryfall/README.md) for the shape).

All five cover Pokemon cards only (14,273 of 16,803; checked
2026-10-01).

| Metric | Field | Label | Rows | Also masked by the dojo |
|---|---|---|---|---|
| `HpRegressionMetric` | `hp` | regression, 30-380 | 14,273 | - |
| `TypesMaskMetric` | `types` | multi-label over the 11 energy types | 14,273 | each attack's `cost` (`attacks[0..3].cost`) |
| `StageMaskMetric` | `subtypes` | Basic / Stage 1 / Stage 2 + OTHER (VMAX, BREAK, ...) | 14,272 | `evolvesFrom`, `evolvesTo` |
| `RetreatCostRegressionMetric` | `convertedRetreatCost` | regression, 0-5 | 13,493 | - |
| `WeaknessMaskMetric` | `weaknesses` | first weakness's type (11 types + OTHER) | 13,895 | - |

Choices:

- Attack costs name the Pokemon's own type on 86% of Pokemon, so the
  types dojo masks every attack's cost. A card's weakness still
  correlates with its type; that is game knowledge, not a leak.
- Retreat cost skips the 780 Pokemon without the key: the source omits
  it instead of writing 0, so their value is unknown.
- Not built: supertype (hp, attacks and rules make it obvious on every
  card), trainer subtype (85% of Trainers carry reminder text in
  `rules`, such as "You may play only 1 Supporter card", and `rules` is
  also their whole effect text), and rarity (covered
  by the `cross_game` rarity tier).

## How to run

```bash
PYTHONPATH=. python scripts/run_metrics.py --source pokemon_tcg
```
