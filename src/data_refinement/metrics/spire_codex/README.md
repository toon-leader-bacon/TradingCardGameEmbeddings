# spire_codex

Four single-card masking metrics over the Slay the Spire 2 `CardBinder`
(`../../card_binder/spire_codex/`), in `card_mask_metrics.py`, with
paired dojos in `src/dojos/spire_codex/card_mask_dojos.py` (see
[`../scryfall/README.md`](../scryfall/README.md) for the shape). The
binder holds 577 cards (checked 2026-10-01), so these dojos are small.

| Metric | Field | Label | Rows | Also masked by the dojo |
|---|---|---|---|---|
| `CostMaskMetric` | `cost` | "0"-"3" + OTHER | 537 | `upgrade.cost` |
| `CardTypeMaskMetric` | `type` | Attack / Skill / Power | 540 | - |
| `RarityMaskMetric` | `rarity` | Basic / Common / Uncommon / Rare / Ancient | 511 | - |
| `ColorMaskMetric` | `color` | the five characters + colorless | 503 | - |

Choices:

- Cost is a fixed class, not regression: 0-3 cover 99% of playable
  cards. Unplayable cards (cost -1) and X-cost cards are skipped.
  `upgrade` holds the upgraded cost on 57 cards, so `upgrade.cost` is
  masked too (the rest of `upgrade` stays visible).
- `rarity` mixes tier and category (Curse, Status, Event, Token,
  Quest), and `color` repeats the category for those cards; only the
  real tiers and character pools are eligible.

## How to run

```bash
PYTHONPATH=. python scripts/run_metrics.py --source spire_codex
```
