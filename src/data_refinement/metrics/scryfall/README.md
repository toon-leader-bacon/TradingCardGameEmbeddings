# scryfall

Six single-card masking metrics over the MTG `CardBinder`
(`../../card_binder/scryfall/`), in `card_mask_metrics.py`. Each one
names a field of the leaned `raw_content`, which cards are eligible, and
the label; the paired dojos (`src/dojos/scryfall/card_mask_dojos.py`)
mask that field plus the keys that would give it away, on every split.

Counts are from the live binder (34,710 cards, checked 2026-10-01).
Every metric except rarity skips the 898 multi-face cards: each entry in
`card_faces` repeats its own mana cost, type line, colors and stats.

| Metric | Field | Label | Rows | Also masked by the dojo |
|---|---|---|---|---|
| `CmcRegressionMetric` | `cmc` | regression, 0-16 | 32,359 | `mana_cost` |
| `CardTypeMaskMetric` | `type_line` | 7 main types + OTHER, by priority (Creature first) | 33,812 | `power`, `toughness`, `loyalty`, `defense` |
| `RarityMaskMetric` | `rarity` | common/uncommon/rare/mythic + OTHER | 34,710 | - |
| `ColorsMaskMetric` | `colors` | multi-label over WUBRG (colorless = none) | 33,812 | `color_identity`, `mana_cost`, `color_indicator` |
| `PowerRegressionMetric` | `power` | regression, whole numbers 0-20 | 18,734 | - |
| `ToughnessRegressionMetric` | `toughness` | regression, whole numbers 0-20 | 18,797 | - |

Choices:

- cmc skips cards with no `mana_cost` (lands, suspend cards): their cmc
  is always 0 and the missing key gives it away. Gleemax (cmc 1,000,000)
  and the half-point Un-cards are skipped too.
- Colors are multi-label (`MaskedFieldMultiLabelMetric`, scored by
  `MASKED_VECTOR_REGRESSION_LOSS_SPEC`) rather than one class per color
  combination: 32 combinations, 26 of them under 1% of cards.
  `oracle_text` still mentions colors ("Add {G}"), so this mask never
  removes all color signal.
- Keywords are not built: on 16,517 of the 17,440 cards that have any,
  every keyword appears verbatim in `oracle_text`, so the task would be
  reading the text back.

## How to run

```bash
PYTHONPATH=. python scripts/run_metrics.py --source scryfall
```

`BRAINSTORM.md` holds the remaining ideas.
