# dominiontabs

Three single-card metrics over Dominion card data, resolving cards
through the same `CardBinder` [`../../card_binder/dominiontabs/`](../../card_binder/dominiontabs/)'s
`DominionTabsCardIngestionStage` populates. See `./BRAINSTORM.md` for
the full design rationale, including why this is only three metrics
(dominiontabs is a deliberately thin source - `raw_content` was
trimmed to six fields at ingestion time) and the sanity-checked
per-label counts each one was tuned against.

## Files

- `cost_regression_metric.py` - `CostRegressionMetric`: masks
  `["cost"]`, a `MaskedFieldRegressionMetric` ([`../generic/masked_field_regression_metric.py`](../generic/masked_field_regression_metric.py))
  consumer - the true label is a float, not a fixed-set class. Eligible
  only for the 624/819 cards whose raw cost is a plain digit string;
  a variable (`"6*"`), minimum (`"3+"`), or uncosted (`""`) card is
  excluded rather than coerced into a fallback number.
- `type_mask_metric.py` - `TypeMaskMetric`: masks `["types"]`,
  predicting only the primary (first-listed) type - `types` is a list
  on every card, too sparse to classify as a full combo. 14 classes
  (13 real primary types plus `OTHER_LABEL`, which the 6 smallest raw
  classes - Artifact, State, Curse, "Start Deck", Trash, Reaction, 11
  cards combined - fold into). Every card eligible.
- `set_mask_metric.py` - `SetMaskMetric`: NOT a `MaskedFieldMetric`
  subclass - `cardset_tags` was deliberately excluded from
  `raw_content` at ingestion, so this class reads
  `data/raw/dominiontabs/cards_db.json` directly and joins each row
  back to its `nocab_uuid` via `CardLookup.get_by_alias(GameId.DOMINION,
  DataSource.DOMINIONTABS, card_tag)`. Canonicalizes the raw
  `cardset_tags` (which mixes real expansion names with 1st/2nd-edition
  variant, "Removed"/"Upgrade", and promo noise) down to 17 real
  expansion classes, excluding the 29/819 cards (basics plus a few
  Guilds/Cornucopia box reprints) that canonicalize to more than one
  expansion - those cards have no single true "home expansion."
- `BRAINSTORM.md` - the design rationale and per-label sanity-check
  counts these three metrics were built and tuned against.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.dominiontabs.cost_regression_metric import (
    CostRegressionMetric,
)
from src.data_refinement.metrics.dominiontabs.type_mask_metric import (
    TypeMaskMetric,
)
from src.data_refinement.metrics.dominiontabs.set_mask_metric import SetMaskMetric

binder = CardBinder.load([Path("data/final/cards/dominion.jsonl")])
CostRegressionMetric(binder).scan()
TypeMaskMetric(binder).scan()
SetMaskMetric(binder).scan()  # also reads data/raw/dominiontabs/cards_db.json
```
