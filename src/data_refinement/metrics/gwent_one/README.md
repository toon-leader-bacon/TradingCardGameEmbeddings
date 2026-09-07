# gwent_one

Eight `MaskedFieldMetric` ([`../masked_field_metric.py`](../masked_field_metric.py))
masking metrics over gwent.one card data, resolving cards through the
same `CardBinder` [`../../card_binder/gwent_one/`](../../card_binder/gwent_one/)'s
`GwentOneCardIngestionStage` populates - this project's first
`CorpusScanMetric`-family consumer (see [`../README.md`](../README.md)
for why this family exists, distinct from `sts_gg/`'s accumulator/
streaming `Metric[dict]` family). Each metric masks one `data-*`
attribute off a card and predicts its true value as a fixed-set
classification label - every field value confirmed directly against
the live 1,260-card `data/raw/gwent_one/page_1.html` corpus, not
guessed.

## Files

- `faction_mask_metric.py` - `FactionMaskMetric`: masks `["faction"]`.
  7 classes (`neutral`, `syndicate`, `monster`, `scoiatael`,
  `nilfgaard`, `northern_realms`, `skellige`). Every card eligible.
- `color_mask_metric.py` - `ColorMaskMetric`: masks `["color"]`. 3
  classes (`gold`, `bronze`, `leader` - power-budget tier, distinct
  from rarity). Every card eligible.
- `type_mask_metric.py` - `TypeMaskMetric`: masks `["type"]`. 5 classes
  (`unit`, `special`, `artifact`, `ability`, `stratagem`). Every card
  eligible.
- `rarity_mask_metric.py` - `RarityMaskMetric`: masks `["rarity"]`. 4
  classes (`legendary`, `epic`, `rare`, `common` - true collector
  rarity, distinct from color). Every card eligible.
- `set_mask_metric.py` - `SetMaskMetric`: masks `["set"]`. 12 classes
  (`baseset` plus 11 named expansions). Every card eligible.
- `provision_mask_metric.py` - `ProvisionMaskMetric`: masks
  `["provision"]`. Classes are `str(value)` for the observed dense
  range `"0"`, `"4"`..`"18"`, plus `OTHER_LABEL` for anything else.
  Excludes `type == "stratagem"` - confirmed against the full corpus,
  `provision="0"` is exactly and only the 12 stratagem cards, not a
  real cost for that type.
- `power_mask_metric.py` - `PowerMaskMetric`: masks `["power"]`.
  Classes are `str(value)` for `"0"`..`"17"` plus `OTHER_LABEL`.
  Restricted to `type == "unit"` - confirmed against the full corpus,
  `power="0"` is exactly and only the 323 non-unit cards.
- `armor_mask_metric.py` - `ArmorMaskMetric`: masks `["armor"]`.
  Classes are `str(value)` for the observed set `"0"`, `"1"`..`"6"`,
  `"10"` plus `OTHER_LABEL`. Restricted to `type == "unit"` - nonzero
  armor is observed exclusively on unit cards, but unlike power/
  provision, `armor="0"` stays a valid, common label for an eligible
  unit rather than being excluded (most units are legitimately
  0-armor).
- `BRAINSTORM.md` - candidate metrics not yet built from this raw
  source. Faction/provision/power/color/rarity/type/set/armor
  prediction (items 1-6, 10, 11) are now built here, superseding their
  original "predict from ability text" framing with the masked-field
  reference-row shape described above; the rest are still unbuilt
  ideas.

## Out of scope

`card-category` (`.card-category`, comma-joined, 1-4 tokens/card, 59
distinct tokens - e.g. `"Human, Cultist"`) is NOT a `MaskedFieldMetric`
candidate: it's genuinely variable-length multi-label (no clean
fixed-position split, e.g. into "race" vs. "job" - tags like
`Witcher`/`Spell`/`Wild Hunt` have no second token at all), a different
metric shape than this family's single-true-label design. Would need
its own multi-label metric shape if built later.

The Dojo-side consumer of these metrics (a `DataConstructor` that
resolves `nocab_uuid` via `CardBinder`, walks `masked_field` into
`raw_content`, replaces that leaf with a mask sentinel, and builds
model input) does not exist yet - these metrics only produce the
reference parquet files; no `Dojo` reads them yet.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.gwent_one.faction_mask_metric import (
    FactionMaskMetric,
)
from src.data_refinement.metrics.gwent_one.provision_mask_metric import (
    ProvisionMaskMetric,
)
from src.data_refinement.metrics.gwent_one.power_mask_metric import (
    PowerMaskMetric,
)
from src.data_refinement.metrics.gwent_one.armor_mask_metric import (
    ArmorMaskMetric,
)
from src.data_refinement.metrics.gwent_one.color_mask_metric import (
    ColorMaskMetric,
)
from src.data_refinement.metrics.gwent_one.rarity_mask_metric import (
    RarityMaskMetric,
)
from src.data_refinement.metrics.gwent_one.type_mask_metric import (
    TypeMaskMetric,
)
from src.data_refinement.metrics.gwent_one.set_mask_metric import SetMaskMetric

binder = CardBinder.load([Path("data/final/cards/gwent.jsonl")])
metrics = [
    FactionMaskMetric(binder),
    ColorMaskMetric(binder),
    TypeMaskMetric(binder),
    RarityMaskMetric(binder),
    SetMaskMetric(binder),
    ProvisionMaskMetric(binder),
    PowerMaskMetric(binder),
    ArmorMaskMetric(binder),
]
for metric in metrics:
    metric.scan()  # each metric's DEFAULT_OUTPUT_PATH now exists
```
