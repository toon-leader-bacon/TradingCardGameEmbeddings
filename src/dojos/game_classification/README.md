# game_classification

`GameClassificationDojo`, the first concrete `Dojo`: predicts a
single card's `GameId` from its own embedding. The simplest possible
dojo — `CardCount(1, 1)`, and labels are read straight off each
card's own `source_game` field, so no join against any other data
source is needed.

## Files

- `game_classification_dojo.py` — `GameClassificationDojo`.
  `prepare_splits` partitions every card a `CardLookup` can enumerate
  into train/test/validate pools by two configured ratios (the
  remainder goes to test), excluding any `held_out_cards` uuid from
  the train pool entirely. `next_batch` draws length-1 `CardSet`s —
  with-replacement sampling for `Split.TRAIN`, an exhaustive
  reshuffling cursor for `Split.TEST`/`Split.VALIDATE`. `compute_loss`
  stacks a batch's embeddings through a cached `torch.nn.Linear`
  decoder head, maps each `GameId` label to its fixed class index, and
  delegates to this dojo's injected `Loss`.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import Split
from src.dojos.game_classification.game_classification_dojo import GameClassificationDojo
from src.dojos.losses.cross_entropy_loss import CrossEntropyLoss

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])

dojo = GameClassificationDojo(
    train_ratio=0.8,
    validate_ratio=0.1,
    loss=CrossEntropyLoss(label_smoothing=0.0),
    rng_seed=0,
)
dojo.prepare_splits(binder, held_out_cards=set())
batch = dojo.next_batch(Split.TRAIN, batch_size=32)
```
