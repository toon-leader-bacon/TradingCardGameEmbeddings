# single_card

`SingleCardModel`: one card in, one embedding out.

## Files

- `single_card_model.py` — `SingleCardModel`, a `typing.Protocol`:
  `forward(card) -> Embedding`, plus `__call__`/`parameters()`
  (declared explicitly so `training/` can call `model(card)` and
  `model.parameters()` against the Protocol type itself, without
  needing to know a concrete implementation is actually a
  `torch.nn.Module`) and an `embedding_dim` property (the model's
  fixed output width, set at construction time).
- `toy_single_card_model.py` — `ToySingleCardModel`, a minimal
  `torch.nn.Module` implementation for exercising the
  `SingleCardModel`/`Dojo`/`Trainer` contract end to end — not a
  serious architecture. Embeds a card by hashing its name (via
  `hashlib.sha256`, not Python's built-in `hash()`, which is
  randomized per process) into a fixed-size `nn.Embedding` lookup
  table.

## How to run

```python
from src.encoder_model.single_card.toy_single_card_model import ToySingleCardModel

model = ToySingleCardModel(embedding_dim=32, vocab_size=4096)
embedding = model(card)  # shape (32,)
```
