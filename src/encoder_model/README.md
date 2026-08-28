# encoder_model

Defines the embedding network architecture(s) themselves — the actual
PyTorch model code — and nothing else. This container has no knowledge
of dojos, training loops, hyperparameter search, or data loading; it
only knows how to turn a card (or a list of cards) into embedding(s).
Keeping it this narrow is deliberate: it's the one artifact of this
project actually intended for publication, so its boundary needs to
stay clean of training-specific concerns that would make it harder to
lift out and reuse on its own.

Two model types, each its own Protocol:

- `single_card/` — `SingleCardModel`: one card in, one embedding out.
- `multi_card/` — not yet built. `MultiCardModel`: a list of cards in,
  one embedding per input card out, same order, never pooling
  internally (pooling is always a dojo-side concern) — see
  `plans/encoder_model.md`.

## Files

- `embedding.py` — `Embedding = torch.Tensor`, the shared output type
  both model Protocols return.
- `single_card/` — `SingleCardModel` and its first implementation. See
  `single_card/README.md`.

## How to run

```python
from src.encoder_model.single_card.toy_single_card_model import ToySingleCardModel

model = ToySingleCardModel(embedding_dim=32, vocab_size=4096)
embedding = model(card)  # shape (32,)
```
