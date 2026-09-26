# encoder_model

The card embedding model itself: PyTorch code only, no data loading or
training. A card is serialized to compact JSON, encoded by a text
encoder, and projected to one embedding by an embedding head. The
multi-card model adds self-attention across the cards of one input, so
each card's embedding is contextualized by the rest of its list.

## Files

- `card_serialization.py` - `serialize_card_to_json(card)`: a card's
  `raw_content` as compact JSON, with no per-game logic (what goes in
  `raw_content` is decided at ingestion; see
  `../data_refinement/card_binder/README.md`).
- `text_encoder.py` - `TextEncoder` Strategy, texts -> `TokenEncoding`.
  `PretrainedTextEncoder` wraps a Hugging Face model (ModernBERT-base by
  default, `max_length=384`, optionally frozen) and batches texts by
  length to limit padding; `StaticEmbeddingTextEncoder` is a small
  from-scratch alternative.
- `embedding_head.py` - `EmbeddingHead` Strategy, `TokenEncoding` -> one
  vector per card: `LinearEmbeddingHead`, `ResidualMlpEmbeddingHead`,
  `AttentionPoolingEmbeddingHead`.
- `single_card_model.py` - `SingleCardModel(text_encoder, embedding_head)`:
  one embedding per card, accepting every input shape in
  `src/schema/type_hints.py` (single card, list, list of lists, ...) and
  returning embeddings in the same nesting.
- `multi_card_model.py` - `MultiCardModel`: the same, plus a
  Transformer encoder across the cards of each list.
- `reference_singlecard_models.py` / `reference_multicard_models.py` -
  preset (text encoder, head) pairings, e.g. `LinearProjectionCardModel`
  (frozen ModernBERT + linear head, the floor baseline). Presets are for
  convenience; an experiment injects its own pair.

## How to run

```python
from src.encoder_model.reference_singlecard_models import LinearProjectionCardModel

model = LinearProjectionCardModel(embed_dim=256)
embedding = model(card)            # one GenericCard -> one tensor
embeddings = model([card_a, card_b])  # list in -> list out, same order
```
