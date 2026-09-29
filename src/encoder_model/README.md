# encoder_model

The card embedding model itself: PyTorch code only, with no data loading
and no training. A card is serialized to compact JSON, encoded by a text
encoder, and projected to one embedding by an embedding head. The
multi-card model adds self-attention across the cards of one group, so
each card's embedding is contextualized by the rest of its group.

## Files

- `card_serialization.py` - `serialize_card_to_json(card)`: a card's
  `raw_content` as compact JSON, with no per-game logic (what goes in
  `raw_content` is decided at ingestion; see
  `../data_refinement/card_binder/README.md`).
- `text_encoder.py` - `TextEncoder` Strategy, texts -> `TokenEncoding`.
  `PretrainedTextEncoder` wraps a Hugging Face model (ModernBERT-base by
  default, `max_length=384`, optionally frozen) and batches texts by
  length to limit padding. `StaticEmbeddingTextEncoder` is a small
  from-scratch alternative.
- `embedding_head.py` - `EmbeddingHead` Strategy, `TokenEncoding` -> one
  vector per card: `LinearEmbeddingHead`, `ResidualMlpEmbeddingHead`,
  `AttentionPoolingEmbeddingHead`.
- `card_encoder_model.py` - `CardEncoderModel`, the Template Method base
  of both models. A subclass supplies `embed_together(cards)`, for cards
  of one group that may see each other, and `embed_apart(cards)`, for
  unrelated cards in one batched pass. The base maps every input shape
  in `src/schema/type_hints.py` onto those two. It has one explicit
  method per shape (`forward_single_card`, `forward_multi_card`,
  `forward_multi_group`, `forward_batched_single_card`,
  `forward_batched_multi_card`, `forward_batched_multi_group`) plus the
  canonical `forward`.
- `single_card_model.py` - `SingleCardModel(text_encoder, embedding_head)`:
  every card embedded on its own, so both building blocks are the same.
- `multi_card_model.py` - `MultiCardModel`: the same, plus a Transformer
  encoder. `embed_together` attends across the group; `embed_apart`
  attends over length-one sequences.
- `reference_singlecard_models.py` / `reference_multicard_models.py` -
  preset (text encoder, head) pairings, e.g. `LinearProjectionCardModel`
  (frozen ModernBERT + linear head, the floor baseline). Presets are for
  convenience; an experiment injects its own pair.

## Input shapes

`model(x)` (`forward`) always reads `x` as a batch, which is what
`Trainer` and the dojos pass. A flat list of cards is a batch of single
cards, and each is embedded without seeing the others, even in
`MultiCardModel`. A list of lists is a batch of multi-card examples,
each contextualized on its own. A bare `GenericCard` is not a batch and
raises `TypeError`. Code that holds one unbatched example calls the
explicit method for its shape.

## How to run

```python
from src.encoder_model.reference_singlecard_models import LinearProjectionCardModel

model = LinearProjectionCardModel(embed_dim=256)
embeddings = model([card_a, card_b])        # batch of single cards -> one tensor each
embedding = model.forward_single_card(card)  # one unbatched card -> one tensor
deck = model.forward_multi_card([a, b, c])   # one deck, cards contextualized together
```
