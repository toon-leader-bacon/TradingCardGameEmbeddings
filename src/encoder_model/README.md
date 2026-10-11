# encoder_model

The card embedding model itself: PyTorch code only, with no data loading
and no training. A card goes through a `CardEmbeddingPipeline` of four
injected stages:

```
text encoder -> [token enrichment] -> token pooling -> embedding head
texts           token states          one vector       one embedding
                (optional)            per card         per card
```

The multi-card model adds self-attention across the cards of one group,
so each card's embedding is contextualized by the rest of its group.

## Files

- `card_serialization.py` - `serialize_card_to_json(card)`: a card's
  `raw_content` as compact JSON, with no per-game logic (what goes in
  `raw_content` is decided at ingestion; see
  `../data_refinement/card_binder/README.md`).
- `text_encoder.py` - `TextEncoder` Strategy, texts -> `TokenEncoding`
  (token states plus attention mask), with `output_dim` (the token
  width). `PretrainedTextEncoder` wraps a Hugging Face model
  (ModernBERT-base by default, `max_length=384`, optionally frozen) and
  batches texts by length to limit padding. `StaticEmbeddingTextEncoder`
  is a small from-scratch alternative.
- `enrichment/` - `TokenEnrichment` Strategy, `TokenEncoding` ->
  `TokenEncoding` (same mask, width may change), the pipeline's optional
  slot. `TokenWhitening(stats)`: `(x - mean) @ projection` on every token,
  in float32 with autocast off (the projection's large scales would
  otherwise run through an fp16 matmul, amplifying its rounding); mean
  and projection are non-trainable buffers, so they are saved in
  `encoder.pt`.
- `whitening/` - fitting a token whitening, in three reusable parts:
  `sample_calibration_cards(card_lookup, CalibrationSampleOptions)` picks
  up to `cards_per_game` TRAIN-tier cards per game under a `HoldoutSpec`
  (held-out games are never TRAIN; each game's unknown-card placeholder
  is skipped) and returns a `CalibrationSample` (the cards plus the
  holdout that picked them); `CovarianceAccumulator` streams a float64
  mean and covariance of any vectors; `fit_token_whitening(text_encoder,
  sample, WhiteningFitOptions)` encodes the sample exactly as the
  pipeline serializes cards and returns `WhiteningStats` - the PCA
  whitening of every real token (top `components`, default all; each
  eigenvalue regularized by adding `relative_eps` times the largest), with the
  checkpoint and holdout it came from. `WhiteningStats.save`/`load` use a
  `.pt` file; `require_fitted_on(checkpoint)` refuses another encoder.
  Command line: `scripts/fit_whitening.py --holdout-from <run config>
  --out <path>`.
- `fixed_width_stage.py` - `FixedWidthStage`, the shared base of the
  enrichment, pooling and head interfaces: validated, read-only
  `input_dim` / `output_dim`.
- `pooling/` - `TokenPooling` Strategy, `TokenEncoding` -> one vector per
  card, ignoring padding whatever it holds (`zero_padding` selects rather
  than multiplies, so NaN at padding cannot leak): `MeanPooling` (no
  parameters),
  `AttentionPooling(num_queries)` (learned queries, each a softmax-weighted
  average of the tokens; outputs concatenated, so `output_dim` is
  `num_queries * input_dim`), and `TransformerPooling(inner)` (trainable
  pre-norm transformer layers over the tokens, then a mean or attention
  pooling).
- `embedding_head.py` - `EmbeddingHead` Strategy, one pooled vector per
  card -> one embedding: `LinearEmbeddingHead`, `ResidualMlpEmbeddingHead`.
  Every head's output ends in a LayerNorm with no gain or bias (mean 0,
  variance 1 across the coordinates, L2 norm sqrt(`output_dim`)), so the
  embedding scale cannot drift; the multi-card model's self-attention
  stack ends in the same norm.
- `card_embedding_pipeline.py` - `CardEmbeddingPipeline(text_encoder,
  enrichment, pooling, embedding_head)`: the one place the stages are
  chained. `embed(cards)` serializes, encodes, enriches (when an
  enrichment is given), pools and projects; every card is independent.
  Every stage reports `input_dim`/`output_dim` (the text encoder only
  `output_dim`), and the constructor raises `ValueError` if adjacent
  widths differ. Its `output_dim` is the one source of a model's
  embedding width.
- `card_encoder_model.py` - `CardEncoderModel`, the Template Method base
  of both models. A subclass supplies `embed_together(cards)`, for cards
  of one group that may see each other, and `embed_apart(cards)`, for
  unrelated cards in one batched pass. The base maps every input shape
  in `src/schema/type_hints.py` onto those two. It has one explicit
  method per shape (`forward_single_card`, `forward_multi_card`,
  `forward_multi_group`, `forward_batched_single_card`,
  `forward_batched_multi_card`, `forward_batched_multi_group`) plus the
  canonical `forward`. Also `embedding_dim` and
  `isolated_embeddings(cards, precision)`: every card embedded on its
  own, as a float32 `(N, embedding_dim)` CPU tensor, deterministically
  (eval mode) - the context-free view evaluation stores and analyses.
- `single_card_model.py` - `SingleCardModel(card_pipeline)`: every card
  embedded on its own by the pipeline, so `embed_together` and
  `embed_apart` are the same.
- `multi_card_model.py` - `MultiCardModel(card_pipeline, num_heads,
  num_layers, ...)`: the pipeline, then a Transformer encoder at the
  pipeline's `output_dim`. `embed_together` attends across the group;
  `embed_apart` attends over length-one sequences.
- `precision.py` - `Precision` (`"fp32"`/`"fp16"`/`"bf16"`),
  `autocast_for(model, precision)`, and `inference_context(model,
  precision)` (eval mode + no_grad + autocast, prior mode restored). Shared
  by `training/` and `isolated_embeddings`.
- `reference_card_pipelines.py` - preset pipelines as factory functions
  (`build_linear_projection_pipeline`, `build_residual_mlp_pipeline`,
  `build_attention_pooling_pipeline`, `build_from_scratch_pipeline`),
  shared by both models' presets.
- `reference_singlecard_models.py` / `reference_multicard_models.py` -
  preset models around those pipelines, e.g. `LinearProjectionCardModel`
  (frozen ModernBERT, mean pooling, linear head: the floor baseline).
  Presets are for convenience; an experiment builds its own pipeline
  (`scripts/run_training.py`, `build_card_pipeline`).
- `TODO.md` - next steps: alternative text encoders, head architectures,
  whitening after pooling, a card encoder over raw tokens, and a
  thawed-encoder model.

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

model = LinearProjectionCardModel(card_embedding_size=256)
embeddings = model([card_a, card_b])        # batch of single cards -> one tensor each
embedding = model.forward_single_card(card)  # one unbatched card -> one tensor
deck = model.forward_multi_card([a, b, c])   # one deck, cards contextualized together
table = model.isolated_embeddings(cards, "fp16")  # (len(cards), 256) float32, no context
```
