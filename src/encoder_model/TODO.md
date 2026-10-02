# TODO (encoder_model): beyond the frozen linear baseline

Started 2026-09-29. Today every reference single-card model is a frozen
pretrained text encoder (`answerdotai/ModernBERT-base`) plus a small head,
and both "rich" heads see little of the frozen encoder's output:
`ResidualMlpEmbeddingHead` mean-pools the token states before its MLP, and
`AttentionPoolingEmbeddingHead` is one learned query plus one linear layer.
Mean pooling a masked-LM's token states blurs exactly the details cards
turn on (a 7 vs an 8, which keyword attaches to which clause, negation),
and no depth after the pool can recover them.

- [ ] **Consider alternative text encoders.** ModernBERT-base was chosen
  in `src/training/TODO.md` section B (license, 8k context, code-heavy
  pretraining). Worth comparing, on the same dojos and the evaluation
  suite (`plans/archive/evaluation.md`): embedding-tuned models
  (`nomic-ai/modernbert-embed-base`, which needs its prefixes;
  `jinaai/jina-embeddings-v2-base-code`, which needs
  `trust_remote_code`), ModernBERT-large if VRAM allows, and a small
  decoder model's hidden states. Re-run `scripts/report_card_content.py`
  for token counts under any new tokenizer.
- [ ] **Consider different head architectures.** Beyond linear, residual
  MLP and single-query attention pooling: multi-query attention pooling
  (several learned queries, concatenated or mixed), and the token-level
  head below. Compare every head against `LinearProjectionCardModel` (the
  floor baseline) on the same run config.
- [ ] **Design a token-level head.** A head that works on the frozen
  token sequence (`TokenEncoding.hidden_states` plus `attention_mask`)
  rather than its mean: e.g. 1-2 small transformer layers over the frozen
  states, then pooling (a CLS-style learned token or attention pooling).
  New class, so it goes through design-recipe-skeleton. Pairs naturally
  with caching frozen encoder outputs (`src/training/TODO.md` notes the
  cache is not built): with the LM frozen, its per-card token states never
  change, so caching them makes head training far faster (fp16 token
  states: gwent ~175 MB, MTG ~9 GB). Caching only covers cards the dojo
  passes through unmodified; mods that change card text (masking, key
  shuffles) bypass it.
- [ ] **Train a model with a thawed text encoder.** `PretrainedTextEncoder`
  already takes `trainable`, and `Phase.encoder_trainable` / `encoder_lr`
  exist, but no reference model or run config sets them. Build one (likely
  a frozen head-only phase first, then a thawed phase at a small
  `encoder_lr`) and practice: VRAM at 256/384 tokens (measurements in
  `src/training/TODO.md` section A: bs 16 at 256 tokens ~5.6 GiB, bs 32
  ~9.2 GiB; 512 tokens needs gradient checkpointing past bs 16), gradient
  checkpointing, the fp16 GradScaler on the real LM, and watching for
  Windows silently spilling VRAM into system RAM.
