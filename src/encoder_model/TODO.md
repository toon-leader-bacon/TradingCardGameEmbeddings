# TODO (encoder_model): beyond the frozen linear baseline

Started 2026-09-29. Every run so far uses a frozen pretrained text
encoder (`answerdotai/ModernBERT-base`), mean pooling and a small head.
Mean pooling a masked-LM's token states blurs exactly the details cards
turn on (a 7 vs an 8, which keyword attaches to which clause, negation),
and no depth after the pool can recover them. The trainable poolings in
`pooling/` (multi-query attention, transformer layers over the tokens)
exist but are not yet compared; see `README.md` and
`experiments/TODO.md`.

- [ ] **Consider alternative text encoders.** ModernBERT-base was chosen
  in `src/training/Notes.md` (license, 8k context, code-heavy
  pretraining). Worth comparing, on the same dojos and the evaluation
  suite (`plans/archive/evaluation.md`): embedding-tuned models
  (`nomic-ai/modernbert-embed-base`, which needs its prefixes;
  `jinaai/jina-embeddings-v2-base-code`, which needs
  `trust_remote_code`), ModernBERT-large if VRAM allows, and a small
  decoder model's hidden states. Re-run `scripts/report_card_content.py`
  for token counts under any new tokenizer.
- [ ] **Cache frozen token states for the trainable poolings.**
  `AttentionPooling` and `TransformerPooling` read every card's token
  states each step. With the LM frozen those never change, so caching
  them makes pooling training far faster (fp16 token states: gwent
  ~175 MB, MTG ~9 GB; not built yet). Caching only covers cards the dojo passes through unmodified;
  mods that change card text (masking, key shuffles) bypass it.
- [ ] **Whitening after pooling.** `TokenWhitening` whitens
  tokens before pooling only. Offline, whitening the mean-pooled card
  vector instead scored best (`experiments/2026-10-10_pooling_probe/NOTEBOOK.md`). After a
  fixed pooling it is fit once (`CovarianceAccumulator` on pooled
  vectors); after a trainable pooling its statistics drift as the
  pooling trains, so it needs a recalibration policy: every N steps or
  rounds, at phase boundaries, or a running estimate (BatchNorm-style).
  Decide the cadence, and whether a trainable head right after it makes
  it redundant.
- [ ] **A card encoder that reads the token sequence directly.** Shelved
  2026-10-10 when the model was split into text
  encoder -> optional token enrichment -> token pooling -> embedding
  head. The alternative skips pooling: the card encoder (or the
  multi-card attention stack) consumes every card's token states, so no
  detail is averaged away before cards meet each other. Much more
  compute per group (cards x tokens instead of cards). Revisit once the
  pooling arms are measured.
- [ ] **Train a model with a thawed text encoder.** `PretrainedTextEncoder`
  already takes `trainable`, and `Phase.encoder_trainable` / `encoder_lr`
  exist, but no reference model or run config sets them. Build one (likely
  a frozen head-only phase first, then a thawed phase at a small
  `encoder_lr`) and practice: VRAM at 256/384 tokens (measurements in
  `src/training/Notes.md`: bs 16 at 256 tokens ~5.6 GiB, bs 32
  ~9.2 GiB; 512 tokens needs gradient checkpointing past bs 16), gradient
  checkpointing, the fp16 GradScaler on the real LM, and watching for
  Windows silently spilling VRAM into system RAM.
