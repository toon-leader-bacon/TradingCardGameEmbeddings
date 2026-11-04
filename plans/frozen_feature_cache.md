# Frozen feature cache (plan)

Status: v1 to v1.3 are built (pooled cut; VRAM, RAM and disk tiers;
scheduled prefetch with pinning; slot policies; the do-not-cache stamp;
the token-level cut with trainable text layers). How they work, and
their known limits, is in `src/encoder_model/feature_cache/README.md`.
Open engineering items are in `src/encoder_model/TODO.md`. This plan
now holds only the v1.4 direction; the full v1-v1.3 plan is in git
history (before 2026-11-03).

## Goal

Make training faster by caching what the model's frozen stages output,
so the same card text is not encoded again every step. The corpus is
expected to grow far past VRAM and RAM, so every choice assumes the
working set does not fit in the top tier.

## Where this meets the teacher-student regime

`plans/teacher_student_training.md`: Stage B trains the text encoder
over ~65k cards, either fully thawed or above a layer-k token cut (this
cache's v1.3). Stage C caches the frozen student's pooled vectors (the
pooled cut, now holding student outputs).

## v1.4: direction

- D: train from cached features and match an uncached run
  (`experiments/TODO.md`). The first comparison is in
  2026-10-21_cache_shakeout (H4, awaiting review).
- F: a fixed pool of K augmentations per card, which makes noise mods
  cacheable. Matters most in Stage B, where each draw of a card can be a
  new modded variant.
- Locality-aware sampling: draw N decks (or one game's cards) per
  episode, and train many batches from that working set, sized to the
  cache. This gives up some diet mixing for throughput, perhaps with a
  lower learning rate. Prior art: GNS and ClusterGCN from GNN training.
  The round schedule is the natural place to express it. Stage B's
  single-game batches already lean this way.
