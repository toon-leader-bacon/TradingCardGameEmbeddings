# TODO (dojos)

- **Other deck-derived dojos should use the holdout's deck split.**
  Contrastive dojos take each deck's split from
  `HoldoutSpec.deck_split_of`. Other data built from decks still splits
  its rows on its own, through parquet split files: the FaB
  hero-from-deck dojo, the held-out-deck-card metric
  (`src/data_refinement/metrics/generic/held_out_deck_card/`) and the StS2
  run metrics. A deck can be TRAIN for one and VALIDATION for another, so
  deck-level results across dojos can't yet be read as transfer
  (`experiments/TODO.md`, "How deck splits interact across dojos").
- **Smarter game sampling for `RarityTierDojo`.** The MVP draws each TRAIN
  row by picking a game uniformly, then one of its rows
  (`file_managers/game_balanced_chunk_reader.py`). That repeats StS2's few
  hundred cards many times per pass and under-uses MTG's 35k. Options:
  a temperature between uniform and proportional, or weights by how much
  each game's rarity actually varies. TEST and VALIDATION are read in file
  order, unbalanced.
- **Train `RarityTierDojo` on OTHER rows.** They are dropped now: the label
  is mostly "this is an StS2 non-card", a shortcut to the card's game
  (`cross_game/rarity_tier_data_constructor.py`).
- **A harder rarity task.** With `rarity` masked, a Pokemon card's `name` and
  `subtypes` still show the ex / V / GX mechanic (about 13% of Pokemon rows,
  nearly all TIER_4). Add noise mods if the task proves too easy.
- **Contrastive batch building is slow on the CPU; find ways to speed it
  up.** Seen in experiment 2026-10-11_model_by_diet_r2
  (`analysis/zero_shot_contrastive.py`): scoring the 30 contrastive dojos
  on VALIDATION (at most 1,024 examples each) took about 8-10 min per
  frozen encoder, 12 encoders, about 1.5-2 h in all. The GPU only ran the
  embedding head there (frozen text-encoder output was cached per card), so
  most of the time is probably building deck examples on the CPU. **Not
  yet profiled.**
  - **Profile first.** Where does a `ContrastiveDojo.batches()` pass spend
    its time: dealing decks, pair construction
    (`contrastive/pair_constructor.py`), staple subsampling, mods,
    serialization or collation?
  - **Build evaluation batches once and reuse them.** A capped
    VALIDATION/TEST pass is deterministic, so every encoder (and every
    training round's TEST pass) rebuilds identical batches. Cache the built
    examples per dojo and split.
  - **Drop undealable decks once, not on every pass.** The same decks are
    skipped and logged on every pass ("Skipping deck ...: N known card(s),
    need M"; 255 lines in that run's log). Filter them when splits are
    assigned, or cache the result.
  - **Parallelize building:** worker processes or a prefetch thread, so
    the CPU builds the next batches while the GPU runs the current ones.
    This applies to training too, where throughput is a stated priority.
- **Bounded, cacheable card-field mods (`variants`).** Every mod in
  `mods/card_field_mods.py` draws fresh on each call and stamps its card
  do-not-cache, so a modded card is encoded live (about 5 ms a card,
  against about 20 us for the mod itself). With the default masks on,
  40% of a 512-card step was encoded live: 642 ms against 182 ms with
  mods off (`src/training/ENV_SIZING.md`). With `ShuffleKeysMod` on,
  about 78% missed, so it was dropped from the defaults. Proposed design
  (discussed 2026-10-21, not built):
  - **Scope.** Only the three mods that write new text:
    `ShuffleKeysMod`, `RandomKeyMaskMod`, `WeightedFieldMaskMod`.
    `MaskTargetKeyMod` is already deterministic. Deck mods and
    `ShuffleDeckMod` only choose which cards appear, and every kept card
    is still a cache hit, so they stay as they are.
  - **`variants: int | None`** on each of the three mods and their specs
    (`{kind: weighted_field_mask, variants: 3, ...}`), capped at about
    100. None keeps today's behavior: fresh draws, stamped. A run-level
    `mod_variants:` override can come with experiment F
    (`experiments/TODO.md`), which sweeps it.
  - **Whether to fire stays fresh; what to do is a variant.** The mod's
    own rng decides whether it fires (`RandomKeyMaskMod.probability`,
    and `WeightedFieldMaskMod`'s empty "mask nothing" rows, pulled out of
    the variant draw) and picks v in 0..variants-1. What it does is a
    pure function of (card, v): which mask, which key, which order. Each
    card then has at most `variants` modded texts per mod, all cacheable.
    Chained mods multiply that number; the cache's slot policy evicts
    the rare ones.
  - **Keep it as fast as today.** Don't seed a `random.Random` per card
    (about 7 us). Draw the variant's randomness from a keyed hash,
    `blake2b(nocab_uuid.bytes + bytes([v]), key=salt, digest_size=8)`
    (about 0.8 us), or from a fixed table of seeds XORed with the uuid.
    Turn the number into a choice directly (an index into the card's
    keys, a permutation, or a new `DropTable.pull_at(fraction)` instead
    of `pull(rng)`). The salt is fixed, not the dojo's `rng_seed`, so
    every dojo and every run gets the same variants of a card and reuses
    the disk tier. No memo of built variant cards: it would be a second
    cache to save about 20 us against a 5 ms encode.
  - **Check:** a card built from a card that already carries the
    do-not-cache stamp must keep it.
  - **Later:** a mod with fixed variants can list every variant of a
    card, so the feature cache could warm the modded corpus before
    training (about 5 min per unit of `variants`). Do it only if the
    first-draw misses show up in a profile.
- **`CardBinder` deep-copies every card it returns** (about 45 us a card,
  2026-10-21, more than twice today's mod cost). Cards are frozen and
  mods never mutate, so the copy may be removable
  (`src/data_refinement/card_binder/card_binder.py`). Confirm how much it
  costs in a real batch-building profile first.
