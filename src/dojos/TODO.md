# TODO (dojos)

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
- **A capped TRAIN pass is not reproducible for `RarityTierDojo`.** Its
  `_chunks` draws randomly from a shared seeded rng, so `max_examples` on
  TRAIN yields a different prefix each pass.
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
