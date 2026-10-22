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
- **A bounded, cacheable key shuffle.** `ShuffleKeysMod` makes a new key
  order on every draw, so each shuffled card is a text never seen before.
  It is stamped do-not-cache and is encoded live every step; with it on,
  about 78% of training cards missed the feature cache (2026-10-21). It
  was taken out of the default augmentations
  (`augmentation_defaults.py`) for that reason. Build a shuffle that
  picks one of N fixed orders per card instead, e.g. seed a
  `random.Random` with (card text digest, k) for k drawn uniformly from
  0..N-1. Each card then has at most N shuffled texts, all cacheable
  (`deterministic_cards = True`), and the cache holds N + 1 copies of each
  card rather than an unbounded number. Make N a spec field
  (`{kind: shuffle_keys, orders: 4}`), then put it back in the defaults.
  `WeightedFieldMaskMod` is bounded the same way (one text per drop-table
  row), so it could become cacheable too. It changes about 75% of cards
  and still stamps them, so cached runs with default mods still encode
  most cards live until it does.
