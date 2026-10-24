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
    Measured 2026-10-22 (`../training/ENV_SIZING.md`, "Training step"):
    with the GPU work down to ~20-50 ms a step, 78% of a training step
    waits on the next batch, and single-card metric dojos over whole
    games wait 0.9-2.3 s per step. The prefetch depth of 2 no longer
    hides building; profile those dojos' builders first.
- **Padded heads for multi-group dojos.** Multi-group dojos still receive
  nested lists (`ModelEmbeddings`' second arm); their heads pack each group
  with `generic/group_batching.py`. Two follow-ups:
  - A padded multi-group type (`(examples, groups, max_cards, d)` or
    groups flattened with an example index), so the heads skip the
    per-card `from_nested` pack.
  - Budget: `group_by_budget` charges examples x the costliest example's
    total cards, but the encoder pads every group to the longest group in
    the batch. A deck-pair batch can then hold up to (groups per example)
    times more attention slots than the budget assumes. Charge
    (examples x groups) x longest group for multi-group inputs.
- **Bounded mod variants: follow-ups.** `variants:` is built for the
  three card-field mods (`mods/variant_seeds.py`, `{kind: ..., variants:
  N}`); every default is still unbounded (-1). Open:
  - **Turn it on.** Pick N per mod and put it in
    `augmentation_defaults.py`, and put `ShuffleKeysMod` back in the
    defaults once bounded. That changes what the model trains on, so it
    goes through experiment F (`experiments/TODO.md`), which can sweep N.
  - **Cache admission and eviction.** Chained bounded mods multiply a
    card's texts (field mask x key mask). Tune the tiers' slot policies
    (`encoder_model/feature_cache/slot_policy.py`) for the balance
    between noise and cache hits; the disk tier only writes during warm
    today, so modded texts are not kept across runs.
  - **Warm the modded corpus.** A bounded mod can list every variant of
    a card, so warm() could encode them before training (about 5 min per
    unit of N). Only if first-draw misses show up in a profile.
  - **Cost.** Seeding a `random.Random` per card is about 7 us; a keyed
    hash turned straight into a choice would be about 1 us, if a
    batch-building profile ever shows it.
