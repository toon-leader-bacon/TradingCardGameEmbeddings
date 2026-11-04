# TODO (dojos)

- Reorganize and clean up the dojos module. At time of writing, the dojos directory contains 10+ sub directories. The majority of these are metric specific organizational structure and largely mirrors data_refinement/metrics.
  However, mixed in are common shared utilities and the `contrastive` dojos. Here's what I'm envisioning for the dojos module:
  - A dedicated sub directory that maps similarly to the data_refinement/metrics directory and hide/ organize the bulk of these directories (and clearly illustrate the coupling with the data_refinement/metrics code). I'm not sure of the name, but maybe `dojos/dojos/*` is fine?
  - The `dojos/dojos/` directory is a top level module, that is comparable to the `dojos/contrastive` module. I see these two types of dojos, contrastive and regular? standard? decoder-head? (the name for these dojos is not clear to me) as the two main splits and we should represent that in our dojos module directory layout.
  - The dojos/dojos directory (Ok we've got to come up with a better name than that) should also separate the common infrastructure from the metrics mirror. So we'll want to separate out the loss, mods, generic, file_managers, etc from the scryfall, 17 lands, sts_gg etc. directories.
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
