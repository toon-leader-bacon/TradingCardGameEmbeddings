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
