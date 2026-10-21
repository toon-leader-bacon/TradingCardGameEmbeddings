# TODO (data retrieval)

- [ ] **More Pokemon decks.** The only deck source, `pokemon_tcg/` (the
  pokemon-tcg-data repo), holds prefab theme decks: about 160 TRAIN and
  11 TEST decks under the default deck split. That is too few for
  bigger contrastive batches. Found 2026-10-21 in the batch-size sweep
  (`src/training/ENV_SIZING.md`): at 512 cards per batch a single-card
  batch wants 256 decks, against 159. Small splits now yield one short
  batch instead of none, so this no longer breaks runs, but every
  Pokemon batch is then the whole box, and its 11-deck TEST split is a
  thin measurement. Look for a source of played decklists
  (tournament or ladder lists). `plans/pre_training_data.md` lists
  Pokemon decks as "nothing more planned" and needs updating when this
  starts.
