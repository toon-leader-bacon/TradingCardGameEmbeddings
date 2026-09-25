# TODO

## Bug fixes

- **Stale Scryfall dump is missing real cards referenced by 17lands game
  data.** During the 2026-09-24 deck box regeneration,
  `SeventeenLandsGameDataDeckExtractionStage` logged unresolved names for
  cards that are real, valid MTG cards, not junk — they're just absent
  from the current local Scryfall snapshot
  (`data/raw/scryfall/oracle-cards-20260912210156.jsonl`, dated
  2026-09-12):
  - `Ademi of the Silkchutes`
  - `Goben, Gene-Splice Savant`
  - `Luis, Pompous Pillager`
  - `Makdee and Itla, Skysnarers`
  - `Nia, Skysail Storyteller`
  - `Yera and Oski, Weaver and Guide`

  Confirmed via `grep` against `data/final/cards/mtg.jsonl`: zero matches
  for any of these names. So this isn't a resolution bug in the
  extraction stage — these cards never made it into the card binder
  because they aren't in the raw Scryfall dump at all, presumably
  because they belong to a set released after 2026-09-12. Fix: re-download
  a fresher Scryfall bulk-data dump (`src/data_retrieval/scryfall`) and
  re-ingest the card binder from it.
