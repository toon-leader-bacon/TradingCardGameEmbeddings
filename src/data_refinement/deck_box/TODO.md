# TODO

## New deck sources

Flat decks only (a `card_nocab_uuids` multiset), per `GenericDeck`'s
no-metadata design. Built today: `sts_gg/`, `spire_codex_runs/`,
`fabtcg_decklists/`, `play_gwent/`, `seventeenlands_game_data/`.

- **`pitchstack`** - `decks.jsonl` holds deck metadata only, no card
  list. The downloader needs the per-version `/cards` endpoint first
  (see `src/data_retrieval/pitchstack/pitchstack.md`).

Not deck sources: `scryfall`, `hearthstonejson`, `gwent_one`,
`cardvault_fabtcg` (card data only), and 17lands `draft_data` (picks,
not decks).

## Data bugs seen during extraction

- **Delete `data/final/decks/_stale_jsonl_backup/`** once the SQLite
  `.db` boxes have been in use for a while. It holds the pre-SQLite
  `.jsonl` boxes, moved there at the 2026-09-27 re-ingestion.

- STS2 cards missing from the spire-codex binder. Across the 7,800 STS2
  decks there are 251 Unknown slots out of 195,027 (0.13%), in 229
  decks. Ids by count (sts2runs + sts_gg): `CARD.FOLLOW_THROUGH`
  126+26, `CARD.GRAPPLE` 41+12, `CARD.PREPARE` 40+0,
  `CARD.ABUNDANCE` 0+4, `CARD.UNDERWORLD` 0+1, `CARD.SIDESTEP` 0+1.
  The 2026-10-08 spire_codex_runs ingest (40 pages, 470,687 decks) logged
  459,509 Unknown slots over 9,613 distinct ids. 6,670 of the ids are
  mod cards (`CARD.<MOD>-<NAME>`, mostly seen once). The bulk are vanilla
  cards newer than `data/raw/spire_codex/cards.json`: `FOLLOW_THROUGH`
  32k, `ABUNDANCE` 24k, `SIDESTEP` 23k, `BLADE_SYMPHONY` 22k, `CONCOCT`
  22k, `HIBERNATE` 22k, `MIDNIGHT` 21k, and more. None of these is in
  `cards.json` (downloaded 2026-10-02). Possible fix, unverified: a fresh
  spire_codex card list (if it has them), then re-ingest the StS2 binder
  and the runs.
  Log: `logs/regen_2026-10-08/06_deck_box_spire_codex_runs.log`.
- Deck-size outliers, probably faithful to the source data: 5 FaB
  decklists hold 1 card (e.g. `soh-zheng-chane-deck-the-imaginarium-skirmish-270621`),
  and 90 hold fewer than 40. sts2runs run 5330 player 0 has 0 cards,
  and run 6781 player 0 has 762.
- **"Pick Your Poison" is two real MTG cards** (sets `cmb2` and `mkm`,
  both `layout: normal`), so every name-based lookup of it is ambiguous
  and 17lands decks get the Unknown sentinel for it. Needs a decision,
  not a bug fix: pick a canonical printing when a name has several real
  cards (in `card_binder/scryfall` or in name matching,
  `card_lookup.uuid_for_name_or_front_face()`), or accept the sentinel
  for true same-name collisions and document it.
- **17lands card names the MTG binder can't match** (from the full-run
  log; 162 distinct names, each logged once per file). All become the
  Unknown sentinel. The OM1 names (Arena printed names), KHM back faces
  and `Bespoke B?` are now matched by aliases (takes effect on the next
  MTG binder rebuild). What remains is accepted cruft: `A-` Alchemy
  rebalanced names (HBG, 2), Arena Cube digital-only cards (7), and the
  same-name collisions `Pick Your Poison` and `Red Herring`.
- Possible speedup, not needed yet: for the ~80% of 17lands rows that
  are later games of a stored draft, `_extract_row` calls
  `get_by_uuid()`, which builds the whole card list only to read the
  provenance. A row-only lookup would skip that.

## Enhancements

- **Abandoned StS2 runs.** Every StS2 deck source keeps them today
  (sts_gg, spire_codex_runs; the last has a
  `keep_abandoned` toggle, default on). The project favors quantity, but a
  quit run's deck is not a finished one and early quits are mostly
  starter decks. If that ever shows up as noise, filter only runs
  abandoned before the first boss (map_point_history / acts give the
  floor reached) rather than all of them.
- **Hearthstone's next card ingestion invalidates its 8 mask metrics
  once.** `build_or_update_card_binder()` seeds each game's Unknown
  sentinel (since 2026-10-02), and Hearthstone's binder does not carry it
  yet, so its version changes one more time. Re-run its metrics after.
