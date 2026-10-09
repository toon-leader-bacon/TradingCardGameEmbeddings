# TODO

## New deck sources

Flat decks only (a `card_nocab_uuids` multiset), per `GenericDeck`'s
no-metadata design. Built today: `sts_gg/`, `spire_codex_runs/`,
`fabtcg_decklists/`, `play_gwent/`, `seventeenlands_game_data/`.

- **`spire_codex` runs** - `src/data_retrieval/spire_codex/run_downloader.py`
  pulls spire-codex's paginated run export. Once downloaded, it would be
  a Slay the Spire 2 run source alongside `sts_gg`; built as
  `spire_codex_runs/`.
- **`pitchstack`** - `decks.jsonl` holds deck metadata only, no card
  list. The downloader needs the per-version `/cards` endpoint first
  (see `src/data_retrieval/pitchstack/pitchstack.md`).

Not deck sources: `scryfall`, `hearthstonejson`, `gwent_one`,
`cardvault_fabtcg` (card data only), and 17lands `draft_data` (picks,
not decks).

## Data bugs seen during extraction

- **SQLite re-ingestion (2026-09-27).** `fabtcg_decklists`,
  `play_gwent`, `sts2runs` and `sts_gg` were re-ingested into fresh
  `.db` boxes; all four stamp `card_binder_version`, and deck counts
  match the pre-SQLite JSONL exactly (FaB 4,161; Gwent 60,197; STS2
  6,796 sts2runs + 1,004 sts_gg). The old `.jsonl` files were moved to
  `data/final/decks/_stale_jsonl_backup/`; delete that folder once the
  `.db` boxes have been used for a while. `seventeenlands_game_data`
  (`mtg.db`) was still running; see below.

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
- **Full `seventeenlands_game_data` run: done (2026-09-28).** Batched
  commits and one deck per `draft_id` took it from ~75 decks/s (~4
  days) to ~1,000 rows/s. Claude Code's background shell was killed by
  low memory twice; a foreground run in a separate terminal then got
  through VOW, and most of WOE judging by the box, before an accidental
  Ctrl+C (log: `logs/deck_box_17_lands_sep28.log`, which ends at VOW;
  no tracebacks). The four WOE files were then re-run on their own via
  `--raw-path` (45,310 more decks, 17 min). Final `mtg.db`: 27.9 GB,
  `card_binder_version` stamped, 4,812,218 decks, exactly the number of
  distinct `draft_id`s across all 133 CSVs. Unknown sentinel: 1,375,781
  of 193,020,442 card slots (0.71%), in 140,425 decks (2.9%), about 10
  per affected deck, which fits the OM1 gap below.
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

- **A read-only, `CardLookup`-shaped counterpart to `DeckBox`** for
  least-privilege readers such as `DeckBoxDealer`, which only calls
  `uuids_ranked_randomly()`/`get_by_uuid()` but takes a full `DeckBox`.
  A single-path `DeckBox.load()` writes straight to the on-disk file, so
  an accidental mutating call through a reader's reference is not
  caught by any `save()` step.

- **Abandoned StS2 runs.** Every StS2 deck source keeps them today
  (sts_gg, spire_codex_runs; the last has a
  `keep_abandoned` toggle, default on). The project favors quantity, but a
  quit run's deck is not a finished one and early quits are mostly
  starter decks. If that ever shows up as noise, filter only runs
  abandoned before the first boss (map_point_history / acts give the
  floor reached) rather than all of them.
- **Contrastive split indexes persist across training runs.**
  `DeckBoxDealer` writes `data/splits/contrastive/<dojo>.db` on a dojo's
  first build and reuses it afterwards, so decks added by a later
  re-ingest are never assigned a split. After re-ingesting a game's deck
  box, delete that game's index (or pass `force_resplit`).
- **A game's first deck ingestion invalidates its metrics.**
  `run_deck_box_ingestion.py` seeds the game's Unknown sentinel card
  (`ensure_unknown_card`) and saves the binder. The first time that happens
  for a game, the binder version changes, and every metric built from the
  old version then fails the dojos' strict version check. This hit Dominion
  on 2026-09-30, and the metrics had to be re-run.

  Fixed 2026-10-02: `build_or_update_card_binder()` now seeds the sentinel,
  so the binder version is final before any metric or deck box is built.
  The binders on disk already carry it for every deck-box game;
  Hearthstone gets it on its next ingestion (a one-time version change,
  which invalidates its 8 mask metrics once).
