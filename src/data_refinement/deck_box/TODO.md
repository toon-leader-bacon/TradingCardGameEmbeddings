# TODO

## New deck sources

Flat decks only (a `card_nocab_uuids` multiset), per `GenericDeck`'s
no-metadata design. Built today: `sts_gg/`, `sts2runs/`,
`fabtcg_decklists/`, `play_gwent/`, `seventeenlands_game_data/`.

- **`spire_codex` runs** - `src/data_retrieval/spire_codex/run_downloader.py`
  pulls spire-codex's paginated run export. Once downloaded, it would be
  a third Slay the Spire 2 run source alongside `sts_gg` and `sts2runs`;
  check for schema overlap with those two before building a
  near-identical stage.
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

- The play_gwent `unresolved card template id '24'` error no longer
  happens: the 2026-09-27 run logged no unresolved ids and put 0
  Unknown cards in 60,197 decks.
- STS2 cards missing from the spire-codex binder. Across the 7,800 STS2
  decks there are 251 Unknown slots out of 195,027 (0.13%), in 229
  decks. Ids by count (sts2runs + sts_gg): `CARD.FOLLOW_THROUGH`
  126+26, `CARD.GRAPPLE` 41+12, `CARD.PREPARE` 40+0,
  `CARD.ABUNDANCE` 0+4, `CARD.UNDERWORLD` 0+1, `CARD.SIDESTEP` 0+1.
- The sts_gg "0 decks" result does not happen on a fresh box: it created
  1,004 decks, one per line of `runs.jsonl`. The likely cause of the
  old result: the count only includes created or changed decks, and the
  earlier run went into a box that already held identical sts_gg decks.
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
- Confirm a full `seventeenlands_game_data` extraction (84 GB, 133 CSVs)
  now completes under the SQLite `DeckBox`; earlier in-memory attempts
  were OOM-killed. A one-CSV smoke test (`KTK.TradSealed.csv` into a
  scratch box) was clean: 2,558 decks, one per game row, 0 Unknown, 40
  cards median, version stamped. The full run started 2026-09-27 13:54
  (log: `data/tmp/ingest_17lands.err`). It uses 0.4-3.7 GB of RAM
  (per pandas chunk), writes about 75-80 decks/s and about 5.4 KB of `.db` per
  deck. Against an estimated ~28M game rows, that is about 4 days and a
  ~150 GB `mtg.db`. The run was stopped on purpose at 14:15 so two
  fixes could land first (2026-09-27): `DeckBox` now commits every
  1,000 writes instead of every write (one fsync per deck was the
  bottleneck), and the 17lands stage keeps one deck per `draft_id`
  (from its lowest match/game) instead of one per game, about 5x fewer
  decks. A one-CSV smoke test (`MSH.TradDraft.csv`: 41,361 games,
  6,663 drafts) gave exactly 6,663 decks at ~1,000 rows/s, now limited
  by pandas rather than commits. The partial per-game `mtg.db` was
  deleted, and the full re-run started 2026-09-27 (logs:
  `data/tmp/ingest_17lands.out`/`.err`); estimate ~8 hours and a
  ~27 GB box. At 15:24 it was stopped by low system memory (not a
  code failure), 38% into `Cube_-_Powered.PremierDraft.csv` (AFR, BLB,
  BRO done), leaving a 2.1 GB `mtg.db` with no version stamp. It was
  resumed at 17:41 and stopped by low memory again at 21:16, 15% into
  `MID.PremierDraft.csv` (file 68 of 133, ~51% of the 90 GB by bytes),
  leaving a 14.3 GB unstamped `mtg.db`. Unmatched names seen so far
  (Unknown sentinel, acceptable cruft): Sol'kanar the Tainted, Yera
  and Oski, Weaver and Guide, Nia, Skysail Storyteller, Makdee and
  Itla, Skysnarers, Luis, Pompous Pillager, Goben, Gene-Splice
  Savant, Ademi of the Silkchutes, plus Pick Your Poison (below).
  **To resume**, run it on its own, not beside another heavy job:
  `PYTHONPATH=. venv/Scripts/python.exe scripts/run_deck_box_ingestion.py --source seventeenlands_game_data`.
  Opening the box rolls back the unfinished batch, and deck ids are
  deterministic, so finished files are re-read without writes (about
  18 MB/s, so ~45 min for the 67 finished files) before new work
  starts. About 4-5 hours of new work remain.
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
