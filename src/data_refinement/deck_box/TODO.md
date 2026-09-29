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

- STS2 cards missing from the spire-codex binder. Across the 7,800 STS2
  decks there are 251 Unknown slots out of 195,027 (0.13%), in 229
  decks. Ids by count (sts2runs + sts_gg): `CARD.FOLLOW_THROUGH`
  126+26, `CARD.GRAPPLE` 41+12, `CARD.PREPARE` 40+0,
  `CARD.ABUNDANCE` 0+4, `CARD.UNDERWORLD` 0+1, `CARD.SIDESTEP` 0+1.
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
  Unknown sentinel, which is acceptable cruft, but most fall into a few
  fixable groups:
  - *`OM1` (Through the Omenpath), ~144 names* such as `Skittering
    Kitten` and `Rizna, the Spider-Crowned`: nearly the whole Arena set.
    None are in the oracle-cards dump
    (`oracle-cards-20260912210156.jsonl`), probably because Scryfall
    files them as alternate names of the Spider-Man (`SPM`) printings,
    which only the default-cards dump carries (e.g. as `flavor_name`).
    This is the biggest gap: OM1 decks are mostly Unknown. Check the
    default-cards dump and add those names as aliases.
  - *Back faces of modal double-faced cards (KHM), 13 names* such as
    `Mistgate Pathway`, `Tibalt, Cosmic Impostor`, `Kaldring, the
    Rimestaff`: the binder has the cards, but
    `card_lookup.uuid_for_name_or_front_face()` only matches the front
    face. Match back faces too.
  - *Alchemy rebalanced `A-` names (HBG), 2 names*: `A-Baba Lysaga,
    Night Witch`, `A-Monster Manual`. The binder has the paper cards;
    either strip the `A-` prefix to the original card or accept them
    as Unknown, since the rebalanced text differs.
  - *Arena Cube cards, 7 names* such as `Ademi of the Silkchutes`,
    `Yera and Oski, Weaver and Guide` (`Cube_-_Powered`): digital-only
    and not in the oracle dump. Probably accept.
  - *Corrupted in the source data*: `Bespoke B?` (TMT) is literally
    `B?` in the 17lands CSV header (most likely `Bespoke Bō`). A
    one-entry alias would fix it.
  - *Same-name collisions*: `Pick Your Poison` (above) and `Red
    Herring` (MKM, four binder entries with that name). Same decision
    as Pick Your Poison.
- **Read only the needed CSV columns.** `pd.read_csv` in
  `_extract_file` parses every column, including the
  `opening_hand_`/`drawn_`/`tutored_`/`sideboard_` card columns the
  stage never uses (about 4/5 of the file). Passing `usecols` (the key
  columns plus `deck_*`) should cut parse time and the 0.4-3.7 GB
  per-chunk memory several-fold, which matters given the low-memory
  kills. It would also remove the 13 `DtypeWarning`s (`opp_rank`,
  `splash_colors`) from the log.
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
