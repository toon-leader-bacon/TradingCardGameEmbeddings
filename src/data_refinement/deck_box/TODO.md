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

- `PlayGwentDeckExtractionStage: unresolved card template id '24'` -
  substitutes the Unknown sentinel early in `guides.jsonl`, then
  appears to start resolving after ~3 GB. Not yet diagnosed.
- STS2 cards missing from the spire-codex binder: `CARD.FOLLOW_THROUGH`,
  `CARD.UNDERWORLD`, `CARD.GRAPPLE`, `CARD.ABUNDANCE`, `CARD.SIDESTEP`,
  `CARD.PREPARE`.
- The sts_gg deck extraction stage created 0 decks on its last run. Not
  yet diagnosed.
- **"Pick Your Poison" is two real MTG cards** (sets `cmb2` and `mkm`,
  both `layout: normal`), so every name-based lookup of it is ambiguous
  and 17lands decks get the Unknown sentinel for it. Needs a decision,
  not a bug fix: pick a canonical printing when a name has several real
  cards (in `card_binder/scryfall` or in name matching,
  `card_lookup.uuid_for_name_or_front_face()`), or accept the sentinel
  for true same-name collisions and document it.
- Confirm a full `seventeenlands_game_data` extraction (84 GB, 133 CSVs)
  now completes under the SQLite `DeckBox`; earlier in-memory attempts
  were OOM-killed.

## Enhancements

- **A read-only, `CardLookup`-shaped counterpart to `DeckBox`** for
  least-privilege readers such as `DeckBoxDealer`, which only calls
  `uuids_ranked_randomly()`/`get_by_uuid()` but takes a full `DeckBox`.
  A single-path `DeckBox.load()` writes straight to the on-disk file, so
  an accidental mutating call through a reader's reference is not
  caught by any `save()` step.
