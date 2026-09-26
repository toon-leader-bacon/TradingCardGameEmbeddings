# TODO

Candidate raw data sources for new `DeckExtractionStage` implementations,
beyond the three already built (`sts_gg/`, `fabtcg_decklists/`,
`play_gwent/` — see this directory's `README.md`). Flat decks only (a
`card_nocab_uuids` multiset) — no per-slot/run-outcome metadata, per
`GenericDeck`'s no-metadata design.

## Ready now (raw data already on disk, cards already ingested)

## Needs upstream work first

- **`spire_codex` runs** — `run_downloader.py`
  (`src/data_retrieval/spire_codex/run_downloader.py`) already exists
  and pulls spire-codex.com's bulk run export as cursor-paginated
  gzipped JSONL, but it hasn't actually been run yet — only
  `data/raw/spire_codex/cards.json` (the card dump) is on disk, no
  `data/raw/spire_codex/runs/` directory. Once downloaded, this would
  be a second Slay the Spire 2 run/deck source alongside `sts_gg` and
  `sts2runs` — worth checking for schema overlap/dedup potential with
  those two before building a third near-identical stage.

- **`pitchstack`** — `data/raw/pitchstack/decks.jsonl` (Flesh and Blood
  decks from pitchstack.gg) is already downloaded, but each row is
  deck *metadata* only (name, hero, format, `deckVersions` ids) — no
  card list. The `GET /v1/deck_versions/{deck_version_id}/cards`
  endpoint that would supply the actual cards is documented in
  `src/data_retrieval/pitchstack/downloader.py`'s module docstring but
  deliberately not implemented (see `src/data_retrieval/README.md`'s
  `pitchstack/` entry). A deck builder here needs that downloader
  extended first — it can't be built from what's on disk today.
  (`src/data_retrieval/pitchstack/pitchstack.md` currently claims the
  card-list endpoint is already implemented; that's stale relative to
  the downloader's actual code and the top-level README — trust the
  code.)

- **`17lands` game data** — no raw file downloaded yet (only
  `SeventeenLandsDownloader` exists, untested against real deck-shaped
  output in this project). 17Lands' `game_data` CSVs are wide-format:
  one row per game played, with a `deck_<CardName>` column per card in
  the set giving the copy count that player's deck ran. This is real
  deck data (the actual deck a player built and played that game), but
  extracting it means melting a wide row into a flat card list rather
  than walking a nested structure like the other candidates above —
  more parsing work, and higher volume (one row per *game*, not per
  deck, so many rows share the same deck). Lowest priority of the
  candidates here given the extra melting logic and no raw file to
  inspect yet.

## Not deck sources

`scryfall`, `hearthstonejson`, `gwent_one`, and `cardvault_fabtcg` each
pull card data only — no deck-shaped raw data exists for any of them
(gwent.one's card data is what `play_gwent`'s decks already resolve
against; HearthstoneJSON has no public deck dump). 17Lands' `draft_data`
CSVs are picks, not decks, and are also out of scope here.


## Bug fixes:

- Fix this: `FabtcgDecklistsExtractionStage: unresolved card name 'Sawbones, Dockhand' — substituting the Unknown sentinel card`
- Fix this: `FabtcgDecklistsExtractionStage: unresolved card name 'Smash With Big Tree' — substituting the Unknown sentinel card`

- PlayGwentDeckExtractionStage is broken `PlayGwentDeckExtractionStage: unresolved card template id '24' — substituting the Unknown sentinel card` for example. But, strangely, after ~3GB of processing the guides.jsonl it seems to start working? Weird. 

- STS cards missin:
  - CARD.FOLLOW_THROUGH
  - CARD.UNDERWORLD
  - CARD.GRAPPLE
  - CARD.ABUNDANCE
  - CARD.SIDESTEP
  - CARD.PREPARE
- Sts_GG deck builder created 0 decks?

- `SeventeenLandsGameDataDeckExtractionStage: unresolved card name 'Pick
  Your Poison' — substituting the Unknown sentinel card` (seen during the
  2026-09-24 regeneration) — but this one isn't a data gap like the
  fabtcg/play_gwent ones above. Scryfall itself has two distinct real
  cards sharing that exact name (`set: cmb2`, a
  counters/snake-token/wrath-effect sorcery, vs. `set: mkm`, a
  sacrifice-a-permanent-type sorcery — both `layout: normal`), so
  `CardBinder.get_by_name()` correctly returns 2 matches and
  `_card_uuid_for_name()` falls back to Unknown, same as it would for any
  genuine ambiguous name. The Scryfall layout-collision fix (see
  `card_binder/scryfall/ingestion_stage.py`'s `_EXCLUDED_LAYOUTS`, added
  2026-09-24 for the unrelated Tarmogoyf-token bug) doesn't touch this —
  both "Pick Your Poison" rows are legitimately `normal`-layout, real,
  playable cards. Needs an actual decision, not a bug fix: either add
  set/printing-aware disambiguation to `_card_uuid_for_name` (pick a
  canonical printing when a name legitimately has multiple distinct real
  cards), or accept the Unknown-sentinel fallback for true same-name
  collisions as expected behavior and document it as such. This cuts
  across `card_binder/scryfall` (whether/how to pick a canonical printing
  at ingestion time) and this stage's name-resolution policy — not purely
  a `deck_box` fix, so the eventual owner may live in either place. 

## Future enhancement (carried over from plans/deckbox_sqlite.md, now implemented)

- A read-only, `CardLookup`-shaped counterpart to `DeckBox`, for
  `DeckBoxDealer`'s (and any future consumer's) least-privilege access —
  it takes a full `DeckBox` today though it only ever reads through it
  (`uuids_ranked_randomly()`/`get_by_uuid()`). Worth doing precisely
  because `DeckBox`'s SQLite rewrite made single-path `load()`
  durable-by-default: a caller accidentally calling a mutating method
  through a `DeckBox` reference that was only ever meant to be read from
  now writes straight to the canonical on-disk file, with no `save()`
  step that would have caught the mistake under the old in-memory
  design. Not a regression from that rewrite, but its priority went up
  because of it.