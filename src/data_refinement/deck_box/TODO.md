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
