# CURRENT_STATUS

A snapshot (2026-09-27) of which `data_retrieval` sources have
matching `data_refinement` support, used as a soft TODO list. Update
a row when its status changes. Hard bugs and chores live in each
container's own `TODO.md`; this file tracks coverage only.

Legend: ✅ exists · ❌ missing · 🟥 brainstorm only · 🟨 partial ·
🟩 complete or mature.

## Per-source matrix


| Source (retrieval)               | Game             | Card binder stage                                            | Complete decks in source?                                                                                                                   | Deck box stage                                                                                                | Metrics                                                                                                                                                                                         |
| -------------------------------- | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scryfall`                       | MTG              | ✅ `ScryfallCardIngestionStage`                               | No (card data only)                                                                                                                         | n/a                                                                                                           | 🟥 Brainstorm only (30 ideas)                                                                                                                                                                   |
| `seventeenlands` › `draft_data`  | MTG              | (uses scryfall)                                              | No (picks, not decks)                                                                                                                       | n/a                                                                                                           | 🟨 6 of ~20 built, all with dojos                                                                                                                                                               |
| `seventeenlands` › `game_data`   | MTG              | (uses scryfall)                                              | Yes (40-card limited decks, `deck_*` columns)                                                                                               | ✅ `seventeenlands_game_data`                                                                                  | 🟨 11 of ~19 built, all with dojos                                                                                                                                                              |
| `seventeenlands` › `replay_data` | MTG              | (uses scryfall)                                              | Yes, but the same games as `game_data`, so redundant                                                                                        | n/a                                                                                                           | 🟨 9 of ~22 built, all with dojos                                                                                                                                                               |
| `pokemon_tcg`                    | Pokemon          | ✅ `PokemonTcgCardIngestionStage`                             | Yes: 83 set files of official 60-card theme decks in `data/raw/pokemon_tcg/decks/`                                                          | ❌ Missing                                                                                                     | 🟥 Brainstorm only (30)                                                                                                                                                                         |
| `hearthstonejson`                | Hearthstone      | ❌ Missing: 84 build snapshots downloaded, no ingestion stage | No (card data only)                                                                                                                         | n/a                                                                                                           | 🟥 Brainstorm only (30)                                                                                                                                                                         |
| `gwent_one`                      | Gwent            | ✅ `GwentOneCardIngestionStage`                               | No                                                                                                                                          | n/a                                                                                                           | 🟨 8 masking metrics of ~30 ideas                                                                                                                                                               |
| `play_gwent`                     | Gwent            | (uses gwent_one)                                             | Yes (community deck guides)                                                                                                                 | ✅ `play_gwent`                                                                                                | 🟨 1 of ~30 (`LeaderMaskedFromDeckMetric`)                                                                                                                                                      |
| `spire_codex` (cards)            | Slay the Spire 2 | ✅ `SpireCodexCardIngestionStage`                             | No (`cards.json`)                                                                                                                           | n/a                                                                                                           | 🟥 Brainstorm only (30)                                                                                                                                                                         |
| `spire_codex` (runs)             | Slay the Spire 2 | (uses spire_codex)                                           | Yes: `players[].deck` holds the final deck; the run record looks like the same schema as sts2runs                                           | ❌ Missing (see `deck_box/TODO.md`)                                                                            | ❌ Nothing, not even a brainstorm (`metrics/spire_codex/BRAINSTORM.md` covers cards only)                                                                                                        |
| `sts2runs`                       | Slay the Spire 2 | (uses spire_codex)                                           | Yes                                                                                                                                         | ✅ `sts2runs`                                                                                                  | 🟥 Brainstorm only (30)                                                                                                                                                                         |
| `sts_gg`                         | Slay the Spire 2 | (uses spire_codex)                                           | Yes                                                                                                                                         | ✅ `sts_gg` (1,004 decks in the 2026-09-27 SQLite run; the old "0 decks" result doesn't happen on a fresh box) | 🟩 24 built, all with dojos (2026-10-01). Wins only (leaderboard source), so the win/killed-by labels are constant and left out of the catalog; `sts2_runs` has them. No `BRAINSTORM.md`, though its README points to one |
| `cardvault_fabtcg`               | Flesh and Blood  | ✅ `CardVaultFabtcgCardIngestionStage`                        | No                                                                                                                                          | n/a                                                                                                           | 🟥 Brainstorm only (30)                                                                                                                                                                         |
| `fabtcg_decklists`               | Flesh and Blood  | (uses cardvault_fabtcg)                                      | Yes (tournament decklists)                                                                                                                  | ✅ `fabtcg_decklists`                                                                                          | 🟥 Brainstorm only (30)                                                                                                                                                                         |
| `pitchstack`                     | Flesh and Blood  | (uses cardvault_fabtcg)                                      | Not yet: `decks.jsonl` holds metadata only; card lists need the unimplemented `/cards` endpoint (`data_retrieval/pitchstack/pitchstack.md`) | ❌ Blocked on retrieval                                                                                        | 🟥 Brainstorm only (30, most blocked on card lists)                                                                                                                                             |
| `dominiontabs`                   | Dominion         | ✅ `DominionTabsCardIngestionStage`                           | No                                                                                                                                          | n/a                                                                                                           | 🟩 All 3 viable ideas built, with dojos, and run                                                                                                                                                |
| `isotropic`                      | Dominion         | (uses dominiontabs)                                          | Yes: summary archives hold each player's `end.deck`; `metrics/isotropic/summary/row_utils.deck_for_player()` already builds a `GenericDeck` | ❌ Missing                                                                                                     | 🟨 24 built (13 summary, 11 games), no dojos; registered in `run_metrics.py` and run (2026-09-27); resignation metrics unbuilt; partial-deck gap in `metrics/isotropic/games/TODO.md`           |
| `dominion/`                      | Dominion         | Not a downloader (research notes and a prototype scraper)    | n/a                                                                                                                                         | n/a                                                                                                           | n/a                                                                                                                                                                                             |
| none                             | Yu-Gi-Oh         | ❌ `GameId.YUGIOH` exists but no retrieval source             | n/a                                                                                                                                         | n/a                                                                                                           | n/a                                                                                                                                                                                             |




## Priority order



### 1. A card binder for every game

- **Hearthstone**: the only game with raw data but no binder. Needs a
decision on which build to ingest, or how to merge across builds.
- **Yu-Gi-Oh**: needs a retrieval source first (`new_data_source`
skill), or drop `GameId.YUGIOH` until one exists.



### 2. Sources with no deck ingestion or no metrics at all

Deck ingestion:

- `pokemon_tcg` theme decks (raw data present).
- `spire_codex` runs (raw data present). Check the schema against
`Sts2RunsDeckExtractionStage` first; likely a reuse or subclass.
- `isotropic` final decks (raw data must be re-downloaded first).

First metrics, deck-bearing sources:

- `sts2runs`
- `fabtcg_decklists`
- `pokemon_tcg`
- `spire_codex` runs (needs a brainstorm first)

First metrics, card-only sources:

- `scryfall`
- `cardvault_fabtcg`
- `spire_codex` cards
- `hearthstonejson` (after its binder exists)

Blocked: `pitchstack`, until retrieval adds the `/cards` endpoint.

### 3. More metrics for sources that already have some

- `play_gwent`: 1 of ~30 built.
- `seventeenlands`: all three families.
- `gwent_one`: the multi-card and multi-group ideas.
- `isotropic`: dojos for the 24 existing metrics are arguably worth
more than new metrics.



### Cheap wins, any time

- Finish the deck box re-ingestion into SQLite `.db` files. Done
2026-09-27 for fabtcg_decklists, play_gwent, sts2runs and sts_gg.
`seventeenlands_game_data` (`mtg.db`, now batched and one deck per
draft) finished 2026-09-28: 4.8M decks, 0.71% Unknown slots; see
`deck_box/TODO.md`, which also groups the unmatched card names. Run heavy
jobs one at a time: two in parallel ran the machine out of memory.

