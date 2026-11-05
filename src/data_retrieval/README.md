# data_retrieval

Collects raw card/deck data from external sources and writes it to
`data/raw/<source>/` untouched. No parsing, normalization or schema
conversion happens here; that is `data_refinement`'s job. Nothing this
container produces is committed to git. Each source gets its own
subdirectory, so a broken or rate-limited source doesn't block the
others.

## Files

Shared by every source:

- `downloader.py`: `Downloader`, the abstract base class every source's
  downloader subclasses (except 17lands, below).
- `download_utils.py`: stream-to-disk GETs with retries
  (`download_to_file`, `download_to_string`, `call_with_retries`) and
  `paced_get` (a paced GET that returns tolerated statuses such as 404
  and waits out 429s by Retry-After), and manifest-tracked appends for
  resumable crawls (`read_manifest`, `append_with_manifest`).
- `rate_limiter.py`: `RateLimiter`, request pacing.

One subdirectory per source:

| Directory | Game | What it downloads |
|---|---|---|
| `archidekt/` | MTG | Archidekt decks via its open read API, one id at a time: the sitemap's ~17.5k decks, then every id up to the newest in a fixed spread order (resumable cursor, sessions; pruned rows, `unlisted` kept as a flag) |
| `cardvault_fabtcg/` | Flesh and Blood | cardvault.fabtcg.com's full card CSV |
| `dominiontabs/` | Dominion | dominiontabs' card database (fields + English text) |
| `fabtcg_decklists/` | Flesh and Blood | fabtcg.com decklist pages |
| `fabtcgmeta_online_decks/` | Flesh and Blood | fabtcgmeta.com's Talishar online decklists: 8,192 static JSON shards, ~800k decks with card lists (source survey: `fab_deck_sources.md`) |
| `gwent_one/` | Gwent | gwent.one card search results (HTML fragments) |
| `hearthstonejson/` | Hearthstone | one `cards.json` per Hearthstone build |
| `hearthstonetopdecks/` | Hearthstone | hearthstonetopdecks.com deck page URLs (the deck listing pages) and each page's title, deck code and details block |
| `zenodo_hearthstone_decks/` | Hearthstone | Zenodo record 10198504's `decks.csv` and `cards.csv` (852k decks, 2013-2020, CC BY 4.0), MD5-checked |
| `isotropic/` | Dominion | Wayback-archived isotropic.org game logs |
| `mtgjson_decks/` | MTG | MTGJSON's AllDeckFiles: every Wizards-printed precon as one JSON file, one directory per daily build, SHA-256-checked (source survey: `plans/mtg_deck_sources.md`) |
| `pitchstack/` | Flesh and Blood | pitchstack.gg deck ids and card lists (`pitchstack.md`: unimplemented endpoints) |
| `limitless_play/` | Pokemon | play.limitlesstcg.com API: PTCG tournament list and each tournament's standings with decklists (paced at 9/min, the server's keyless limit) |
| `play_gwent/` | Gwent | playgwent.com deck guides |
| `pokemon_tcg/` | Pokemon | the pokemon-tcg-data repo's card and deck JSON, and `sets/en.json` (set codes) |
| `scryfall/` | MTG | Scryfall oracle-cards and default-cards (every printing) bulk files |
| `seventeenlands/` | MTG | 17lands per-set/per-format draft, game and replay CSVs (own [README](seventeenlands/README.md)) |
| `spire_codex/` | Slay the Spire 2 | spire-codex `cards.json` and its paginated run export |
| `sts_gg/` | Slay the Spire 2 | sts.gg run ids and run detail JSON |
| `dominion/` | Dominion | not a downloader: research leads (`todo.md`) and a prototype replay scraper |

## How it works

A `Downloader` fetches in two phases. `phase_1()` does a source's first
fetch: a single static file, a self-contained paginated walk, or, for an
id-list-then-detail source, just the id list. `phase_2()` does fetching
that depends on `phase_1()`'s output and defaults to returning
`phase_1()`'s result for a source with nothing further to fetch (see
`Downloader`'s docstring for the full contract). Every constructor
accepts an optional `rate_limiter` (default
`RateLimiter(requests_per_minute=60)`) and an optional `raw_data_dir`
(default: the class's `DEFAULT_RAW_DATA_DIR`), plus any source-specific
arguments.

## How to run

```
PYTHONPATH=. python3 scripts/run_data_retrieval.py --list
PYTHONPATH=. python3 scripts/run_data_retrieval.py --source scryfall
```

Or from Python:

```python
from src.data_retrieval.scryfall.downloader import ScryfallOracleDownloader

downloader = ScryfallOracleDownloader(oracle_cards_url)
downloader.phase_1()
raw_path = downloader.phase_2()  # data/raw/scryfall/...
```
