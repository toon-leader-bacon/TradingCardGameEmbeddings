# data_retrieval

Collects raw card/deck data from external sources and writes it to
disk untouched — no parsing, normalization, or schema conversion
happens here, that's `data_refinement`'s job. Nothing this container
produces is committed to git; it only ever writes into `data/raw`.
Each source gets its own subdirectory so a broken or rate-limited
source doesn't block the others.

Internally, each source subdirectory is expected to be a fairly
independent scraper/downloader with its own retry and auth concerns
specific to that source — none of that is shared logic, so it stays
local to the source rather than pushed up into this file. Two things
live at this shared level instead: `rate_limiter.py` (deliberately
placed here from the start, ahead of a second consumer — see its own
module docstring) and `download_to_file.py` (extracted after the same
stream-to-disk block showed up independently in three downloaders —
the "rule of three" case, not a preemptive one). What else is shared
across sources, once something actually needs it, belongs at this
level too rather than duplicated per source.

Implemented today:

- `scryfall/` — pulls from Scryfall's structured card-dump API.
- `seventeenlands/` — pulls 17Lands' public per-set/per-format CSV
  dumps. See [`seventeenlands/README.md`](seventeenlands/README.md).
- `pokemon_tcg/` — pulls card and deck JSON from the community-maintained
  pokemon-tcg-data GitHub repo (V1; may be replaced by an official API
  source later, as a separate V2 rather than a rewrite of this one).
- `hearthstonejson/` — pulls per-build card JSON dumps from
  HearthstoneJSON (api.hearthstonejson.com).
- `gwent_one/` — pulls raw HTML page fragments from gwent.one's card
  search AJAX endpoint (no bulk dump exists for Gwent).
- `play_gwent/` — pulls deck guides from playgwent.com, in two phases:
  `phase_1()` pages through the site's guides-list API to collect
  every guide id, `phase_2()` fetches each guide's HTML detail page
  and extracts the deck payload embedded in it (a `data-state` HTML
  attribute holding HTML-escaped JSON — no separate API endpoint
  exists for deck details). See
  [`play_gwent/TODO.md`](play_gwent/TODO.md) for a known,
  deliberately-deferred naming inconsistency with the other sources
  in this container.
- `rate_limiter.py` — shared politeness pacer (`RateLimiter`).
- `download_to_file.py` — shared streamed-GET-to-disk helper
  (`download_to_file`), used by any source's plain single-file
  downloads.

Planned, not yet built — no subdirectory exists for either yet:

- **Reddit** — pulling data from Reddit archive dumps/comments.
- **General-purpose website scraping** — for sources without a
  dedicated API.

## How to run

Run these from the project root, so the `src` package resolves. Each
writes into `data/raw/<source>/`.

**Scryfall** (`oracle-cards` filename changes with each dump — get the
current one from https://scryfall.com/docs/api/bulk-data):

```bash
python3 -c "
from pathlib import Path
from src.data_retrieval.scryfall.downloader import ScryfallOracleDownloader

downloader = ScryfallOracleDownloader(
    'https://data.scryfall.io/oracle-cards/oracle-cards-20260820090157.jsonl.gz',
)
print(downloader.fetch())
"
```

**Pokemon TCG:**

```bash
python3 -c "
from pathlib import Path
from src.data_retrieval.pokemon_tcg.downloader import PokemonTcgDataDownloader

downloader = PokemonTcgDataDownloader(
    'https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/zipball',
)
print(downloader.fetch())
"
```

**HearthstoneJSON** (downloads ~80+ files — the `RateLimiter` paces
requests, `tqdm` shows progress):

```bash
python3 -c "
from pathlib import Path
from src.data_retrieval.hearthstonejson.downloader import HearthstoneJsonDownloader
from src.data_retrieval.rate_limiter import RateLimiter

downloader = HearthstoneJsonDownloader(
    'https://api.hearthstonejson.com/v1/',
    rate_limiter=RateLimiter(requests_per_minute=12),
)
print(downloader.fetch())
"
```

**Gwent (gwent.one)** (a single POST to the search AJAX endpoint,
capped above the live card count, returns every card in one page —
see `gwent_one/downloader.py`'s module docstring for how that
endpoint was reverse-engineered):

```bash
python3 -c "
from src.data_retrieval.gwent_one.downloader import GwentOneDownloader
from src.data_retrieval.rate_limiter import RateLimiter

downloader = GwentOneDownloader(
    rate_limiter=RateLimiter(requests_per_minute=12),
)
print(downloader.fetch(result_limit=1300))
"
```

**Play Gwent** (playgwent.com deck guides — two phases, run in order;
`phase_2()` skips any guide id already recorded in
`guides_manifest.txt` from a prior run):

```bash
python3 -c "
from src.data_retrieval.play_gwent.downloader import PlayGwentDownloader
from src.data_retrieval.rate_limiter import RateLimiter

downloader = PlayGwentDownloader(RateLimiter(requests_per_minute=12))
downloader.phase_1()
print(downloader.phase_2())
"
```

This file grows as each source directory above is actually built out.
