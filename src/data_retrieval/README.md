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
local to the source rather than pushed up into this file. Three things
live at this shared level instead: `rate_limiter.py` (deliberately
placed here from the start, ahead of a second consumer — see its own
module docstring), `download_utils.py` (extracted after the same
stream-to-disk block showed up independently in three downloaders —
the "rule of three" case, not a preemptive one), and `downloader.py`
(the `Downloader` abstract base class every source's downloader class
subclasses — see plans/downloader_base_class.md for the design
discussion). What else is shared across sources, once something
actually needs it, belongs at this level too rather than duplicated
per source.

Every downloader class below except `seventeenlands/` subclasses
`Downloader` (`downloader.py`) and follows its `phase_1()`/`phase_2()`
convention — `SeventeenLandsDownloader`'s primary method takes a
required `refs` list plus filters, which doesn't fit the no-arg
`phase_1()` contract, so it deliberately keeps its own shape rather
than being forced into this one. Every other source's convention: `phase_1()` always
does a source's first fetch (a single static file, a fully
self-contained paginated walk, or — for an id-list-then-detail source
— just the id-list collection step); `phase_2()` does further fetching
that depends on `phase_1()`'s own output, and defaults to a no-op
(returning `phase_1()`'s own result unchanged) for a source with
nothing further to fetch — see `Downloader`'s own docstring for the
full contract. Every downloader's constructor accepts, at minimum, an
optional `rate_limiter` (defaulting to `RateLimiter(requests_per_minute=60)`)
and an optional `raw_data_dir` (defaulting to that class's own
`DEFAULT_RAW_DATA_DIR`), and may declare further source-specific
constructor arguments beyond those two.

Each writes into `data/raw/<source>/`.
