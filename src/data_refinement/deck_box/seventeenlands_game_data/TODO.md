# TODO

## Real fix needed: unbounded in-memory accumulation for this source

**Found 2026-09-25**, while watching the third attempt at a full
`seventeenlands_game_data` deck box regeneration (the first two were
killed by the harness's background-shell memory-pressure reaper,
mid-run, with zero output - see `src/training/TODO.md` section E for
that incident).

**The problem:** `DeckBox` (`src/data_refinement/deck_box/deck_box.py`)
is, by design, a pure in-memory store - `self._decks_by_uuid:
dict[UUID, GenericDeck] = {}`, no incremental persistence at all.
`save()` writes the entire box in one shot; `build_or_update_deck_box()`
(`src/data_refinement/deck_box/build.py`) calls it exactly once, after
`extract()` has fully returned. This is completely fine for every other
deck source in this project - per `src/training/TODO.md`'s own
data-coverage table, FaB is 4.1k decks, Gwent 60k, STS2 7.8k, all
trivially small to hold in memory.

`seventeenlands_game_data` is a different scale entirely:
`data/raw/17lands/game_data` is **84GB across 133 CSV files**. This
stage mints one distinct deck per `(draft_id, match_number,
game_number)` triple - every game ever played gets its own entry, with
no content-based deduping across games (see this directory's own
`extraction_stage.py` docstring, "PER-GAME IDENTIFIER"). That's
plausibly tens of millions of `GenericDeck` objects held simultaneously
in one Python dict for the entire multi-hour run, before a single byte
reaches disk. This is almost certainly a major contributor to (maybe
the whole explanation for) the two prior OOM kills, and it means even a
*successful* run has no safety net - a crash at 99% loses every deck
extracted so far, not just the most recent file's worth.

**The real fix:** restructure this stage's extraction loop to flush and
*clear* each source CSV file's decks from memory before moving to the
next file, writing incrementally rather than rewriting the whole box
from a fully-accumulated in-memory state at the very end. Nothing about
this source's identity scheme requires holding every previously-seen
deck in memory to do this correctly: each game's composite key
(`draft_id:match_number:game_number`) is already globally unique across
the whole dataset by construction (17Lands' own convention, per the
docstring's "PER-GAME IDENTIFIER" section) - there is no cross-file
duplicate this stage would ever need to detect by keeping prior files'
decks resident. The "hold everything" behavior today is purely an
artifact of `DeckBox`'s simple, monolithic in-memory design, not a
functional requirement of this particular source. Same shape of fix,
same root cause pattern, as the `ParquetWriter`/row-group bug already
found and fixed elsewhere this session for oversized metric outputs
(unbounded resident memory across a huge streamed source) - see
`src/data_refinement/metrics/parquet_builder.py`.

A cheaper, partial mitigation (save after every source CSV file instead
of waiting for the whole directory) is NOT a fix for the memory
problem itself - `DeckBox` never evicts anything from
`_decks_by_uuid`, so peak memory still grows unbounded across the whole
run either way. It only bounds how much work a crash loses. Worth doing
regardless, but don't mistake it for solving the actual OOM risk.

## Also worth a wider discussion, separately from the above

The pure-Python-object-in-memory `DeckBox` design has served this
project well everywhere else, but `seventeenlands_game_data` is the
first source big enough to actually outgrow it. Before committing to a
specific fix here, it's worth a broader architecture conversation about
what technology/strategy this project wants for genuinely massive raw
sources going forward (this won't be the last one - 17lands' replay
data is ~139GB raw, also not yet ingested into a deck box or metrics,
per `src/training/TODO.md`'s data-coverage table): a real database
(e.g. a dockerized Postgres/MongoDB) as the deck/metric store instead
of in-memory-objects-plus-JSONL; leaning harder on `pandas`/`pyarrow`
chunking end-to-end; or some other pattern entirely. Deliberately not
decided here - flagged so the incremental fix above isn't accidentally
treated as the final answer for how this project handles its largest
data sources.
