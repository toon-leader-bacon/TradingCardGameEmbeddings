# `mtg_crawls/`: MTG decks from deck-site crawls

Three deck sites' crawls go into the MTG deck box (`data/final/decks/mtg.db`),
next to the mtgjson and 17lands decks. Each site has its own package:

| Site | Package | `--source` | Raw dir (`src/data_retrieval/<site>/`) | Card ids it gives |
|---|---|---|---|---|
| Archidekt | `archidekt_decks/` | `archidekt_decks` | `data/raw/archidekt/` | Scryfall oracle id |
| Moxfield | `moxfield_decks/` | `moxfield_decks` | `data/raw/moxfield/` | Scryfall printing id |
| Deckstats | `deckstats_decks/` | `deckstats_decks` | `data/raw/deckstats/` (public decks; `private/` is not read) | Scryfall printing id (absent on 0.2% of entries) |

**Run these only after the MTG binder carries `SCRYFALL_PRINTING` aliases**
(the Scryfall card stage registers them; a binder built before 2026-11-08 has
none). A card that misses those aliases may fall to the Unknown card. That
changes the deck's multiset hash, so decks stored early sit under ids a later
run won't reproduce.

## Files

- `crawled_deck.py`: the typed records every site parser emits.
  - `MtgCardRef(name, oracle_id, printing_id)` names a card.
  - `MtgDeckEntry(count, card)` is one line of a decklist.
  - `CrawledDeck(source_id, name, entries)` is a whole deck.
  - `DeckSkip(reason)` is a parser declining a deck; `DeckSkipReason` is the
    one list of reasons.
- `mtg_card_matcher.py`: `MtgCardMatcher` matches a card in this order:
  1. the oracle id as a `SCRYFALL` alias;
  2. the printing id as a `SCRYFALL_PRINTING` alias;
  3. the name, through `uuid_for_name_or_front_face`, which also accepts
     either face of a two-faced card and never guesses;
  4. otherwise Unknown.

  It extends `CachedCardMatcher` (`../card_matching.py`), so each distinct
  card is looked up once and every match is counted.
- `extraction_stage.py`: `CrawledMtgDeckExtractionStage` is the loop the three
  sites share:
  - it reads the shards with `read_shard_rows`
    (`src/data_retrieval/resumable_crawl/crawl_files.py`); run it with the
    site's crawl stopped;
  - it hands each row to the site's parser;
  - it skips decks outside 20-256 cards;
  - it matches the cards and stores the deck with `store_multiset_deck`;
  - it logs outcomes, skip reasons, match counts and the most common
    unmatched card names.

  A site's stage subclasses it, setting `DATA_SOURCE`, `DEFAULT_RAW_PATH` and
  the parser.

Each site's `deck_parser.py` is the only code that knows that site's schema.

## What counts as the deck

Cards that are in play from the start of a game: the main deck plus the
command zone (commanders, partners, companions, Oathbreaker signature spells).
The parsers decide by what a section does, not what it is called:

- **Archidekt:** every card row except one whose primary category the deck
  marks `includedInDeck: false` (Maybeboard, usually Sideboard). Rows flagged
  `companion` always count, and rows with a `deletedAt` never do.
- **Moxfield:** boards `mainboard`, `commanders`, `companions`,
  `signatureSpells`.
- **Deckstats:** zones `main` and `commander`, plus entries flagged
  `is_commander` or `is_companion` (Deckstats files companions under
  `sideboard`). Token cards never count.

Dropped everywhere: between-games sideboards, maybeboards, tokens,
attractions, stickers, contraptions, schemes, planes.

## Which decks

- **Archidekt unlisted decks** are skipped unless the stage is built with
  `ArchidektDeckExtractionStage(include_unlisted=True)`. Their authors chose
  not to list them, so they stay out of published training data.
- **Moxfield `package` lists** are card bundles, not decks, so they are
  skipped.
- **Every other format** is kept, Casual and formatless decks included.
- **Size:** decks under 20 or over 256 cards are skipped.
- **Rows with no copies** (a quantity below 1, seen once in 1.7M Moxfield rows)
  are dropped from their deck rather than failing the run.

## Identity

A deck's uuid is its copy-count multiset hash, so a list posted twice, or on
two sites, is stored once, under the first row's name (its site id when
unnamed) and provenance (`DataSource.ARCHIDEKT` / `MOXFIELD` / `DECKSTATS` plus
the site's deck id).
