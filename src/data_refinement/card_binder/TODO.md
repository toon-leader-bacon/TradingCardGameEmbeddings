# TODO

## Store the binder as Arrow instead of JSONL (someday, not urgent)

`CardShelf` loads a game's whole binder into memory today. That fits
now (MTG is about 35k cards), but would not if we ingest custom cards
people post online, which could bring MTG to millions. An Arrow IPC
file, memory-mapped, would let training read only the cards it touches.

Motivation:
1. Learning Arrow beyond the training snapshots (`../snapshots/`).
2. Scale: an in-memory binder stops working at millions of cards.

Starting point: the card snapshot (`cards.arrow`, in
`../snapshots/`) is already a level-2 Arrow file of the
binder, one row per card, with the row number as the card index. An
Arrow binder would add a `raw_content` column to it and a `CardLookup`
implementation over it; deck snapshots keep their indices unchanged.

## Move the six tcgjson games to tcgcsv (high priority)

tcgcsv (`tcgcsv/`) mirrors the same TCGplayer catalog as tcgjson, but
daily; our tcgjson release is from 2026-08-31. Compared 2026-11-07 on
Lorcana, One Piece and Riftbound: tcgcsv has 5-24% more card products
(whole new sets: Hyperia City, Radiance), never fewer, and the same
game fields (tcgjson adds only releaseDate and detailNote, neither game
text). Where values differ tcgcsv is the newer fix (Riftbound names
"Darius, Trifarian", bracketed keywords "[ACCELERATE]").

Work for the switch:
1. Add the six category ids to `CATEGORY_IDS` in
   `src/data_retrieval/tcgcsv/downloader.py` (Digimon 63, One Piece 68, Lorcana 71, Star Wars Unlimited 79, Union Arena 81,
   Riftbound 89).
2. Rewrite each spec for tcgcsv's extendedData names ("Cost Ink", not
   "costInk"); One Piece rarity is a code ("SR"); list fields are one
   ";"-joined string ("Straw Hat Crew;Supernovas"), so LIST fields need
   splitting.
3. Bug: Riftbound's type key is "Card Type" (space), but
   `tcgcsv/catalog.py` reads only "CardType", so every Riftbound card
   would come in untyped.
4. Re-ingest the binders (re-ingest, don't migrate), then rename the
   pipeline source-neutral and retire `src/data_retrieval/tcgjson/`.

Yu-Gi-Oh is not part of this: it moved to YGOPRODeck's card table
(`ygoprodeck/`, 2026-11-07), since tcgcsv has no Level/Rank and the
decks name cards by passcode.

## Move hearthstonejson/ and tcgjson/ onto pool_mirror.py

`pool_mirror.py` holds the create-or-merge and delete-the-rest steps of
a stage that mirrors a whole card pool; `ygoprodeck/` uses it, but
`hearthstonejson/ingestion_stage.py` and `tcgjson/ingestion_stage.py`
still carry their own copies (`_store_group`'s create-or-merge branch,
`_delete_untaken_cards`). Replace both with the shared functions.

## Yu-Gi-Oh: YGOPRODeck's temporary pre-release ids

After the 2026-11-07 rebuild, 0.30% of Yu-Gi-Oh deck card copies (in
2.1% of decks) are Unknown. Most are 9-digit ids such as `101305096`,
which look like the ids YGOPRODeck gives a card before it has a passcode
(not checked); `cards.csv` would list such a card under its real
passcode since. A mapping from old
temporary ids to passcodes would recover them, if one can be found.
