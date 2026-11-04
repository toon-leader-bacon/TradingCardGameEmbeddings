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
