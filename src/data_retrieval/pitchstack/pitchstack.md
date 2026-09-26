# pitchstack

`PitchstackDeckDownloader` (`downloader.py`) collects pitchstack.gg deck
ids from the sitemap (`phase_1()`, `deck_ids.txt`) and each deck's full
record from `GET /v1/decks/{deck_id}` (`phase_2()`, `decks.jsonl`). A
deck record carries its deck version ids (`activeDeckVersionId`,
`deckVersions`) but no card list.

## Not yet implemented

- `GET /v1/deck_versions/{deck_version_id}/cards` - one deck version's
  card list. Needed before a `DeckExtractionStage` can be built for
  this source (see `src/data_refinement/deck_box/TODO.md`); the version
  ids to call it with are already in `decks.jsonl`.
- `GET /v1/deck_versions/{deck_version_id}/history` - a deck version's
  edit history (cards added/removed between versions). A possible
  future deck-evolution metric.
