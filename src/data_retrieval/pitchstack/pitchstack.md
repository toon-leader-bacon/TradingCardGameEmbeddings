# pitchstack

`src/data_retrieval/pitchstack/` (deck version id collection + per-
version card list collection) is implemented — see
`src/data_retrieval/README.md`'s `pitchstack/` entry for current state.

## Not yet implemented

Two more pitchstack.gg endpoints exist and are worth documenting for
future reference — not implemented, no stub methods exist for them in
`PitchstackDeckDownloader`:

- `GET /v1/deck_versions/{deck_version_id}/history` — a deck version's
  edit history (cards added/removed between versions over time).
  Potential future training metric (deck evolution).
- `GET /v1/decks/{deck_id}/meta` — deck-level metadata (e.g. format the
  deck was played in). **Open question**: `deck_id` (`d-<uuid>`) is a
  different id than `deck_version_id` (`dv-<uuid>`) collected by
  `phase_1()` — how one derives the other (e.g. whether `/cards` or a
  future response embeds the parent `deck_id`) is not yet confirmed.
  Do not guess at this relationship; a future pass building on top of
  `/meta` needs to confirm it against the live API first.
