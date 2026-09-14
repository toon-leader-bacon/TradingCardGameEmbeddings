# METRIC_BRAINSTORM — 17lands (index, retired)

This file's original per-source sections (draft data / game data / replay
data) have been superseded by three separate, deeper passes — each
written directly against the raw CSVs under the current `metric_writing`
skill, rather than this file's small pre-extracted samples (which no
longer exist in this checkout):

- [`draft_data/BRAINSTORM.md`](draft_data/BRAINSTORM.md) — 11 single-card
  + 9 multi-card/group candidates.
- [`game_data/BRAINSTORM.md`](game_data/BRAINSTORM.md) — 10 single-card +
  9 multi-card candidates.
- [`replay_data/BRAINSTORM.md`](replay_data/BRAINSTORM.md) — 13 single-card
  + 9 multi-card/group candidates.

Each of those resolves this file's open items directly: the
color-identity dependency (unblocked — `raw_content["color_identity"]`
on any card resolved via the MTG `CardBinder`), wheel rate's table-size
assumption (derive per-draft, don't hardcode 8), and replay data's
previously-unverified per-turn column semantics (attacker/blocker/
combat-kill columns resolved; both sides' board state turned out to be
fully public, only hand contents are actually asymmetric).

Kept here only as a historical index — do not add new ideas to this
file; add them to the relevant subdirectory's `BRAINSTORM.md` instead.
