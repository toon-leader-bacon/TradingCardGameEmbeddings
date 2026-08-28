**Resolved (2026-08-25):** this blocker is closed — `replay_data_metrics`
now exists (`arena_id_cache.py`, `replay_metric.py`,
`replay_metric_scanner.py`, `jobs/`), built via `design-recipe-skeleton`
against exactly the arena_id resolution path this file predicted below
(`CardBinder.get_by_alias(source_game, DataSource.ARENA, arena_id)`).
Kept as-is (not deleted) since `README.md` references this file's
original per-turn-telemetry analysis. See `README.md` for the current
design.

---

# TODO: replay_data_metrics is blocked on card-identity work, not just a metrics build

**Update (CardBinder refactor):** the "CardRegistry has no way to
resolve an Arena ID today" half of this problem is now resolved —
`CardBinder`/`AliasLedger`/`ScryfallCardIngestionStage` (see
ARCHITECTURE.md's "17lands ingestion + card binder + metric engine"
section) index `arena_id` (and `mtgo_id`/`mtgo_foil_id`/
`multiverse_ids`) alongside `oracle_id`, resolvable via
`CardBinder.get_by_alias(source_game, DataSource.ARENA, arena_id)`.
The second problem below — resolving a pipe-delimited CELL of many
Arena IDs, not a single id — is still open and unaddressed by that
refactor; ARCHITECTURE.md explicitly defers it until
`replay_data_metrics` itself is designed.

Confirmed by directly inspecting a real file
(`data/tmp/replay_data/MSH.PremierDraft.csv`, 2.5GB, 2615 columns, one
row per game) on 2026-08-24. This is a genuinely different shape from
`game_data`/`draft_data` and needs real design work before a
`design-recipe-skeleton` pass makes sense here.

## The core problem: card references aren't names, and aren't indexed

Almost the entire file is per-turn telemetry with no card references
at all — columns like `user_turn_1_cards_drawn`,
`oppo_turn_12_creatures_cast`, `user_turn_1_eot_user_life`, repeated
for ~30 turns × ~30 stats × 2 players. No `deck_<name>`-style columns
exist anywhere.

The only card references live in `candidate_hand_1`–`7` and
`opening_hand`, and they are **not card names** — each cell is a
pipe-delimited string of Arena's internal numeric card IDs, e.g.:

```
'104936|104917|105170|105174|104926|105180|104936'
```

Two separate problems, not one:

1. **`CardRegistry` has no way to resolve an Arena ID today.**
   `ScryfallCardIngestionStage` only indexes Scryfall's `oracle_id` as
   `source_id` — nothing indexes `arena_id`.
2. **The value shape is different.** A cell holds a *list* of IDs
   (pipe-delimited), not a single per-card column the way
   `game_data`/`draft_data` work — `column_lookup.py`'s whole
   approach (discover names from the header, resolve once) doesn't
   apply; this needs per-cell parsing plus resolution of each ID in
   the list.

## A correction to my own first assumption

I (the assistant) initially guessed this would need "a dedicated
downloader for Arena card data" as a new raw source. That's not
obviously true and shouldn't be assumed going in — **Scryfall's own
oracle-cards dump already carries `arena_id`** as a field inside each
card's raw JSON (confirmed directly: the Bruce Banner card we
inspected earlier in this project has `"arena_id":104942` right in its
Scryfall row). So the missing piece may just be:

- Adding an `arena_id` index to `CardRegistry` (a `get_by_arena_id`-
  style lookup, parallel to `get_by_source_id`), fed from data
  `ScryfallCardIngestionStage` is *already downloading* — not a new
  raw source at all.

A dedicated Arena-ID-specific downloader would only become necessary
if Scryfall's `arena_id` coverage turns out to be incomplete for cards
that actually show up in `candidate_hand`/`opening_hand` values (e.g.
older sets, or non-Arena-legal promos that still appear historically)
— that's an open question, not a settled fact. Check Scryfall's actual
`arena_id` coverage against a real `replay_data` file's ID list before
assuming a new source is needed.

## Open questions to resolve in an /architect session before skeleton

- Where does Arena-ID resolution live? A new `CardRegistry` index
  (`get_by_arena_id`), analogous to the existing
  `(source_game, source_id)` index — but `source_id` is currently a
  single field per `GenericCard`, already fixed to Scryfall's
  `oracle_id`. Does `arena_id` need its own dedicated index/field, or
  can the existing `source_id` machinery be generalized to support
  multiple ID namespaces per card? (This is the same shape of question
  the `card_registry` collision-handling work already worked through
  once — worth re-reading that section of `ARCHITECTURE.md` before
  redesigning from scratch.)
- What does "resolve a pipe-delimited cell of IDs" look like as a
  reusable primitive — does it belong in `card_registry` (generic:
  "resolve many IDs to many cards") or stay local to
  `replay_data_metrics` (17lands-specific: "parse this cell format")?
- What metrics are actually meaningful given this file's real
  granularity? Aside from `opening_hand`/`candidate_hand` (which give
  real per-card signal — e.g. "keep rate," "mulligan rate by card in
  hand"), the bulk of the file is turn-level *aggregate* counts
  (`creatures_cast`, `lands_played`, etc.) that don't name *which*
  card did the casting/playing — so most per-turn columns may not be
  usable as per-card metrics at all without a fundamentally different
  join than anything built so far.

Not starting implementation until these are actually discussed — this
file exists so the next session doesn't have to re-derive the above
from scratch.
