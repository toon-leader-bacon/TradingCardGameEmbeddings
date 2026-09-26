# METRIC_BRAINSTORM — 17lands draft_data

A candidate metric list for `data/raw/17lands/draft_data/*.csv`, produced
by running the `metric_writing` skill fresh against the raw CSV
(`MSH.PremierDraft.csv`, sampled directly with `ConvertFrom-Csv` over
the first ~3000 lines — no Python interpreter is available in this
environment, only PowerShell/Bash) rather than the small pre-extracted
samples the original top-level `../BRAINSTORM.md` was written from.
This supersedes that file's "## Draft data" section. Built so far:
six of these ideas (see `README.md`'s Files for which).

## What this session verified directly (corrections to the old draft)

- **Row order is grouped by `draft_id`, then monotonically increasing
  `(pack_number, pick_number)` within it.** Confirmed on a 3000-row/72-
  draft sample: zero cases of a `draft_id` reappearing after a
  different `draft_id` started. This means a `CsvScanner`-style chunked
  read can safely keep small per-`draft_id` state and evict it once a
  new `draft_id` is seen, without needing a full-file sort pass first —
  the "may span chunk boundaries" risk the old doc flagged is real
  (a draft's rows could straddle a chunk edge) but the *within-file
  ordering* risk it also flagged is not; only chunk-boundary carryover
  needs handling, not out-of-order arrival.
- **"Wheel" is directly observable within one `draft_id`, not a
  cross-`draft_id` join.** Each `draft_id` is one tracked player's own
  seat, but a pack **passes through every seat and back** before that
  player's `pick_number` sequence for one `pack_number` ends — for this
  sample (`MSH.PremierDraft`), every `pack_number` runs `pick_number`
  0..13 (14 picks/pack, an 8-seat premier draft table). A card seen in
  `pack_card_<name>` at `pick_number = k` "wheels" if it's still in
  `pack_card_<name> > 0` at `pick_number = k + table_size` (here,
  `k + 8`) within the *same* `(draft_id, pack_number)`. Table size and
  pack size are **per-set/per-event_type facts to derive at scan time**
  (`max(pick_number) + 1` observed for that `draft_id`), not a hardcoded
  constant — `HOB.PickTwoDraft.csv`/Cube events plausibly differ from
  this sample's 14.
- **`pool_<name>` is the count owned *entering* this pick, not
  including the current row's own pick.** Verified directly: picking
  "The Wondrous Wasp" at `pick_number=0` shows `pool_The Wondrous
  Wasp = 0` on that same row, becoming `1` starting `pick_number=1`.
  Any pool-based metric must read `pool_<name>` as "before this pick,"
  never accidentally double-count the row's own `pick`.
- **`pick_2` was blank on every sampled row** (`MSH.PremierDraft` is a
  plain `PremierDraft` event). It's a separate mechanic's column (see
  `HOB.PickTwoDraft.csv` in the same directory) — a second simultaneous
  pick for "pick two" formats, not a general secondary-choice field.
  Any metric reading `pick` should not assume `pick_2` is always
  present/meaningful; gate on `event_type` if it's ever used.
- **`pick_maindeck_rate`/`pick_sideboard_in_rate` are genuine
  precomputed per-(card, row) aggregates**, ranging over `[0, 1]`
  (observed `0.0`/`1.0`/fractional values in-sample) — confirmed
  reusable as-is, no re-derivation needed.
- **Card-name resolution risk, not yet hit in this sample but worth
  flagging before build time**: 17lands' `pack_card_<name>`/`pool_<name>`
  column-name convention and Scryfall's canonical `name` field are known
  to diverge for split/adventure/MDFC cards (17lands commonly uses only
  the front/first face, Scryfall's `name` is `"A // B"`). MSH's card
  pool didn't surface a clear case in-sample, but `CardBinder.get_by_name_single`
  should not be assumed to always hit — a metric-building pass should
  budget for a fallback (e.g. try splitting on `" // "` and matching the
  first face) and log/count unresolved names rather than silently
  dropping or raising per-row.
- **Color identity is unblocked**: resolving a `pack_card_<name>`/`pick`
  string via `CardBinder.get_by_name_single(GameId.MTG, name)` returns a
  `GenericCard` whose `raw_content` is the verbatim Scryfall JSON —
  `raw_content["color_identity"]` (list of `W/U/B/R/G`) is present per
  Scryfall's own schema. This removes the old doc's "open dependency"
  blocker for every color-commitment idea below.
- `data/final/cards/mtg.jsonl` does not exist yet in this checkout —
  the Scryfall `CardIngestionStage` exists
  (`card_binder/scryfall/ingestion_stage.py`) but hasn't been run to
  produce a binder file. Any of these metrics needs that build step
  done first (out of scope for this brainstorm pass).

---

## Single-card metrics (11)

1. **Card Take Rate** — `P(picked | in pack, pick_number, pack_number)`.
   Single card. Label: probability (continuous, `[0,1]`). Accumulator:
   running `(times_in_pack, times_picked)` per card, keyed off
   `pack_card_<name) > 0` and `pick == <name>`. The conditioning on
   `pick_number`/`pack_number` matters — take rate is expected to drift
   as packs thin out (see #7).

2. **Average Last Seen At (ALSA)** — average `pick_number` of the last
   row within a `(draft_id, pack_number)` group where
   `pack_card_<name> > 0`, before the card is gone. Single card. Label:
   regression-continuous (a pick-number float). Accumulator, grouped by
   `(draft_id, pack_number)` — safe given the confirmed row-ordering
   guarantee above, modulo chunk-boundary carryover.

3. **Wheel Rate** — `P(pack_card_<name> > 0 at pick_number = k + table_size | pack_card_<name> > 0 at pick_number = k)`,
   for every `k` where `k + table_size` is still a valid `pick_number`
   in that `(draft_id, pack_number)`. Single card. Label: probability.
   Accumulator; `table_size` derived per-draft as
   `max(pick_number) + 1` for that `(draft_id, pack_number)`, not
   hardcoded — see verified note above. This is now a well-defined,
   directly-computable metric (the old doc flagged this as an
   assumption needing confirmation; it's confirmed computable from
   single-seat data alone).

4. **First-Pick Rate (P1P1 Take Rate)** — `P(pick == card | pack_number == 0, pick_number == 0, card in pack)`.
   Single card. Label: probability. Accumulator. Isolates
   bomb/first-pick signal from take rate's overall pick-speed average.

5. **Hate-Pick / Overdraft Signal** — average `pick_number` at which a
   card is taken, restricted to rows where that pick's own
   `pick_maindeck_rate` is low (e.g. below a fixed threshold, tunable
   at build time — say `< 0.2`). Single card. Label: regression-
   continuous (average pick_number, low-maindeck-rate picks only).
   Accumulator, reuses `pick_maindeck_rate` directly rather than
   re-deriving playability.

6. **Rank-Stratified Take Rate** — same shape as #1, but with a
   separate running `(times_in_pack, times_picked)` tally per `rank`
   bucket (`bronze`/`silver`/`gold`/`platinum`/`diamond`/`mythic` —
   confirmed as the full `rank` vocabulary in-sample). Single card
   (extra stratifying dimension). Label: probability, per rank bucket.
   Accumulator. Surfaces whether strong drafters value a card
   differently than the field average — sharper than the old doc's
   "rank-stratified pick number" since take rate also captures cards
   strong drafters skip entirely.

7. **Pick-Number Decay Curve** — for a card, `P(picked | in pack)` as a
   function of `pick_number` within its `pack_number` (a per-card
   14-bucket, here, table-size-bucket histogram rather than one
   scalar). Single card. Label: classification-variable-set in spirit
   (a fixed-length vector keyed by pick_number bucket) or just a
   `list[float]` column. Accumulator. A genuinely new idea this
   session — cards with a flat decay curve are "always fine," cards
   that cliff hard past pick 3-4 are speed-picked signal-reads;
   distinguishes that from #1's single scalar.

8. **Color-Pair Pick Rate** — given the picked card's own
   `color_identity` (now unblocked, see above), `P(picked | in pack,
   card's color pair)` — i.e., is this specific card overperforming or
   underperforming its color pair's baseline pick rate? Single card.
   Label: a signed float (card's take rate minus its color pair's
   average take rate). Accumulator — needs a second, color-pair-keyed
   running tally alongside #1's per-card one.

9. **Speculative Splash Signal** — rate at which a card is taken
   unusually early/late relative to its own average pick number,
   *conditioned on* the card's own color being off the drafter's
   two-color pair inferred from `pool_<name>` at pick time (a same-
   dataset proxy — no cross-file join to `game_data`'s `splash_colors`
   needed, since `pool_<name>` at any pick already tells you what's
   been committed to so far). Single card. Label: regression-
   continuous (pick-number delta). Accumulator — same
   `(draft_id)`-scoped state as #2/#3's family, plus a per-pick color-
   identity read on every `pool_<name) > 0` card.

10. **Sideboard-In Rate Correlation** — is a card's
    `pick_sideboard_in_rate` (already on the row when picked) predictive
    of it being an early or late pick? Single card. Label:
    regression-continuous (correlation-style scalar, or just report the
    raw average `pick_sideboard_in_rate` per card alongside #1, letting
    a dojo compute correlation downstream). Accumulator — cheapest
    metric in this list, a straight running average of an already-
    present column.

11. **P1P1-vs-Overall Take Rate Delta** — `#4` minus `#1`, per card, as
    its own explicit output rather than left for a dojo to compute from
    two separate parquet files. Single card. Label: signed float. This
    genuinely needs both accumulators to run together (same underlying
    tallies as #1 and #4) — flagged here as a candidate for whichever
    one metric class ends up producing it, not as evidence it needs a
    third accumulator pass.

---

## Multi-card / multi-group metrics (9)

1. **Pool Color-Commitment Prediction** — given the pool
   (`pool_<name>` columns, at any pick where at least N cards are
   owned) as a multi-card group, predict the color pair the pool's
   cards' `color_identity` values will ultimately concentrate into by
   the *end* of the same draft (last row of that `draft_id`, not a
   cross-file join to `game_data.main_colors` — this draft_id's own
   final pool already tells you the outcome). Multi-card (pool group).
   Label: classification (fixed-set — one of the 10 two-color pairs,
   or a mono/3+-color overflow bucket). Streaming in spirit per
   snapshot-row, but needs the *same draft's own last row* as the
   label source, so practically an accumulator that buffers per-
   `draft_id` snapshots and resolves the label at `finalize()`.

2. **Color Openness / Pack Depletion Signal** — as `pick_number`
   increases within one `(draft_id, pack_number)`, the color
   distribution (via `color_identity`) of cards remaining in
   `pack_card_<name>`. Doesn't fit single/multi/multi-group cleanly —
   flagged again here, unresolved, per the old doc; the unit is "a
   color's representation in a shrinking set," not a card or named
   group. Accumulator regardless (per-`(draft_id, pack_number)` running
   color tallies, snapshotted per `pick_number`).

3. **Draft Color-Trajectory Consistency** — across one full draft
   (a `draft_id`'s ordered rows), an entropy/consistency score of the
   pool's inferred color distribution as it evolves pick-to-pick.
   Multi-card (pool group, evolving). Label: regression-continuous (one
   scalar per draft, keyed to... itself, not a card — see note below).
   Accumulator, same `(draft_id)`-scoped state as the single-card
   family above.

4. **Pack-to-Pick Choice Set** — for every row, the full set of cards
   in `pack_card_<name>` (options) plus which one was `pick`ed — a
   direct multi-group "options vs. choice" contrastive/ranking example,
   distinct from #1's per-card take-rate aggregation: this preserves
   the actual competing-options context per decision instead of
   collapsing it into a marginal probability. Multi-group (the pack's
   full option set is one group, the single `pick` is the label within
   it). Label: ranking / CLIP-style contrastive (pick is the positive,
   every other pack member a negative, for that one row). Streaming —
   one row already carries a complete example, no cross-row state
   needed. This is the richest new idea from this pass: `README.md`'s
   ungrouped multi-card model plus a downstream dojo-side comparison
   head is exactly the shape this metric is built for.

5. **Pool-Conditioned Pick Prediction** — given the pool so far
   (`pool_<name>`, multi-card group) AND the current pack options
   (`pack_card_<name>`, a second multi-card group), predict which pack
   option gets picked. Multi-group (pool group + pack-options group).
   Label: ranking / classification-variable-set (the pick, chosen from
   the pack-options group). Streaming — richer than #4 alone since it
   also conditions on the drafter's committed pool, letting a dojo learn
   "given what I'm already building, what do I take" rather than
   context-free take rate.

6. **Wheel-Conditioned Choice Set** — a variant of #4 restricted to
   rows at `pick_number >= table_size` within a `(draft_id,
   pack_number)` — i.e., pack-options-choice examples where the pack
   has already wheeled once. Multi-group, same shape as #4. Label:
   ranking. Streaming. Separated out because the *option pool itself*
   here is a biased sample (everything strong has already been taken by
   7 other drafters), a genuinely different distribution than #4's
   full-field version — worth its own metric rather than folding into
   #4 and losing that distinction.

7. **Full-Draft Pool → Final Color Pair** (multi-group, whole-draft
   granularity) — same predictive target as #1, but as a single
   streaming example per `draft_id` (final pool only, one row output)
   rather than #1's per-snapshot accumulator version. Multi-card
   (full final pool as one group). Label: classification (fixed-set,
   color pair). Streaming — the pool at the *last* row of a `draft_id`
   already is the complete input, no running state needed if a scanner
   is willing to buffer one `draft_id` at a time and emit on
   `draft_id` change. Kept distinct from #1 since #1's value is in the
   *sequence* of predictions across pick numbers (useful for a dojo
   studying commitment timing), while this is the simpler single-shot
   version a dojo might prefer for pure accuracy benchmarking.

8. **Rank-Stratified Pack Composition Bias** — do higher-rank drafters'
   packs (`pack_card_<name>` at pick 0 of pack 0, i.e. an un-touched
   pack) skew differently in card-quality distribution than lower-rank
   drafters' — likely near-zero signal (packs are randomly assigned,
   not rank-dependent) but worth a cheap check/negative-control metric:
   if this shows real signal, something upstream (e.g. rank-correlated
   event/set filtering) is confounding other metrics in this list.
   Multi-card (pack group). Label: regression-continuous (a pack-
   quality proxy, e.g. average `pick_maindeck_rate` across the pack's
   cards) per rank bucket. Accumulator.

9. **Draft Archetype Trajectory (multi-group, sequence-level)** — the
   *sequence* of per-pick `(pool_snapshot, pack_options, pick)` triples
   for a full `draft_id`, as one training example per draft rather than
   per pick — for a model architecture that consumes a whole draft's
   decision sequence at once (contrast #4/#5's per-pick streaming
   granularity). Multi-group, draft-level. Label: none inherent — this
   is closer to a self-supervised sequence-modeling input than a
   labeled metric, flagged here as a stretch/architecture-level idea
   rather than a concrete P(...) statement, worth a design conversation
   before committing to an output shape.

---

## Open items for the build phase (not resolved here, brainstorm-only pass)

- Confirm the split/MDFC card-name mismatch risk against the actual
  `mtg.jsonl` binder once it's built, across more than one set (MSH
  alone didn't surface a clear case).
- Decide a real threshold for #5's "low `pick_maindeck_rate`" cutoff
  (used `0.2` as a placeholder above).
- #2, #8, and #9 above don't cleanly fit the single/multi/multi-group
  taxonomy or the accumulator/streaming split as neatly as the rest —
  flagged, not resolved, consistent with the original doc's treatment
  of its own "Color Openness" idea.


## Human Review Short List of Top Metrics

Single Card Metrics
- **Card Take Rate** — `P(picked | in pack, pick_number, pack_number)`
- **First-Pick Rate (P1P1 Take Rate)** — `P(pick == card | pack_number == 0, pick_number == 0, card in pack)`
- **Rank-Stratified Take Rate**
- **Pick-Number Decay Curve**

Multi Card metrics
- **Pack-to-Pick Choice Set**

Multi Group Metrics:
- **Pool-Conditioned Pick Prediction**
- Given a pool, a pack, and the selection, predict which Rank the player who's drafting is (Classification task)