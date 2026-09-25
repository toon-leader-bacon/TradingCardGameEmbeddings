# TODO

## Bug fixes

- ~~**`leader_labels.py`'s frozen `LEADER_NAMES` tuple is stale...**~~
  **Fixed 2026-09-25.** Root cause: `leader_deck_counts.py` built
  `LEADER_NAMES` by counting playgwent.com's raw guide text
  (`leader.name`), but `LeaderMaskedFromDeckMetric` validates against
  the gwent.one `CardBinder`'s canonical name (via `leaderId` ->
  `get_by_alias`) — a different source that spells some leaders
  differently. Fixed `count_decks_per_leader()` to resolve and count by
  the same canonical `CardBinder` name the metric actually checks
  against, then regenerated `LEADER_NAMES` from it: `"Doulbe Cross"` ->
  `"Double Cross"`, `"Reckless Fury"` -> `"Reckless Flurry"` (still 42
  leaders total, no new ones). See `leader_labels.py`'s FRESHNESS note
  and this directory's `README.md` for the corrected methodology.
  Original notes, kept for context:

  During the 2026-09-25 metrics
  regeneration, `LeaderMaskedFromDeckMetric` logged, ~3700 times total:

  ```
  LeaderMaskedFromDeckMetric: leader card UUID(...) has name '<name>',
  which is not in LEADER_NAMES - falling back to OTHER (leader_labels.py's
  frozen list may need regenerating)
  ```

  for two names, neither of which is in the frozen tuple:
  - `Double Cross` — 2,500 occurrences (`UUID('4f3e249a-d5fd-4edb-8480-f0a3634c655c')`).
    The tuple already has an entry for this same leader, but spelled
    `Doulbe Cross` — the file's own docstring explains that spelling is
    a *deliberate* verbatim match of a typo in playgwent.com's raw guide
    data, not a mistake introduced here. So either the raw corpus now
    also contains (or has switched to) the correctly-spelled `Double
    Cross` for the same leader, or this metric resolves the leader name
    from a different source (e.g. the card's canonical name via the
    gwent_one CardBinder) than the `count_decks_per_leader()` scan that
    originally built this tuple from `guides.jsonl` text — worth checking
    which, since it changes whether the fix is "add `Double Cross` as a
    second alias for the same leader" or "the name-resolution path
    itself drifted from what generated this list."
  - `Reckless Flurry` — 1,243 occurrences (`UUID('620a3284-de9c-43ee-8276-534a15657aa2')`).
    The tuple has `Reckless Fury` (no "l"), which reads as a genuinely
    different name, not just a typo variant — likely either a renamed
    leader or a second, newer leader that didn't exist (or wasn't
    sampled) in the 2026-09-09 snapshot the list's FRESHNESS comment
    describes.

  This is exactly the staleness scenario `leader_labels.py`'s own
  docstring already warns about ("a future gwent.one/playgwent.com
  expansion adding a new leader will silently be absent from
  LEADER_NAMES until this list is regenerated"). Fix: re-run
  `leader_deck_counts.count_decks_per_leader()` against the current
  `data/raw/play_gwent/guides.jsonl`, diff its keys against
  `LEADER_NAMES`, and regenerate the tuple (deciding, for `Double
  Cross`/`Doulbe Cross`, whether both spellings need to stay as
  aliases or whether one has fully replaced the other in current data).
