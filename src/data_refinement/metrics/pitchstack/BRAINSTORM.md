# BRAINSTORM (pitchstack / Flesh and Blood tournament deck metadata)

A todo/idea list of candidate metrics for pitchstack.gg's tournament
deck records (`data/raw/pitchstack/decks.jsonl`, 1,678 records as of
this review — see `src/data_retrieval/pitchstack/downloader.py`).
None of this is designed or built yet — this is raw material for
picking what to build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed against the live 1,678-row `decks.jsonl`:

- **No card lists exist in this container yet.** Each record is one
  `GET /v1/decks/{deck_id}` response: `id`, `userId`, `name`,
  `author`, `heroId`, `format`, `visibility`, `createdAt`/`updatedAt`,
  `deckVersions`/`activeDeckVersionId`, `deckKind`, `sourceKind`,
  `sourceReference`, `tournamentType`, `eventId`. The
  `/v1/deck_versions/{id}/cards` endpoint that would supply actual
  card contents is confirmed to exist but deliberately not yet called
  (see `pitchstack.md`) — **every metric below that needs card
  contents is blocked on that future download work**, flagged
  individually.
- `heroId` (e.g. `arakni-marionette`, `cindra-dracai-of-retribution`)
  is populated on every record and is itself a rich signal even
  without card data — top heroes each appear 40-70+ times in this
  sample.
- `format`: `Classic Constructed` (1,210), `Silver Age` (255), `Living
  Legend` (112), `Blitz` (68), `Sealed` (24), `Commoner` (9) — a
  meaningfully skewed format distribution.
- `tournamentType`: `Battle Hardened` (803), `Calling` (357), `Sunday
  Showdown` (160), `National Championship` (116), `Pro Tour` (96),
  `Pro Quest` (92), `Road to Nationals` (31), plus a few smaller/rarer
  tiers — a real tournament-prestige hierarchy.
- `deckKind`/`sourceKind` are almost entirely
  `DECK_KIND_REFERENCE`/`DECK_SOURCE_KIND_TOURNAMENT_RESULT` (1,666/
  1,678) — this corpus is overwhelmingly *tournament result* decks,
  not casual/user-submitted ones.
- **`name` frequently embeds tournament placement as free text**,
  e.g. `"Sunday Showdown: Kansas City 2026 - 5th"`,
  `"... - 3rd"`, `"... - 1st"` — confirmed live, a genuinely
  structured-looking pattern (`<tournament>: <location> <year> -
  <placement>`) but not a dedicated field; extracting placement
  reliably would need its own parsing/validation pass across the
  corpus before being trusted as a label.
- `eventId` is populated on only 350/1,678 records — sparse, would
  need to be treated as optional/missing-often in any metric that
  uses it.

## Known biases / known-unknowns

- **This corpus is entirely tournament-result decks** — it says
  nothing about the wider population of decks played casually or that
  didn't place; any popularity/hero-representation stat here reflects
  competitive-tournament metagame share, not overall player-base
  preference.
- **Placement is unstructured** — flagged above; a metric built on
  parsed placement should be understood as resting on a heuristic
  regex over `name`, not a validated ground-truth field.
- **No card-content data exists yet** — this is the single biggest
  gap; a large fraction of the interesting metrics below (deck
  composition, card-hero synergy, masking tasks) are blocked until a
  future `/cards` download pass lands. Metadata-only metrics are the
  only ones buildable today.
- **`format` distribution is corpus-skewed, not necessarily
  metagame-representative** — Classic Constructed dominates this
  sample; any cross-format comparison should account for the very
  different sample sizes.

## Candidate Metrics

### Single-card (7, all blocked on future card-list download)

1. **Card Play Rate by Hero** — Input: single-card. Label:
   probability. `P(card in deck | heroId)` — blocked on card data;
   the FaB analogue of a hero-conditioned inclusion rate.
2. **Card Play Rate by Format** — Input: single-card. Label:
   probability. `P(card in deck | format)` — blocked on card data;
   surfaces cards whose viability differs sharply by format (e.g.
   Living Legend vs. Classic Constructed banlists/power level).
3. **Card Play Rate by Tournament Tier** — Input: single-card. Label:
   probability. `P(card in deck | tournamentType)` — blocked on card
   data; whether higher-tier (Pro Tour/Calling) decks favor different
   cards than lower-tier (Battle Hardened) ones.
4. **Card Placement Association** — Input: single-card. Label:
   regression-continuous. Average parsed placement (lower = better)
   across decks containing the card — blocked on both card data and
   the placement-parsing caveat above.
5. **Card Recency Trend** — Input: single-card. Label:
   regression-continuous. Play-rate trend over `createdAt`/
   `updatedAt` time buckets — blocked on card data; a metagame-drift
   signal.
6. **Cross-Hero Generic Card Popularity** — Input: single-card. Label:
   count. How many distinct heroes' decks include a given
   (non-hero-restricted) card — blocked on card data; a
   generic-utility proxy.
7. **Card Copy-Count by Format** — Input: single-card. Label: count
   (distribution). Typical copy count of a card, split by format —
   blocked on card data; formats like Blitz/Sealed likely show
   different curve shapes than Classic Constructed.

### Multi-card (11, mostly blocked on future card-list download)

8. **Hero Popularity by Format** — Input: multi-card (no card data
   needed — uses `heroId`×`format` directly). Label: classification
   (fixed-set, over heroes). `P(hero | format)` — the one rich metric
   buildable today without waiting on card downloads.
9. **Hero Popularity by Tournament Tier** — Input: multi-card (uses
   `heroId`×`tournamentType` directly, no card data needed). Label:
   classification (fixed-set, over heroes). `P(hero | tournamentType)`
   — whether certain heroes over/under-perform at higher-prestige
   events in terms of raw representation (not win rate — no outcome
   data beyond placement parsing exists).
10. **Placement-Conditioned Hero Rate** — Input: multi-card (uses
    `heroId` + parsed placement, no card data needed, but depends on
    the placement-parsing caveat). Label: probability. `P(hero | top
    N placement)` — which heroes disproportionately appear at the top
    of results, a coarse win-rate proxy in the absence of true match
    outcomes.
11. **Masked Card-in-Deck Prediction** — Input: multi-card (deck with
    one card masked). Label: classification (variable-set, over the
    legal card pool for the hero). Blocked on card data — this
    source's version of the standard masking task, once card lists
    exist.
12. **Deck-Pair Same-Hero Classification** — Input: multi-card (two
    full decks). Label: probability. `P(same_hero | deck_1, deck_2)`
    — blocked on card data; a sanity-check baseline.
13. **Deck Placement Regression from Card List** — Input: multi-card
    (full deck). Label: regression-continuous. Predict parsed
    placement from deck composition — blocked on card data and the
    placement-parsing caveat; the closest this source gets to a
    genuine "predict tournament success from deck" metric.
14. **Card Co-Occurrence / Synergy Statistic** — Input: multi-card
    (decks containing card A vs. card B, compared as two groups).
    Label: probability. `P(card_B in deck | card_A in deck)` —
    blocked on card data; comparable to the `fabtcg_decklists`
    version of this idea but drawn from a competitive-only sample.
15. **Format Legality Cross-Check** — Input: multi-card (full deck,
    joined against `cardvault_fabtcg`'s format-legality data if
    available). Label: probability (sanity-check). `P(every card in
    deck is legal in format | deck, format)` — blocked on card data;
    a data-quality filter more than a gameplay signal.
16. **Deck Similarity Within a Tournament** — Input: multi-card (two
    decks from the same `eventId`). Label: probability. `P(decks
    share a majority of cards | same event, same hero)` — blocked on
    card data; a "how much did the field converge on one build"
    signal, restricted to the 350 records with a populated `eventId`.
17. **Version-History Card Churn** — Input: multi-card (a deck's
    multiple `deckVersions` over time, if that history is later
    fetched via the deliberately-unimplemented `/history` endpoint).
    Label: count. Cards added/removed between versions — flagged as
    doubly blocked (needs both card data and the `/history` endpoint,
    neither implemented today).
18. **Sealed/Blitz Deck Size Sanity Check** — Input: multi-card (full
    deck, `format` in {Sealed, Blitz}). Label: probability
    (sanity-check). `P(deck card count matches the format's
    constructed-size rule | deck, format)` — blocked on card data.

### Multi-group (12, mixture of metadata-only and card-data-blocked)

19. **Hero-by-Format-by-Tier Cross-Tabulation** — Input: multi-group
    (hero counts grouped jointly by format and tournamentType, no
    card data needed). Label: count (distribution). A three-way
    breakdown of where each hero's representation concentrates —
    buildable today.
20. **Event Field Composition** — Input: multi-group (all decks
    sharing one `eventId`, as a group). Label: classification
    (variable-set, over heroes present). What hero diversity looked
    like at one specific event — buildable today for the 350 records
    with a populated `eventId`, no card data needed.
21. **Tournament-Tier Metagame Drift** — Input: multi-group (all decks
    in one `tournamentType`, bucketed by time from `createdAt`/
    `updatedAt`). Label: count (distribution). How hero representation
    within one tournament tier shifts over time — buildable today, no
    card data needed.
22. **Placement Distribution by Hero** — Input: multi-group (all
    decks for one hero, as a group, their parsed placements
    collected). Label: count (distribution). Whether a hero's results
    cluster near the top or are broadly spread — depends on the
    placement-parsing caveat but not on card data.
23. **Cross-Format Hero Portability** — Input: multi-group (one hero's
    decks across every format, each format a group). Label:
    probability. `P(hero appears in >= k formats | hero)` — whether
    a hero is a format-flexible pick or format-specific — buildable
    today, no card data needed.
24. **Deck-vs-Deck Full Comparison Within an Event** — Input:
    multi-group (all decks at one event, pairwise). Label:
    probability. `P(deck_1 beat deck_2 in a head-to-head | both
    decks, both from the same event)` — blocked: this source has no
    match-level (deck-vs-deck) result data at all, only aggregate
    placement; flagged as the most speculative multi-group idea, not
    buildable without a different data source.
25. **Field Diversity Index Over Time** — Input: multi-group (all
    decks in a format, bucketed by time, no card data needed). Label:
    regression-continuous. An entropy-style diversity score over hero
    representation per time bucket — buildable today.
26. **Group-Level Masked-Slot Prediction (per hero-archetype cluster)**
    — Input: multi-group (once card data exists: decks for one hero,
    clustered into candidate archetype groups, one group's typical
    card slots masked). Label: classification (variable-set). Blocked
    on card data; the pitchstack analogue of `fabtcg_decklists`'
    group-aware masking idea, but starting from a competitive-only,
    placement-labeled sample.
27. **Tournament-Tier Hero Ban/Errata Impact** — Input: multi-group
    (one hero's representation across tournamentType groups, before
    vs. after a known errata/balance-change date). Label:
    regression-continuous. Change in representation attributable to a
    known game-balance event — buildable today (no card data needed),
    but requires an external list of errata dates not currently in
    this project.
28. **Sourced-vs-User Deck Comparison** — Input: multi-group (the 1,666
    `DECK_SOURCE_KIND_TOURNAMENT_RESULT` decks vs. the 12
    `DECK_KIND_USER`/`DECK_SOURCE_KIND_UNSPECIFIED` ones, as two
    groups). Label: probability. `P(deck is tournament-sourced |
    deck's own features)` — a data-provenance classifier, useful
    mainly for understanding this corpus's own composition rather
    than a gameplay signal; the 12-record minority class is
    extremely small, flag as low-data.
29. **Event-Level Hero Co-Occurrence** — Input: multi-group (heroes
    present at the same event, as a group). Label: probability.
    `P(hero_B present at an event | hero_A present at that event)` —
    a metagame-interaction signal at the event level, no card data
    needed.
30. **Format Migration Signal** — Input: multi-group (one hero's decks
    split by format, time-ordered). Label: probability. `P(hero's
    representation shifts from format_A to format_B over time |
    hero)` — a coarse "is this hero migrating toward a new format"
    trend detector, no card data needed.
