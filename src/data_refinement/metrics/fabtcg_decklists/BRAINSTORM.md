# BRAINSTORM (fabtcg_decklists / Flesh and Blood tournament decklists)

A todo/idea list of candidate metrics for fabtcg.com's published
tournament decklists (`data/raw/fabtcg_decklists/decklists/*.html`,
1,774 files as of this review — see
`src/data_retrieval/fabtcg_decklists/downloader.py` and the existing
`DecklistCardsMetric`/`FabtcgDecklistsExtractionStage` in
`deck_box/fabtcg_decklists/`). None of this is designed or built yet
— this is raw material for picking what to build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed by sampling the saved decklist fragments (one
`<section class="decklist-list-view block hidden">` per file, per
`downloader.py`'s `_extract_decklist_fragment`):

- Every deck is split into exactly 5 named groups via `<h3>` headers:
  `Hero / Weapon / Equipment`, `Pitch 0`, `Pitch 1`, `Pitch 2`,
  `Pitch 3` (confirmed across 30 sampled files). This is Flesh and
  Blood's real deck-construction structure — pitch value buckets, plus
  a combined hero/gear group — not an artifact of the site's own
  presentation.
- Each group is a `<ul class="cards-container">` of `<li
  class="card-item">` entries, each giving a copy count (`1x`, `2x`,
  `3x`) and a card name (`fragment_parsing.py`'s
  `iter_card_quantities_and_names()` already parses this shape — see
  `deck_box/fabtcg_decklists/README.md`).
- The existing `DecklistCardsMetric`/`FabtcgDecklistsExtractionStage`
  both flatten every group into one `card_nocab_uuids` list, discarding
  which group each card came from — a deliberate first-pass
  simplification per `legacy/fabtcg_decklists/TODO.md`. The group
  label is available for free at parse time and would need to be
  threaded through (not re-derived) for any of the multi-group metrics
  below.
- The saved filename is the deck's URL slug, which typically encodes
  the player name, hero name, and often a tournament/event name/date
  (e.g. `cayle-mccreath-bravo-deck-new-zealand-nationals-22-1-22.html`,
  `gabe-sher-lexi-deck-calling-antwerp-220523.html`) — this is
  free-text, not structured metadata; extracting event/date reliably
  would need its own parsing pass and isn't confirmed to be
  consistently formatted across all 1,774 slugs.
- Card *names* only are extracted here — this source has no card-
  intrinsic data of its own (no cost/pitch/class/type fields). Every
  card-intrinsic fact used below is expected to be joined in from
  `cardvault_fabtcg`'s `CardBinder` entries once a card name resolves
  to a `nocab_uuid`, the same cross-source resolution
  `sts_gg`/`spire_codex` already demonstrate for Slay the Spire 2.

## Known biases / known-unknowns

- **No win/loss or placement data** — this source is decklists only;
  a deck's mere presence here means it was played at some tournament,
  not that it won or performed well. Any metric implying deck quality
  from inclusion alone would be unfounded.
- **Only decks players/organizers chose to publish** — likely skews
  toward decks that did reasonably well (a classic survivorship bias,
  the same shape flagged for sts2runs) or toward specific
  events/regions the site chose to cover; not a random sample of all
  tournament decks played.
- **Card-intrinsic joins depend on cardvault_fabtcg's ingestion having
  run first** — a name that fails to resolve against that
  `CardBinder` (e.g. a promo/alt-art printing under a slightly
  different name) would silently drop that card from any joined
  metric; this hasn't been checked at scale.
- **Slug-derived metadata (event/date) is unverified/unstructured** —
  flagged above; don't treat it as reliable without a dedicated
  parsing pass first.

## Candidate Metrics

### Single-card (10, mostly cardvault_fabtcg-joined)

1. **Deck Inclusion Rate** — Input: single-card. Label: probability.
   `P(card in deck | card is deck-legal for the deck's hero)` — the
   FaB analogue of a "play rate" stat, the base rate any of the
   pairwise/group metrics below should be compared against.
2. **Hero-Conditioned Inclusion Rate** — Input: single-card (with
   hero as a conditioning attribute, not a second input). Label:
   probability. `P(card in deck | hero)` — surfaces cards that are
   staples for one hero but rare for others.
3. **Pitch-Bucket Assignment Rate** — Input: single-card. Label:
   classification (fixed-set: Pitch 0/1/2/3). Given a card's own
   `true_pitch` (joined from `cardvault_fabtcg`), predict which pitch
   bucket it's actually slotted into across observed decks — most
   cards should trivially match their own pitch value; deviations
   (e.g. a pitch-1 card slotted as a 3-drop for on-chain reasons) are
   the interesting minority.
4. **Card Copy-Count Distribution** — Input: single-card. Label:
   count (distribution: 1x/2x/3x). How many copies of a card decks
   typically run, when run at all.
5. **Class-Restricted Card Usage Legality Check** — Input: single-card
   (joined class/talent restriction from cardvault_fabtcg). Label:
   probability (sanity-check). `P(card's class/talent restriction is
   compatible with the deck's hero | card in deck)` — should be ~1.0;
   violations flag either a data error or a rules exception worth
   understanding.
6. **Weapon/Equipment Slot Fill Rate** — Input: single-card
   (Equipment-typed, joined `face_1_subtypes` slot from
   cardvault_fabtcg). Label: probability. `P(slot is filled in deck |
   hero, slot)` — which body slots (Head/Arms/Chest/Legs/Item) are
   commonly left empty vs. always filled.
7. **Generic vs. Class Card Ratio Contribution** — Input: single-card.
   Label: probability. `P(card is Generic-class | card in an average
   deck)` — a per-card contribution to the deck's overall
   Generic-vs-class-card mix (see the corpus-level version below).
8. **Card Popularity Trend Across the Corpus** — Input: single-card.
   Label: count. Raw count of decks in the corpus containing a given
   card — a popularity/staple-ness proxy, cheap to compute, useful for
   stratified sampling of any other metric here.
9. **Card Cost-Curve Position** — Input: single-card (joined
   `true_cost`). Label: regression-continuous. Average deck size at
   the cost the card occupies — whether cheap cards cluster into
   higher-count decks or vice versa.
10. **Rarity-vs-Inclusion Association** — Input: single-card (joined
    `rarity`, if resolvable). Label: probability. `P(card in deck |
    rarity)` — whether rarer/harder-to-acquire cards see lower play
    rates independent of power level, a collector-availability signal
    rather than a gameplay one.

### Multi-card (11)

11. **Masked Hero Prediction** — Input: multi-card (the non-Hero cards
    of a deck, i.e. the rest of the Hero/Weapon/Equipment group plus
    all four pitch groups). Label: classification (fixed-set, over
    known heroes). Given the rest of the deck, predict the masked
    Hero card — this project's Commander-guessing analogue for FaB.
12. **Masked Card-in-Deck Prediction** — Input: multi-card (deck with
    one non-Hero card masked). Label: classification (variable-set,
    over the legal card pool for that hero). Predict the masked card
    from the rest of the deck.
13. **Weapon Prediction from Hero + Rest of Deck** — Input: multi-card
    (Hero card + Pitch groups, Weapon masked). Label: classification
    (variable-set, over Weapon-typed cards). Predict the deck's
    Weapon choice — narrower and more specific than #12.
14. **Deck-Pair Same-Hero Classification** — Input: multi-card (two
    full decks). Label: probability. `P(same_hero | deck_1, deck_2)`
    — should be near-trivial from the Hero group alone, included as a
    sanity-check baseline (same role as cardvault_fabtcg's
    pitch-sufficiency idea).
15. **Deck Archetype Clustering Signal** — Input: multi-card (two full
    decks, same hero). Label: probability. `P(decks belong to the
    same informally-recognized archetype | deck_1, deck_2)` — no
    ground-truth archetype labels exist in this source, so this would
    need an unsupervised similarity proxy (e.g. card-overlap ratio)
    rather than a true classification target; flagged as speculative.
16. **Pitch-Curve Shape Prediction** — Input: multi-card (deck's Hero
    + Equipment group only). Label: count (distribution: cards per
    pitch bucket). Predict the deck's overall Pitch 0/1/2/3 card-count
    distribution from just its Hero and gear choices.
17. **Card Co-Occurrence / Synergy Statistic** — Input: multi-card
    (all decks containing card A vs. all containing card B, compared
    as two groups). Label: probability. `P(card_B in deck | card_A in
    deck)` — the FaB analogue of the sts2runs co-occurrence idea,
    from real tournament data rather than card-text inference (unlike
    cardvault_fabtcg's talent-based proxy for the same concept).
18. **Deck Size Consistency Check** — Input: multi-card (full deck).
    Label: probability (sanity-check). `P(deck's total non-Hero,
    non-Weapon, non-Equipment card count matches FaB's 60-card
    constructed deck-size rule)` — a data-quality filter more than a
    gameplay signal.
19. **Equipment Loadout Prediction** — Input: multi-card (Hero + Pitch
    groups, Equipment masked). Label: classification (variable-set,
    over Equipment cards legal for the hero). Predict the deck's full
    Equipment loadout (potentially multiple slots) from the rest of
    the deck.
20. **Generic-Card Ratio Prediction** — Input: multi-card (partial
    deck: one pitch group). Label: regression-continuous. Predict
    what fraction of the full deck's cards are Generic-class, from a
    single pitch bucket alone.
21. **Cross-Hero Card Overlap** — Input: multi-card (two decks with
    different heroes). Label: count. Number of shared non-Hero,
    non-class-restricted cards between two decks for different
    heroes — a generic-card-utility signal.

### Multi-group (9)

22. **Group-Aware Masked-Slot Prediction** — Input: multi-group
    (deck's 4 groups: Hero/Weapon/Equipment, Pitch 1, Pitch 2, Pitch 3
    — Pitch 0 held out as the target). Label: classification
    (variable-set, over cards typically found in Pitch 0). Given the
    other groups, predict the Pitch 0 group's contents — the richest
    version of the masking family, genuinely exercising per-group
    structure rather than a flattened list (the deferred idea named
    in `legacy/fabtcg_decklists/TODO.md`).
23. **Cross-Group Balance Prediction** — Input: multi-group (any 3 of
    the 4 pitch groups). Label: count (distribution). Predict the
    4th pitch group's card count, given the other three — decks
    generally balance pitch buckets against a target curve, so this
    tests whether that regularity is learnable.
24. **Deck-vs-Deck Same-Archetype Prediction** — Input: multi-group
    (two full decks, each itself a group-of-groups). Label:
    probability. `P(same archetype | deck_1's groups, deck_2's
    groups)` — the group-structured version of #15, letting a model
    weigh per-slot similarity (e.g. same Equipment loadout) rather
    than raw card overlap.
25. **Hero-Metagame Membership Prediction** — Input: multi-group (one
    hero's full set of decks in the corpus, as a group, vs. a
    candidate new deck). Label: probability. `P(candidate deck fits
    the hero's established metagame | hero's deck corpus, candidate
    deck)` — an outlier/novelty-detection framing.
26. **Pitch-Group Composition Prediction Across Two Decks** — Input:
    multi-group (deck_1's Pitch 1 group + deck_2's Hero group).
    Label: probability. `P(deck_1's Pitch 1 group and deck_2's Hero
    choice are commonly paired | deck_1's Pitch 1 group, deck_2's
    Hero)` — a cross-deck compatibility signal for suggesting
    Hero/pitch-curve pairings.
27. **Full-Deck Reconstruction from Partial Groups** — Input:
    multi-group (Hero/Weapon/Equipment group + 2 of 3 pitch groups).
    Label: classification (variable-set, joint over the missing
    group's likely card set). The most demanding masking variant —
    reconstructing an entire missing pitch group at once, rather than
    one masked card.
28. **Metagame Diversity Over Time** — Input: multi-group (all decks
    for one hero, bucketed by inferred event/date from the slug, each
    time bucket a group). Label: count (distribution). How much a
    hero's typical card choices shift across time buckets — flagged
    as dependent on the unverified slug-date parsing above.
29. **Archetype Discovery via Group Clustering** — Input: multi-group
    (the full corpus of one hero's decks, clustered by their group-
    level card sets). Label: classification (variable-set, over
    discovered clusters, not a fixed label set). An unsupervised
    framing rather than a strict prediction target — useful for
    surfacing candidate archetype labels this source doesn't provide
    directly.
30. **Group-Level Novelty Detection** — Input: multi-group (a
    candidate deck's 4 groups vs. the historical distribution of
    group contents for that hero). Label: probability. `P(this deck's
    combination of groups has not been seen before in the corpus |
    hero's historical decks)` — a "how unusual is this build" signal,
    useful for filtering near-duplicate decks out of any training set
    built from this source.
