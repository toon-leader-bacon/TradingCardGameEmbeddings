# METRIC_BRAINSTORM (cardvault_fabtcg / Flesh and Blood)

A todo/idea list of candidate metrics for Flesh and Blood, produced by
a `pandas` scan of the real 46,660-row
`data/tmp/public_card_data.csv` dump (see
`src/data_retrieval/cardvault_fabtcg/card_downloader.py` and
`src/data_refinement/card_binder/cardvault_fabtcg/ingestion_stage.py`).
None of this is designed or built yet — this is raw material for
picking what to build next, not a spec. Every metric names its:

- **Card shape** — single card / multi card / multi group, per
  `card_binder/README.md`'s taxonomy.
- **Label** — what a dojo would eventually try to predict from this
  metric's output.
- **Builder shape** — see the note below; this container has no
  `CsvScanner`-equivalent infrastructure yet.

## What's structurally different about this source

17lands and sts_gg both have game/draft/run *logs* — rows of actual
play sessions a metric can accumulate win rates, pick rates, or
sequences from. **cardvault.fabtcg.com's dump is card data only** — no
match results, no decklists, no draft logs exist for Flesh and Blood
in this project yet. Every idea below is therefore either:

- a **card-intrinsic** task, derivable from a single card's own
  `raw_content` fields (en `print_language` row's true_* stats plus
  rules/flavor text) — useful as an auxiliary/pretraining signal for
  `SingleCardModel` (`plans/encoder_model.md`), the same role
  masked-field prediction plays for text encoders, or
- a **card-relationship** task, over pairs/groups of already-ingested
  `GenericCard`s pulled from `CardBinder` — no raw CSV needed once
  `card_binder/cardvault_fabtcg/ingestion_stage.py` has run, or
- a **corpus-level** stat, a single pass over `CardBinder.all_cards()`
  producing one summary rather than a per-card/per-pair label —
  useful for stratified sampling or dataset-balance decisions in a
  training pipeline, not itself a prediction target.

None of these three shapes needs a chunked-CSV builder the way
17lands' multi-GB files do (this dump is 32MB, safe to hold in
memory) — "Builder shape" below just names which of these three a
metric is, not an accumulation/streaming distinction.

## What the raw data actually looks like (found via this review)

Confirmed against the full en-`print_language` corpus (5,046 unique
`card_id`s):

- `face_1_true_color`/`face_1_true_pitch`: red=1, yellow=2, blue=3
  (near-perfectly correlated — a handful of colorless/`purple`
  exceptions exist), present on ~74% of cards (~26% have no pitch —
  Hero/Equipment/Weapon/Token cards are colorless).
- `face_1_classes`: 2,882 `Generic` (any class), the rest single- or
  multi-class (`;`-joined, e.g. `"Necromancer;Pirate"`), 11 named
  classes total observed (Warrior, Guardian, Runeblade,
  Mechanologist, Brute, Ninja, Illusionist, Wizard, Ranger, Assassin,
  Pirate).
- `face_1_talents`: mostly empty (12,292/17,161 non-Generic rows);
  when present, one of 8 named talents (Shadow, Light, Lightning,
  Elemental, Draconic, Mystic, Earth, Ice) plus two "faction" talents
  (Revered, Reviled) and one wildcard (Chaos) — `;`-joined when a card
  spans two.
- `face_1_types`: dominated by `Action` (11,261) and `Equipment`
  (1,835), with `Instant`/`Attack Reaction`/`Hero`/`Defense
  Reaction`/`Weapon`/`Token`/`Block`/`Resource`/`Demi-Hero`/`Event`
  also observed, some `;`-joined (e.g. `"Action;Equipment"`).
- `face_1_subtypes`: includes both attack-shape subtypes (`Attack`,
  `Arrow;Attack`) and Equipment slot subtypes (`Head`, `Arms`,
  `Chest`, `Legs`, `Item`) plus `Aura`, `Young` (specialization-eligible
  heroes) — confirms Equipment cards carry an explicit body-slot
  subtype worth its own metric (see below).
- `face_1_true_cost`: mode is 0 (5,467 rows — mostly Attack Reactions/
  Instants/free effects), otherwise 1-9, heavily right-skewed.
- Hero cards (`face_1_types` contains `"Hero"`, 160 unique names)
  carry `face_1_true_life`/`face_1_true_intellect` — the two fields no
  non-Hero card populates.
- `face_1_rules_text` is populated on 98.8% of en rows; flavor text on
  only 13.8% — rules text is a much more reliable model input than
  flavor text.
- Many Hero names repeat a shared prefix across their specializations
  (e.g. "Arakni, 5L!p3d 7hRu 7h3 cR4X" / "Arakni, Black Widow" /
  "Arakni, Funnel Web" / ... — 10 confirmed for one hero alone) — a
  real, exploitable naming convention for the family-grouping idea
  below, not a guess.
- `face_1_artist` has 335 distinct values across the corpus (Carlos
  Cruchaga alone credited on 549 rows) — high enough cardinality that
  an artist-conditioned metric would need real support-count filtering
  before use.

**Open dependency flagged, not resolved:** several ideas below need a
canonical list of Flesh and Blood's named keyword abilities (e.g. "Go
again," "Dominate," "Ward," "Blade Break") to extract as structured
labels from `rules_text`. That list isn't derived from this project's
own data anywhere yet — it would need to be sourced and verified
against real `rules_text` samples before any keyword-extraction metric
below could be built, the same kind of unresolved prerequisite the
17lands brainstorm flags for color identity.

---

## Card-Intrinsic (10)

Single card, label derivable from that same card's own `raw_content` —
i.e. self-supervised "reconstruct a masked field" tasks in the same
spirit as BERT-style pretraining, not tasks needing outside game
outcomes.

1. **Class Prediction from Rules Text** — predict `face_1_classes`
   (masked out) from the rest of the card's `raw_content` (rules text,
   cost, power/defense, type). Single card. Label: a
   multi-label class set (11 named classes + Generic). Corpus-scan
   (one pass over `CardBinder.all_cards(GameId.FLESH_AND_BLOOD)`, no
   cross-card state).

2. **Talent Prediction from Rules Text** — same idea, for
   `face_1_talents` (8 named talents + Revered/Reviled/Chaos). Single
   card. Label: multi-label talent set (mostly empty — a genuinely
   imbalanced target, worth flagging before committing). Corpus-scan.

3. **Pitch Value Prediction** — predict `face_1_true_pitch` (1/2/3)
   from rules text + cost + power/defense, excluding the (highly
   correlated) color field itself. Single card. Label: an int in
   {1,2,3} (colorless cards excluded — no pitch to predict). Corpus-scan.

4. **Cost Prediction ("power level" proxy)** — predict `face_1_true_cost`
   from rules text + power/defense alone — a model that can guess a
   card's cost from its effect text is capturing real game-balance
   semantics. Single card. Label: int cost. Corpus-scan.

5. **Power/Defense Prediction** — for cards where `face_1_types`
   contains `Attack`-shaped subtypes, predict `face_1_true_power`/
   `face_1_true_defense` from rules text + cost — the FaB analogue of
   predicting MTG power/toughness from oracle text. Single card.
   Label: two ints. Corpus-scan.

6. **Type/Subtype Prediction** — predict `face_1_types`/
   `face_1_subtypes` from rules text alone. Single card. Label:
   multi-label type/subtype set. Corpus-scan.

7. **Equipment Slot Prediction** — narrower version of #6, restricted
   to `Equipment`-typed cards: predict which body slot
   (`Head`/`Arms`/`Chest`/`Legs`/off-hand `Item`) a card occupies from
   its rules text alone (slot is usually implicit in flavor/name, not
   always spelled out in rules text — worth checking how often it's
   inferable at all before committing). Single card. Label: a slot
   class. Corpus-scan.

8. **Keyword Ability Detection** — presence/absence of each named
   keyword ability (Go again, Dominate, Ward N, Blade Break, etc. —
   see the open keyword-list dependency above) as a multi-label
   target extracted by regex/string match over `rules_text`. Single
   card. Label: multi-label keyword set. Corpus-scan, but blocked on
   sourcing the keyword list first.

9. **Hero Life/Intellect Prediction** — for Hero-typed cards only,
   predict `face_1_true_life`/`face_1_true_intellect` from the hero's
   rules text (a Hero's ability text often trades off against its
   stat line — e.g. an aggressive hero tends to run lower life).
   Single card. Label: two ints. Corpus-scan, small population (160
   unique heroes) — flag as low-data.

10. **Flavor Text Presence Prediction** — predict whether a card has
    flavor text at all (13.8% base rate) from its rarity/type/rules
    text. Single card. Label: bool. Weakest/most speculative of the
    ten — a card-design-process artifact more than a gameplay signal,
    included for completeness rather than as a priority pick.
    Corpus-scan.

---

## Card-Relationship (9)

Multi card (pairs, or a card plus a named group), over already-
ingested `GenericCard`s — no raw CSV access needed once ingestion has
run.

11. **Specialization Pairing (face_1 → face_2)** — given a two-faced
    card's front face (`face_1_true_name`/rules text), predict the
    flip side's `face_2_true_name`/rules text — this project's own
    ingestion already keys these together (`"{face_1} // {face_2}"`
    naming, 489/46,660 rows). Multi card (a card's own two faces, so
    really still single-`nocab_uuid` but a genuinely two-part label).
    Label: face_2's identity or rules-text embedding target.
    Corpus-scan over just the DFC subset.

12. **Class-Restriction Deck Legality (pairwise)** — given two cards,
    predict whether they could ever legally share a deck, purely from
    each one's `classes`/`talents` fields (e.g. two different
    non-Generic, non-matching classes are almost always mutually
    exclusive; `Generic` is legal with anything) — a deck-building
    legality classifier derivable from card text alone, no decklist
    data needed. Multi card (pair). Label: bool. Pairwise (needs
    enumerating card pairs, not a straight corpus-scan — flag the
    combinatorial cost, 5,046 cards ⇒ ~12.7M unordered pairs, before
    committing to exhaustive enumeration over training rather than
    negative-sampling).

13. **Talent Co-Occurrence / Synergy Signal** — given two cards,
    predict whether they share a talent — the FaB analogue of the
    sts2runs brainstorm's "given two cards, predict P(both in deck |
    either in deck)" co-occurrence idea, except derived here purely
    from each card's own talent field rather than observed deck data
    (which doesn't exist for this source yet). Multi card (pair).
    Label: bool. Pairwise, same combinatorial-cost caveat as #12.

14. **Hero–Equipment Class Fit** — given a Hero card and an Equipment/
    Weapon card, predict compatibility from the Hero's
    `classes`/`talents` against any class-restriction text on the
    item (e.g. "For Warrior heroes only") — narrower and more
    directly useful than #12's general pairwise legality task, since
    it's scoped to the Hero-plus-gear relationship that actually
    matters for deckbuilding. Multi card (Hero + item). Label: bool.
    Pairwise, but bounded by 160 heroes × ~2,360 Equipment/Weapon
    cards rather than the full combinatorial pair space.

15. **Family/Cycle Grouping (name-prefix)** — given two cards, predict
    whether they belong to the same named hero/cycle family, using the
    observed real naming convention (e.g. "Arakni, X" specializations
    sharing a comma-delimited prefix). Multi card (pair). Label: bool.
    Pairwise — but the label itself is cheaply derivable via a string
    match on `face_1_true_name`, not something requiring a model to
    predict at all; flagging this mainly as a **free auxiliary label**
    other pair-tasks (#12, #13, #14) could condition on or exclude
    same-family pairs from, rather than a metric worth its own dojo.

16. **Reprint/Print-Invariance Check** — given two *raw rows* under the
    same `card_id` but different `print_id`/`set_code`/`rarity`
    (already deduplicated by `card_binder`'s own identity logic —
    this metric operates one layer below that, on the raw printings
    themselves), verify an encoder produces near-identical embeddings
    for both. Multi card (pair, but same underlying `nocab_uuid`).
    Label: none in the traditional sense — this is an embedding
    **invariance check**, not a supervised prediction target; flagging
    it as a genuinely different metric shape (an evaluation metric for
    `SingleCardModel`, not a training label) worth a naming/design
    conversation before forcing it into the Label-column convention
    the rest of this list uses.

17. **Equipment Slot Conflict** — given two Equipment cards, predict
    whether they occupy the same body slot (and thus can't both be
    equipped at once) — reuses #7's slot label as an input rather than
    re-deriving it per pair. Multi card (pair). Label: bool. Pairwise,
    but bounded to the Equipment-only subset (1,835 cards).

18. **Pitch-for-Cost Sufficiency** — given a card and a hypothetical
    "pitch card" (any card played face-down for its `true_pitch`
    value), predict whether pitching the second card fully pays the
    first's `true_cost` — a purely arithmetic relationship (no text
    understanding needed) included mainly as a **sanity-check
    baseline** a real model should trivially solve, not a research
    target. Multi card (pair). Label: bool. Pairwise.

19. **Class Representation in a Hypothetical Pool** — given a small set
    of cards (a group, not a pair), predict the plurality class
    across the group — a synthetic stand-in for "predict archetype
    from partial pool" until real decklist data exists for this
    source. Multi group. Label: a class. Flagged as the most
    speculative of this section, since without real decklists the
    "pool" has to be synthetically sampled rather than drawn from
    actual play data — worth deprioritizing until draft/deck data for
    Flesh and Blood exists.

---

## Corpus-Level (6)

A single pass over `CardBinder.all_cards(GameId.FLESH_AND_BLOOD)`
producing one summary stat, not a per-card or per-pair label — useful
for dataset-balance/stratified-sampling decisions when building any
of the dojos above, not itself a training target.

20. **Print Count per Card (Reprint Frequency)** — number of distinct
    `set_code`s a `card_id` has appeared under, resolved from
    secondary `print_id` aliases already registered in `AliasLedger`
    — a popularity/power-level proxy (more-reprinted cards are
    generally more foundational to the format). Corpus-scan.

21. **Rarity-Across-Reprints Drift** — for cards reprinted at a
    different `rarity` than their original printing (confirmed to
    happen — 1,401/5,046 `card_id`s in the corpus have more than one
    observed rarity value), track the direction of drift (rarity ↓
    Balance change, or ↑ typically means a Legendary/promo one-off).
    Corpus-scan, but needs each `card_id`'s full print history, not
    just the canonical merged card — same one-layer-below-`card_binder`
    caveat as #16.

22. **Language Coverage Completeness** — how many of the 9
    `print_language` values a `card_id` has ever been printed in —
    every card has at least one (`en`, confirmed), but coverage of the
    other 8 varies (some `card_id`s span all 6 European+Japanese
    languages sampled in this review, others fewer) — a
    popularity/print-run-size proxy independent of #20's set-count
    version. Corpus-scan.

23. **Class Representation Balance** — count of cards per class across
    the whole pool (2,882 `Generic` vs. 11 named classes at very
    different counts, per this review's findings above) — a dataset-
    balance stat any of the Card-Intrinsic dojos above (#1 especially)
    would need before training, to avoid a model that just learns to
    predict the majority class. Corpus-scan.

24. **Talent Representation Balance** — same idea as #23, for talents
    (mostly-empty field, 8 named values plus 3 special-case ones) —
    flagged as needing more care than #23 since the class imbalance
    here is more severe (12,292/17,161 rows have no talent at all).
    Corpus-scan.

25. **Rules Text Length Distribution** — word/character-count
    distribution of `rules_text` across the corpus, split by
    `rarity`/`types` — a sanity-check stat (are higher-rarity cards
    reliably wordier?) more than a training signal, useful mainly to
    validate that #1/#2/#6/#8's "predict structured fields from rules
    text" tasks have enough textual signal to work with in the first
    place, especially for the ~1.2% of rows with empty rules text.
    Corpus-scan.
