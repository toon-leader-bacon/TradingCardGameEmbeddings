# BRAINSTORM (cardvault_fabtcg / Flesh and Blood)

A todo/idea list of candidate metrics for Flesh and Blood, produced by
a `pandas` scan of the real 46,660-row
`data/tmp/public_card_data.csv` dump (see
`src/data_retrieval/cardvault_fabtcg/card_downloader.py` and
`src/data_refinement/card_binder/cardvault_fabtcg/ingestion_stage.py`).
None of this is designed or built yet — this is raw material for
picking what to build next, not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

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
  heroes).
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
  "Arakni, Funnel Web" / ... — 10 confirmed for one hero alone).
- `face_1_artist` has 335 distinct values across the corpus (Carlos
  Cruchaga alone credited on 549 rows) — high cardinality, would need
  support-count filtering before use as a target.

## Known biases / known-unknowns

- **Card data only, no play data** — no match results, decklists, or
  draft logs exist for Flesh and Blood in this project yet, so every
  metric here is card-intrinsic or card-relationship, never outcome-
  conditioned. This is the main structural gap vs. `sts_gg`/`sts2runs`/
  17lands.
- **Keyword ability list not sourced** — several ideas below need a
  canonical list of named keyword abilities (e.g. "Go again,"
  "Dominate," "Ward," "Blade Break") extracted from `rules_text`. That
  list isn't derived from this project's own data anywhere yet and
  would need sourcing/verification first.
- **Print-level fields (rarity drift, print count) sit one layer below
  `card_binder`'s merged/deduplicated view** — those metrics need the
  raw per-printing rows, not the canonical `GenericCard`.
- **Artist/flavor-text fields are print-flavor metadata, not gameplay
  signal** — any metric built on them should be understood as weaker/
  more speculative than rules-text-driven ones.

## Candidate Metrics

### Single-card (14)

1. **Class Prediction from Rules Text** — Input: single-card. Label:
   classification (variable-set, multi-label). Predict `face_1_classes`
   from the rest of the card (rules text, cost, power/defense, type),
   with the class field itself masked.
2. **Talent Prediction from Rules Text** — Input: single-card. Label:
   classification (variable-set, multi-label). Same idea for
   `face_1_talents`; heavily imbalanced (mostly empty), flag before
   training.
3. **Pitch Value Prediction** — Input: single-card. Label:
   classification (fixed-set, {1,2,3}). Predict `face_1_true_pitch`
   from rules text + cost + power/defense, excluding color (highly
   correlated with pitch).
4. **Cost Prediction ("power level" proxy)** — Input: single-card.
   Label: count. Predict `face_1_true_cost` from rules text + power/
   defense alone.
5. **Power/Defense Prediction** — Input: single-card. Label: count
   (two values). For Attack-shaped cards, predict `face_1_true_power`/
   `face_1_true_defense` from rules text + cost.
6. **Type/Subtype Prediction** — Input: single-card. Label:
   classification (variable-set, multi-label). Predict `face_1_types`/
   `face_1_subtypes` from rules text alone.
7. **Equipment Slot Prediction** — Input: single-card. Label:
   classification (fixed-set: Head/Arms/Chest/Legs/Item). For
   Equipment-typed cards, predict body slot from rules text alone.
8. **Keyword Ability Detection** — Input: single-card. Label:
   classification (variable-set, multi-label). Detect named keyword
   abilities present in `rules_text`; blocked on sourcing the keyword
   list.
9. **Hero Life/Intellect Prediction** — Input: single-card. Label:
   count (two values). For Hero-typed cards, predict
   `face_1_true_life`/`face_1_true_intellect` from the hero's rules
   text. Low-data (160 unique heroes).
10. **Flavor Text Presence Prediction** — Input: single-card. Label:
    probability. `P(has_flavor_text | rarity, type, rules_text)` —
    base rate 13.8%; weakest/most speculative, a print-process
    artifact more than a gameplay signal.
11. **Specialization Face Prediction** — Input: single-card (two-faced
    card's own two faces, still one `nocab_uuid`). Label:
    classification (variable-set). Given `face_1`'s name/rules text,
    predict `face_2_true_name`/rules-text identity. Scoped to the
    489/46,660 DFC rows.
12. **Print Count per Card (Reprint Frequency)** — Input: single-card.
    Label: count. Number of distinct `set_code`s a `card_id` has
    appeared under, resolved via `AliasLedger` print aliases.
13. **Rarity-Across-Reprints Drift** — Input: single-card. Label:
    classification (fixed-set: up/down/none). For the 1,401/5,046
    `card_id`s with more than one observed rarity, predict drift
    direction between original and most recent printing.
14. **Language Coverage Completeness** — Input: single-card. Label:
    count. How many of the 9 `print_language` values a `card_id` has
    ever been printed in — a popularity/print-run-size proxy.

### Multi-card (9)

15. **Class-Restriction Deck Legality (pairwise)** — Input: multi-card
    (pair). Label: probability. `P(pair_is_legal_together |
    class_1, talent_1, class_2, talent_2)` — derivable from card text
    alone; ~12.7M unordered pairs across 5,046 cards, so needs
    negative-sampling rather than exhaustive enumeration.
16. **Talent Co-Occurrence / Synergy Signal** — Input: multi-card
    (pair). Label: probability. `P(share_a_talent | card_1, card_2)`
    — the FaB analogue of a co-occurrence signal, derived from card
    text since no real deck data exists for this source.
17. **Hero–Equipment Class Fit** — Input: multi-card (Hero + item
    pair). Label: probability. `P(equip_legal | hero_classes,
    hero_talents, item_restriction_text)` — bounded to 160 heroes ×
    ~2,360 Equipment/Weapon cards.
18. **Family/Cycle Grouping (name-prefix)** — Input: multi-card (pair).
    Label: probability. `P(same_family | name_1, name_2)` — cheaply
    derivable via string match on the comma-delimited name prefix;
    more useful as a free auxiliary label/filter for #15-#17 than a
    metric worth its own dojo.
19. **Reprint/Print-Invariance Check** — Input: multi-card (pair, same
    underlying `nocab_uuid`, different printings). Label: none
    (embedding-similarity evaluation, not a supervised target) — flag
    as an eval metric for `SingleCardModel`, not a training label.
20. **Equipment Slot Conflict** — Input: multi-card (pair, Equipment-
    only subset). Label: probability. `P(same_slot | equip_1_slot,
    equip_2_slot)` — reuses metric #7's slot label as input.
21. **Pitch-for-Cost Sufficiency** — Input: multi-card (pair). Label:
    probability (deterministic sanity-check, not a research target).
    `P(pitch_value(card_2) >= cost(card_1))` — a trivially-solvable
    baseline any real model should nail.
22. **Class Representation in a Hypothetical Pool** — Input: multi-card
    (small synthetic group). Label: classification (fixed-set, 11
    classes + Generic). Predict the plurality class across a
    synthetically-sampled pool — a stand-in for "predict archetype
    from partial pool" until real decklist data exists. Deprioritize
    until draft/deck data for this game exists.
23. **Print-Family Count** — Input: multi-card (all printings of one
    `card_id`, group). Label: count. Total number of distinct
    `print_id`s across all languages/sets for one card — a broader
    version of #12/#14 combined into one group-level stat.

### Multi-group / corpus-level (7)

24. **Class Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Count of cards per class
    across the whole pool — a dataset-balance stat any single-card
    class metric (#1 especially) needs before training.
25. **Talent Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Same as #24 for talents;
    more severe imbalance (12,292/17,161 rows have no talent).
26. **Rules Text Length Distribution** — Input: multi-group (whole
    corpus, split by rarity/type). Label: count (distribution).
    Word/character-count distribution of `rules_text` — validates
    whether #1/#2/#6/#8's "predict from rules text" tasks have enough
    textual signal, especially for the ~1.2% of empty-rules-text rows.
27. **Hero Family Size Distribution** — Input: multi-group (all
    specializations grouped by hero name-prefix). Label: count
    (distribution). Number of specializations per named hero — a
    dataset-balance check for any per-hero metric (#9, #11).
28. **Class-Pair Co-Legality Matrix** — Input: multi-group (all class
    pairs). Label: probability (matrix). Corpus-wide
    `P(any_card_of_class_A legal with any_card_of_class_B)` summary
    table — a precomputed lookup other pairwise metrics (#15-#17)
    could sanity-check predictions against.
29. **Equipment Slot Coverage** — Input: multi-group (Equipment
    subset). Label: count (distribution). Count of Equipment cards
    per body slot — a balance check for #7/#20.
30. **Rarity Distribution by Type** — Input: multi-group (whole corpus,
    split by `face_1_types`). Label: count (distribution). Whether
    certain card types (e.g. Hero) skew toward rarer print rarities
    than others — a sanity/stratification check before any rarity-
    conditioned metric.
