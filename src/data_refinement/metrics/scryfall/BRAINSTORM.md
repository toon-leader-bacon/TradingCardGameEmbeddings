# BRAINSTORM (scryfall / Magic: the Gathering card data)

A todo/idea list of candidate metrics for Magic: the Gathering's card
pool, produced by scanning the live 38,626-row
`data/raw/scryfall/oracle-cards-20260820090157.jsonl` bulk dump (one
row per unique Oracle card, not per printing — see
`src/data_retrieval/scryfall/downloader.py`). None of this is designed
or built yet — this is raw material for picking what to build next,
not a spec.

Each metric below states:

- **Input** — single-card / multi-card / multi-group.
- **Label** — count / probability / classification (fixed-set) /
  classification (variable-set) / regression-continuous / ranking.
- A one-sentence description, with formal `P(...)` notation where the
  metric is probability-flavored.

## What the raw data actually looks like

Confirmed against the live Oracle Cards bulk file's first record and
its full key set (`src/data_retrieval/scryfall/example.jsonl`):

- **Color identity is already a first-class, normalized field here**
  — `colors` and `color_identity` are both plain arrays of WUBRG
  letters, present directly on every row. This resolves the open
  "is color a normalized key in MTG's `raw_content`" dependency the
  `seventeenlands` brainstorm flags as unverified: once 17lands' cards
  are cross-referenced against this project's own scryfall-backed
  `CardBinder`, any 17lands metric needing color identity can join it
  in from here rather than treating it as unresolved.
- `mana_cost` (a string like `{3}{G}`), `cmc` (float mana value),
  `type_line` (e.g. `Legendary Creature — Elf Druid`, a supertype/
  type/subtype string), `oracle_text` (full rules text), `power`/
  `toughness` (creature stats, as strings — includes non-numeric
  values like `*` for variable P/T, needs explicit handling), `keywords`
  (a structured array, e.g. `["Landfall"]` when Scryfall recognizes a
  named keyword ability in the text — a real structured-keyword field,
  no regex/sourcing dependency the way cardvault_fabtcg's equivalent
  idea has).
- `legalities`: a per-format legal/not_legal/banned/restricted map
  across every constructed/limited format Scryfall tracks — richer
  and more numerous formats than `pokemon_tcg`'s three-format
  equivalent.
- `rarity`, `set`/`set_name`, `released_at`, `reprint` (bool),
  `layout` (e.g. `normal`, `transform`, `split` — multi-face card
  shape), `edhrec_rank` (a popularity ranking specific to Commander
  format, a genuine external ranking signal no other field in this
  project provides), `prices` (market price snapshot per finish/
  currency), `game_changer` (a Scryfall-curated bool flag for
  Commander-format-impactful cards), `all_parts` (links to other
  faces/tokens/meld pieces a card references).
- `finishes`/`foil`/`nonfoil`/`promo`/`full_art`/`textless`/`border_color`/
  `frame`/`frame_effects`/`security_stamp`/`watermark`: a large set of
  print-cosmetic fields distinct from gameplay fields — mirrors
  cardvault_fabtcg's print-vs-gameplay-field distinction but with far
  more granularity.
- This is a **bulk "Oracle Cards" dump**: one row per unique card by
  Oracle identity, not per printing — reprint/rarity-drift metrics
  (like cardvault_fabtcg's #21) would need a different Scryfall bulk
  file ("Default Cards" or "All Cards") not currently downloaded here.

## Known biases / known-unknowns

- **Card data only in this container** — no match/draft/deck outcome
  data lives here; that's `seventeenlands`' role for this same game.
  Every metric below is card-intrinsic or card-relationship, joining
  outward to `seventeenlands` only where explicitly noted.
- **This dump reflects one point-in-time snapshot** (the filename's
  own timestamp) — like `spire_codex`/`gwent_one`, a single overwritten
  pull with no built-in history; power-creep/banlist-history metrics
  would need deliberately archived successive pulls.
- **`edhrec_rank`/`prices` are external, non-gameplay signals** —
  useful as auxiliary labels but shouldn't be treated as ground truth
  about card power level; price reflects market scarcity/collector
  demand as much as playability.
- **Oracle-level dedup means power/toughness "as printed on the
  original card" nuances for cards later errata'd are invisible here**
  — this file always reflects Scryfall's current Oracle ruling, not
  a card's original printed text.

## Candidate Metrics

### Single-card (15)

1. **Mana Value (CMC) Prediction** — Input: single-card. Label: count.
   Predict `cmc` from `oracle_text` + `type_line` + power/toughness —
   the MTG analogue of every other source's cost-prediction "power
   level" proxy.
2. **Color Identity Prediction** — Input: single-card. Label:
   classification (variable-set, multi-label over WUBRG). Predict
   `color_identity` from `oracle_text` + `type_line`, mana symbols
   masked out of the input.
3. **Type Line Prediction** — Input: single-card. Label:
   classification (variable-set, multi-label over supertypes/types/
   subtypes). Predict `type_line` from `oracle_text` alone.
4. **Power/Toughness Prediction** — Input: single-card. Label: count
   (two values, `*`/variable cases excluded or flagged separately).
   For creatures, predict `power`/`toughness` from `oracle_text` +
   `cmc` + `colors`.
5. **Keyword Ability Detection** — Input: single-card. Label:
   classification (variable-set, multi-label). Predict `keywords`
   from the rest of `oracle_text` — structured ground truth already
   exists (Scryfall's own keyword extraction), no sourcing dependency.
6. **Rarity Prediction** — Input: single-card. Label: classification
   (fixed-set: common/uncommon/rare/mythic). Predict `rarity` from
   `oracle_text` + `type_line` + `cmc`.
7. **EDHREC Rank Prediction** — Input: single-card. Label: regression-
   continuous (or ranking). Predict `edhrec_rank` from `oracle_text` +
   `type_line` + `color_identity` — a genuine external popularity
   signal to predict, distinct from any metric this project computes
   itself.
8. **Game-Changer Flag Prediction** — Input: single-card. Label:
   probability. `P(game_changer | oracle_text, type_line, cmc)` —
   whether Scryfall's own curated "format-warping card" judgment is
   inferable from card text alone.
9. **Layout/Multi-Face Prediction** — Input: single-card. Label:
   classification (fixed-set: normal/transform/split/adventure/etc.).
   Predict `layout` from `oracle_text` + `type_line` — e.g. text
   containing "Transform" strongly implies a `transform` layout.
10. **Format Legality Prediction** — Input: single-card. Label:
    classification (variable-set, over `legalities`' format keys).
    Predict which formats a card is legal in from `set`/`released_at`/
    `rarity` — a rotation/banlist-inference task, richer than
    `pokemon_tcg`'s 3-format version given Scryfall's larger format
    list.
11. **Price Regression** — Input: single-card. Label: regression-
    continuous. Predict a card's `prices` (e.g. USD nonfoil) from
    `rarity` + `set` + `edhrec_rank` — a market-value proxy, flagged
    as the weakest/most speculative single-card idea (mirrors
    cardvault_fabtcg's flavor-text-presence caveat: a market artifact,
    not a gameplay signal).
12. **Reserved List Membership Prediction** — Input: single-card.
    Label: probability. `P(reserved | set, released_at, rarity)` —
    the Reserved List is a fixed historical policy list, not
    text-inferable in principle, included as a known-unlearnable
    sanity-check baseline (a model should NOT be able to solve this
    from card text, since the label has no textual basis).
13. **Full-Art/Textless Frame Prediction** — Input: single-card.
    Label: probability. `P(full_art | set, rarity)` — a print-
    cosmetic prediction task, weakest/most speculative of this
    section, included for completeness only.
14. **Watermark Presence Prediction** — Input: single-card. Label:
    probability. `P(has a watermark | set, type_line)` — another
    print-cosmetic, low-priority idea.
15. **Story/Preview Spotlight Prediction** — Input: single-card.
    Label: probability. `P(story_spotlight | flavor_text presence,
    set)` — speculative, a marketing/flavor artifact rather than a
    gameplay signal.

### Multi-card (10)

16. **Color-Identity Legality Pairing** — Input: multi-card (pair).
    Label: probability. `P(pair_legal_in_same_Commander_deck |
    color_identity_1, color_identity_2)` — a deck-legality classifier
    derivable from color identity alone, comparable to
    cardvault_fabtcg's class-restriction idea.
17. **Keyword Co-Occurrence / Synergy Signal** — Input: multi-card
    (pair). Label: probability. `P(share_a_keyword | card_1, card_2)`
    — a card-text-derived synergy proxy; real deck co-occurrence data
    for this game lives in `seventeenlands` instead (see cross-source
    idea #23 below).
18. **All-Parts Relationship Prediction** — Input: multi-card (pair:
    a card and one of its `all_parts` entries — a token, meld piece,
    or other face). Label: classification (fixed-set: token/meld_part/
    combo_piece/other). Predict the relationship type between two
    linked cards.
19. **Multi-Face Pairing (front → back)** — Input: multi-card (a
    transform/modal-DFC card's two faces). Label: classification
    (variable-set). Given one face's `oracle_text`, predict the other
    face's identity — the MTG analogue of cardvault_fabtcg's
    specialization-pairing idea (#11 there).
20. **CMC-for-Ramp Sufficiency** — Input: multi-card (pair). Label:
    probability (deterministic sanity-check). `P(a ramp/mana-dork
    card_2 can cast card_1 one turn early)` — an arithmetic baseline,
    same role as other sources' pitch/cost-sufficiency ideas.
21. **EDHREC-Rank-Conditioned Pairing** — Input: multi-card (pair,
    both from similar `color_identity`). Label: probability. `P(cards
    are commonly played together | edhrec_rank proximity, shared
    color_identity)` — a coarse, rank-proxy synergy signal usable
    without real deck data.
22. **Reprint Power-Level Consistency** — Input: multi-card (pair:
    same-named card across two different bulk-file pulls over time,
    once archived). Label: none (embedding-invariance evaluation) —
    flagged as needing the archived-pulls prerequisite noted above,
    same shape as cardvault_fabtcg's reprint-invariance idea (#16
    there).
23. **Cross-Source Draft Pick Correlation (scryfall × seventeenlands)**
    — Input: multi-card (a card's Scryfall-derived features + its
    17lands-derived pick-rate metric, joined by name). Label:
    regression-continuous. Whether `cmc`/`rarity`/`keywords` predict a
    card's observed `AveragePickNumberMetric`/take-rate from
    `seventeenlands` — the concrete cross-source join the
    `seventeenlands` brainstorm's color-identity dependency was
    blocking, now unblocked by this source's own fields.
24. **Format-Legal Curve Complementarity** — Input: multi-card (pair,
    same `color_identity`). Label: probability (sanity-check).
    `P(cmc(card_1) + cmc(card_2) forms a coherent two-card curve)` —
    a weak, mostly-baseline idea, included for completeness.
25. **Type-Line Overlap Pairing** — Input: multi-card (pair). Label:
    probability. `P(share a creature type or subtype | card_1,
    card_2)` — tribal-synergy detection, comparable to `pokemon_tcg`'s
    evolution-chain idea but for MTG's looser tribal-typal structure.

### Multi-group / corpus-level (5)

26. **Color Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Count of cards per
    `color_identity` combination — a dataset-balance stat before
    training #2.
27. **CMC Curve by Rarity** — Input: multi-group (whole corpus, split
    by rarity). Label: count (distribution). Whether higher-rarity
    cards skew toward higher/more variable `cmc` — a sanity check for
    #1/#6.
28. **Keyword Representation Balance** — Input: multi-group (whole
    corpus). Label: count (distribution). Frequency of each Scryfall-
    recognized keyword across the corpus — needed before training #5.
29. **Format Legality Coverage Over Time** — Input: multi-group (whole
    corpus, grouped by `released_at` era). Label: count (distribution).
    How much of each era's card pool remains legal in each modern
    format — a rotation/banlist-history study, informing #10.
30. **EDHREC Rank Distribution by Color Identity** — Input:
    multi-group (whole corpus, split by `color_identity`). Label:
    count (distribution). Whether certain color combinations skew
    toward better (lower) `edhrec_rank` on average — a Commander-
    format color-power-level sanity check, informing #7/#21.
