# BRAINSTORM (dominiontabs / Dominion card data)

A todo/idea list of candidate metrics for Dominion's card pool, sourced
from `data/final/cards/dominion.jsonl` (819 cards, ingested via
`DominionTabsCardIngestionStage` - see
[`../../card_binder/README.md`](../../card_binder/README.md)).

Unlike `gwent_one` (eight fields, several genuinely independent axes:
faction/color/rarity/type/set/power/armor/provision), dominiontabs is
a deliberately thin source: `raw_content` was trimmed at ingestion
time to six fields only (`name`, `types`, `cost`, `description`,
`potcost`, `debtcost` - see
`../../card_binder/dominiontabs/ingestion_stage.py`'s module docstring
for why every other raw field, e.g. `cardset_tags`/`group_tag`/
`count`/`randomizer`/`image`, was excluded). `potcost`/`debtcost` are
each present on fewer than 15/819 cards - too sparse for their own
masking metric (`_is_eligible()` would reject over 98% of the corpus).
`description` is free text, not a fixed-set classification target.
That leaves three real single-card masking candidates, which is all
this container builds:

1. **Input**: single-card. **Label**: regression-continuous.
   `E[cost | rest of card]` - mask a card's coin cost and predict it.
   Originally designed as classification (every observed plain-digit
   value its own class, `OTHER_LABEL` for the rest, same encoding as
   `gwent_one/power_mask_metric.py`) but rebuilt as regression after
   sanity-checking the live corpus: `OTHER` alone held 195/819 (23.8%)
   of cards, and classification discards the ordinal information that
   makes cost worth predicting (a cost-1 and a cost-8 card are equally
   "wrong" guesses under classification, which isn't true). `cost`
   stays dominiontabs' own raw string at ingestion time (`"6*"`,
   `"3+"`, `""` for uncosted Landmarks/Ways/etc.), so this metric is
   eligible only for the 624/819 (76.2%) cards whose cost is a plain
   digit - a variable/minimum/uncosted cost isn't well-modeled as one
   number, so those rows are excluded, not coerced into a fallback
   value. Backed by a new generic base,
   `../generic/masked_field_regression_metric.py` (mirrors
   `MaskedFieldMetric`, float label instead of a fixed-set one - no
   prior consumer of this shape existed in this project). **Built
   as**: `CostRegressionMetric`.
2. **Input**: single-card. **Label**: classification (fixed-set).
   `P(primary type | rest of card)` - mask a card's type and predict
   it. `types` is a list (e.g. `["Action", "Attack"]`, `["Action",
   "Duration"]` - 94 distinct combos across 819 cards, far too sparse
   to classify the full combo), so this predicts only the *first*
   listed type (19 distinct primary values - confirmed by direct
   sampling of `data/raw/dominiontabs/cards_db.json`), the same
   simplification `gwent_one/type_mask_metric.py` doesn't need to make
   (gwent.one's `type` is already scalar) but is the natural one here.
   **Built as**: `TypeMaskMetric`.
3. **Input**: single-card. **Label**: classification (fixed-set).
   `P(expansion | rest of card)` - mask which Dominion expansion a
   card ships in and predict it. This is the one metric in this
   container that can't read `raw_content` alone: `cardset_tags` was
   deliberately excluded from ingestion (see item list above), so this
   metric reads `data/raw/dominiontabs/cards_db.json` directly and
   joins each raw entry back to its `nocab_uuid` via
   `CardLookup.get_by_alias(GameId.DOMINION, DataSource.DOMINIONTABS,
   card_tag)` - not a `MaskedFieldMetric` subclass, since that base
   assumes its field already lives in `raw_content`. `cardset_tags` is
   also genuinely messy (1st/2nd-edition variants, "Removed"/"Upgrade"
   transition tags, `-bigbox2-de` German promo tags - confirmed by
   full-corpus sampling), so this metric canonicalizes each tag first
   (stripping edition/removal/promo noise) and excludes the 29/819
   cards (mostly basics - Copper/Silver/Gold/Estate/Duchy/Province/
   Curse/Colony/Platinum/Trash/Start Deck - plus a handful of Guilds
   cards also reprinted in Cornucopia's 2nd-edition box) whose
   canonicalized tag set still names more than one expansion - those
   cards don't have one true "home expansion" to predict, so forcing a
   single label would be a fabricated ground truth, not a real one.
   `LABEL_VALUES` has 17 real classes, not 19 - "base" and "guilds" are
   valid canonicalization targets but, confirmed against the live
   corpus, can never survive as a card's SOLE canonical label (every
   base-tagged card also canonicalizes to a numbered expansion; every
   guilds-tagged card also canonicalizes to cornucopia or promo), so
   both were dropped from the classification vocabulary itself.
   **Built as**: `SetMaskMetric`.

## Known biases / known-unknowns

- Every metric here is card-intrinsic - no deck, draft, or game-outcome
  data exists for Dominion in this project (dominion.games' scraper is
  paused per the server owner's own request - see
  `src/data_retrieval/dominion/todo.md` - and isotropic.org's archived
  game logs are almost entirely gone from the Wayback Machine - see
  `src/data_retrieval/isotropic/downloader.py`). No multi-card metric
  is possible from this source today.
- `SetMaskMetric`'s excluded 29/819 cards are not random noise - they
  are exactly Dominion's cross-expansion basics (every game includes
  Copper/Silver/Gold/Estate/Duchy/Province/Curse regardless of
  kingdom) plus a few Guilds/Cornucopia box-reprint cards. A consumer
  of this metric's output should not assume "819 cards in, 819 rows
  out" - the row count is 790.
