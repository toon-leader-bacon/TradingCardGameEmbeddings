"""Regression metric: a dominiontabs card's coin cost.

See src/data_refinement/metrics/generic/masked_field_regression_metric.py
for the shared scan() sequence this fixes MASKED_FIELD/eligibility for,
and ./BRAINSTORM.md item 1 for the full design.

Originally built as a classification metric (bucketing every observed
digit cost into its own class, OTHER for anything else), then rebuilt
as regression after sanity-checking the live corpus: OTHER alone held
195/819 (23.8%) of cards, and even among the digit-cost classes,
classification treats a cost-1 and a cost-8 card as equally "different"
- discarding the ordinal information that makes cost worth predicting
in the first place.

cost is kept as dominiontabs' own raw string at ingestion time (see
../../card_binder/dominiontabs/ingestion_stage.py). Confirmed against
the full live corpus (data/raw/dominiontabs/cards_db.json), every
non-plain-digit cost is exactly one of two shapes - never anything
else:

  - "<digits><*|+>" (52 cards: 46 "*", 6 "+") - a REAL printed number
    with a rules footnote ("*" = variable/not directly Supply-buyable,
    e.g. Prizes, Travellers, Night cards; "+" = a printed minimum, the
    true cost may be higher depending on game state, e.g. Doctor/
    Herald/Masterpiece/Stonemason). The digits are recovered via
    _COST_WITH_SUFFIX_PATTERN and used as this card's regression value
    - the suffix is a rules caveat, not evidence the number is fake.
  - "" (143 cards) - genuinely no printed coin cost (Landmark/Way/
    Boon/Hex/State/Trait/Project - these aren't Supply piles at all).
    TWO cards ("Transmute", "Vineyard") are a special case within this
    group: both have a real printed coin-cost of 0 (their full cost is
    "0 Coins + 1 Potion" - dominiontabs renders the 0 as blank when a
    Potion cost is also present, confirmed by both being exactly the
    only cost=="" rows that also carry "potcost"), so those two ARE
    still eligible, unlike every other cost=="" card.

_GROUP_SUMMARY_CARD_TAGS carves out 12 of those 52 "*" cards that
aren't real individual cards at all: dominiontabs' own cards_db.json
also emits one row per divider-tab GROUP (e.g. "Augurs" represents the
Herald/Soothsayer/Lurker/Sibyl family sharing one physical tab, not a
card actually named "Augurs"). Confirmed via the RAW cards_db.json
fields (not visible in raw_content - excluded at ingestion time):
every one of these 12 has "count": "0" (a real card always has either
no count override or a nonzero one) and "group_top": true. Since
raw_content alone can't distinguish a group-summary row from a real
card, this is a deliberate hardcoded, confirmed-exhaustive name list
rather than a structural check - see this metric's own module for the
tradeoff (a proper fix would exclude these 12 at ingestion time, which
would also clean up TypeMaskMetric/SetMaskMetric for free; scoped out
of this metric alone for now).

Net eligible count: 624 (plain digit) + 40 (recoverable "*"/"+", the 52
minus the 12 group-summary rows) + 2 (Transmute/Vineyard) = 666/819
(81.3%), up from the original 624/819 (76.2%).
"""

import re
from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.generic.masked_field_regression_metric import (
    MaskedFieldRegressionMetric,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_COST_WITH_SUFFIX_PATTERN = re.compile(r"^(\d+)[*+]$")


class CostRegressionMetric(MaskedFieldRegressionMetric):
    """Card -> coin cost, masked, as a regression target. See module
    docstring for exactly which cards are eligible and why."""

    SOURCE_GAME: ClassVar[GameId] = GameId.DOMINION
    MASKED_FIELD: ClassVar[list[str]] = ["cost"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/dominiontabs/cost_regression.parquet"
    )

    # Confirmed-exhaustive against the live corpus - see module
    # docstring for how these were identified (count=="0" + group_top
    # in the RAW cards_db.json, not visible in raw_content).
    _GROUP_SUMMARY_CARD_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            "Augurs",
            "Clashes",
            "Forts",
            "Odysseys",
            "Townsfolk",
            "Wizards",
            "Catapult - Rocks",
            "Encampment - Plunder",
            "Gladiator - Fortune",
            "Patrician - Emporium",
            "Settlers - Bustling Village",
            "Sauna - Avanto",
        }
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        if card.raw_content["name"] in self._GROUP_SUMMARY_CARD_TAGS:
            return False

        cost = self._raw_field_value(card)
        if cost == "":
            # Only Transmute/Vineyard: a real coin-cost of 0, rendered
            # blank because a potcost is also present - see module
            # docstring.
            return "potcost" in card.raw_content
        return cost.isdigit() or bool(_COST_WITH_SUFFIX_PATTERN.match(cost))

    def _value_for_card(self, card: GenericCard) -> float:
        cost = self._raw_field_value(card)
        if cost == "":
            return 0.0
        match = _COST_WITH_SUFFIX_PATTERN.match(cost)
        return float(match.group(1)) if match else float(cost)
