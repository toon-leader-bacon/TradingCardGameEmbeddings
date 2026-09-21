"""Concrete metric: BRAINSTORM.md multi-card #19 - given the rest of a
finished winning deck, guess the one card masked out of it.

Built on ../../generic/deck_card_mask_metric.py's DeckCardMaskMetric
Template Method - this project's second consumer of that base (after
play_gwent's LeaderMaskedFromDeckMetric), fixing the three
game-specific hooks it requires (_deck_uuid_for_row, which card
_target_card_uuid_for_row masks out, and what _label_for_card returns).

MASKING RULE, DECIDED HERE (was an open design question at skeleton
time): _choose_masked_card_name() picks uniformly at random among the
winner's NON-BASIC cards (_BASIC_CARD_NAMES excluded - the same "every
game includes these regardless of kingdom" set BRAINSTORM.md's
SetMaskMetric discussion names) that actually appear in the winning
deck. Chosen over "the card with the most copies" (would make the
masking target trivially predictable from copy-count alone, defeating
the point of a masking task) and "a fixed card type" (Dominion card
types aren't uniform enough across the whole card pool to fix one
type ahead of time the way MaskedFieldMetric's single-field masking
can). Basics are excluded because masking "Copper" or "Estate" - present
in nearly every deck, in similar proportions - carries far less signal
than masking a kingdom card the winner specifically chose to buy.

LABEL_VALUES: every dominiontabs card name reachable as a masking
target under the rule above - computed once at class-definition time
by reading the already-ingested data/final/cards/dominion.jsonl
directly (see _default_label_values()), the same "read the ingested
card file once" approach BRAINSTORM.md's SetMaskMetric uses for its own
cardset_tags lookup, just against the final ingested file rather than
dominiontabs' raw one (no field this class needs was stripped at
ingestion time, unlike SetMaskMetric's cardset_tags).
"""

import json
import logging
import random
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.generic.deck_card_mask_metric import (
    DeckCardMaskMetric,
)
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    deck_for_player,
    winner_entry,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_DOMINION_CARDS_PATH = Path("data/final/cards/dominion.jsonl")

# Every game includes these regardless of kingdom - see module
# docstring's MASKING RULE section.
_BASIC_CARD_NAMES: frozenset[str] = frozenset(
    {
        "Copper",
        "Silver",
        "Gold",
        "Platinum",
        "Potion",
        "Estate",
        "Duchy",
        "Province",
        "Colony",
        "Curse",
    }
)


def _default_label_values() -> tuple[str, ...]:
    """Every non-basic dominiontabs card name, read once from the
    already-ingested card file.

    Module-level, not a method - called exactly once, at class-body
    evaluation time, to populate WinningDeckMaskedCardMetric.LABEL_VALUES.

    Inputs: none (reads _DOMINION_CARDS_PATH).
    Output: every "name" field in _DOMINION_CARDS_PATH's rows that
        isn't in _BASIC_CARD_NAMES, sorted for a stable, diffable
        ClassVar - or () if _DOMINION_CARDS_PATH doesn't exist yet
        (dominiontabs not ingested in this environment), so importing
        this module never hard-fails a fresh checkout.
    Side effects: emits one logging.warning() if _DOMINION_CARDS_PATH
        is missing; otherwise none (a single read of an already-final
        file).
    Exceptions: whatever json.loads() raises on a malformed line.
    """
    if not _DOMINION_CARDS_PATH.exists():
        _logger.warning(
            "WinningDeckMaskedCardMetric: %s not found - LABEL_VALUES left "
            "empty until dominiontabs cards are ingested",
            _DOMINION_CARDS_PATH,
        )
        return ()

    names: set[str] = set()
    with open(_DOMINION_CARDS_PATH, "r", encoding="utf-8") as cards_file:
        for line in cards_file:
            if not line.strip():
                continue
            card = json.loads(line)
            if card["name"] not in _BASIC_CARD_NAMES:
                names.add(card["name"])
    return tuple(sorted(names))


class WinningDeckMaskedCardMetric(DeckCardMaskMetric):
    """Winner's final deck, one non-basic card masked out -> that
    card's name.

    SOURCE_GAME/LABEL_VALUES fixed per DeckCardMaskMetric's contract -
    see module docstring for both the masking rule and how
    LABEL_VALUES is populated.
    """

    SOURCE_GAME = GameId.DOMINION
    LABEL_VALUES: ClassVar[tuple[str, ...]] = _default_label_values()
    DEFAULT_OUTPUT_PATH = Path(
        "data/metrics/isotropic/winning_deck_masked_card.parquet"
    )

    def _deck_uuid_for_row(self, row: dict, deck_box: DeckBox) -> UUID:
        """Resolve this row's winner's final deck and ensure it's
        stored in deck_box, via row_utils.deck_for_player() - the same
        shared per-player deck builder
        full_deck_win_prediction_metric.py, deck_pair_winner_metric.py,
        and multiplayer_placement_metric.py all delegate to.

        Inputs:
            row: one parsed Flavor A summary row.
            deck_box: same box passed to __init__.
        Output: the winner's final deck's uuid, now guaranteed present
            in deck_box. Raises if row has no resolvable winner (see
            Exceptions below) - unlike accumulate()'s target-lookup
            path, a deck is REQUIRED for every row this metric is fed,
            so callers upstream (e.g. the scanner) are expected to
            pre-filter to winner-resolvable rows, or accept the raise.
        Side effects: writes exactly one deck into deck_box via
            create_if_absent().
        Exceptions: raises if row_utils.winner_entry(row) returns None.
        """
        winner = winner_entry(row)
        if winner is None:
            raise ValueError(
                "WinningDeckMaskedCardMetric: row has no resolvable winner"
            )
        deck = deck_for_player(self._card_lookup, winner)
        deck_box.create_if_absent(deck)
        return deck.nocab_uuid

    def _target_card_uuid_for_row(
        self, row: dict, card_lookup: CardLookup
    ) -> UUID | None:
        """Pick the single card masked out of this row's winning deck.

        Inputs:
            row: one parsed Flavor A summary row.
            card_lookup: registry to resolve isotropic card names
                against.
        Output: the masked card's nocab_uuid, via
            _choose_masked_card_name() - or None if row has no
            resolvable winner, or the winner's deck is too small/
            unsuitable for masking under whatever rule
            _choose_masked_card_name() implements (a real "no valid
            target" outcome, not an error).
        Side effects: none.
        Exceptions: none.
        """
        winner = winner_entry(row)
        if winner is None:
            return None
        masked_name = self._choose_masked_card_name(winner["end"]["deck"])
        if masked_name is None:
            return None
        return card_uuid_for_name(card_lookup, masked_name)

    def _label_for_card(self, card: GenericCard) -> str:
        """The masked card's own name - the simplest possible label
        for this shape (predict the card's identity, not a derived
        property of it).

        Inputs:
            card: the target card _target_card_uuid_for_row() picked.
        Output: card.name.
        Side effects: none.
        Exceptions: none.
        """
        return card.name

    def _choose_masked_card_name(self, winner_end_deck: dict[str, int]) -> str | None:
        """Game-specific masking-target rule - see module docstring's
        MASKING RULE section: uniformly at random among the winner's
        non-basic cards.

        Private helper - single consumer is _target_card_uuid_for_row().

        Inputs:
            winner_end_deck: the winner's players[i]["end"]["deck"]
                dict (name -> copy count).
        Output: one card name from winner_end_deck's keys, or None if
            no valid choice exists under whatever rule is chosen.
        Side effects: implementation-defined.
        Exceptions: implementation-defined.
        """
        candidates = [name for name in winner_end_deck if name not in _BASIC_CARD_NAMES]
        if not candidates:
            return None
        return random.choice(candidates)
