"""Dojos over the isotropic Dominion metrics
(src/data_refinement/metrics/isotropic/): thin generic-cell wrappers, one
per metric. Card-level dojos need only the Dominion CardBinder; deck-level
ones also need the metrics-private DeckBox at
data/metrics/isotropic/deck_box.db.
"""

from src.dojos.isotropic.card_rate_dojos import (
    AverageCopiesBoughtDojo,
    OpeningBuyRateDojo,
    PileExhaustionRateDojo,
    TurnCountAssociationDojo,
    VetoRateDojo,
)
from src.dojos.isotropic.deck_card_mask_dojos import WinningDeckMaskedCardDojo
from src.dojos.isotropic.deck_label_dojos import (
    FullDeckWinPredictionDojo,
    KingdomEndingTypeDojo,
    KingdomGameLengthDojo,
    NextTurnActionCountDojo,
)
from src.dojos.isotropic.group_label_dojos import (
    DeckCardSetCopyCountDojo,
    DeckPairWinnerDojo,
    EventualWinDojo,
    KingdomEndingPileDojo,
    MidGameDeckPairWinnerDojo,
    OpeningBuyOutcomeDojo,
    WinningDeckCountDojo,
    WinningDeckMembershipDojo,
)
from src.dojos.isotropic.pick_dojos import (
    KingdomOpeningBuyDojo,
    KingdomVetoDojo,
    NextBuyDojo,
    NextTrashedCardDojo,
)

__all__ = [
    "AverageCopiesBoughtDojo",
    "DeckCardSetCopyCountDojo",
    "DeckPairWinnerDojo",
    "EventualWinDojo",
    "FullDeckWinPredictionDojo",
    "KingdomEndingPileDojo",
    "KingdomEndingTypeDojo",
    "KingdomGameLengthDojo",
    "KingdomOpeningBuyDojo",
    "KingdomVetoDojo",
    "MidGameDeckPairWinnerDojo",
    "NextBuyDojo",
    "NextTrashedCardDojo",
    "NextTurnActionCountDojo",
    "OpeningBuyOutcomeDojo",
    "OpeningBuyRateDojo",
    "PileExhaustionRateDojo",
    "TurnCountAssociationDojo",
    "VetoRateDojo",
    "WinningDeckCountDojo",
    "WinningDeckMaskedCardDojo",
    "WinningDeckMembershipDojo",
]
