"""game_data_metrics-specific data types.

See this container's own README for the contract CardColumnSet was
built against. MetricResult lives at the shared
src/data_refinement/seventeenlands/ level instead (../metric_result.py),
since it has no game_data-specific needs of its own.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class CardColumnSet:
    """One card's five 17lands game_data column names, resolved to a card.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: UUID
    opening_hand: str  # column names for this card, in this file
    drawn: str
    tutored: str
    deck: str
    sideboard: str
