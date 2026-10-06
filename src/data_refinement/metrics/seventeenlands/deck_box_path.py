"""Where a 17lands family's private DeckBox lives.

scripts/run_metrics.py's game_data and replay_data runs write it; their
deck-level dojos read it back (each row's deck_uuid points into it).
draft_data has no deck box.
"""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.partition import (
    METRICS_ROOT,
    SOURCE_DIRECTORY,
)
from src.data_retrieval.seventeenlands.refs import DataType

DECK_BOX_FILE_NAME = "deck_box.db"


def seventeenlands_deck_box_path(
    family: DataType, metrics_root: Path = METRICS_ROOT
) -> Path:
    """The family's deck box: <metrics_root>/seventeenlands/<family>/deck_box.db.

    Inputs: family (GAME or REPLAY), metrics_root.
    Output: Path. Side effects: none.
    Exceptions: ValueError for DataType.DRAFT, which has no deck box.

    Example:
        >>> seventeenlands_deck_box_path(DataType.GAME)
        Path('data/metrics/seventeenlands/game_data/deck_box.db')
    """
    if family is DataType.DRAFT:
        raise ValueError("draft_data has no deck box")
    return metrics_root / SOURCE_DIRECTORY / family.value / DECK_BOX_FILE_NAME
