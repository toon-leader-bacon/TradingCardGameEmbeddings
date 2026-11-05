"""TeacherInputs: one game's deck snapshot with a holdout applied, as the
caller hands it to the teacher.

src/teacher never imports the dojos, so the deck split arrives as plain
row arrays (the caller takes them from DeckSplitIndex.rows_of); card
tiers come from the HoldoutSpec itself.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.data_refinement.snapshots.stamps import DeckSnapshotStamp
from src.schema.holdout import CardTier, HoldoutSpec
from src.schema.splits import Split


@dataclass(frozen=True, eq=False)
class TeacherInputs:
    """One game's decks, their split, and the holdout behind it.

    decks: the game's open deck snapshot.
    deck_rows_by_split: every split's snapshot rows (int64, ascending), as
        DeckSplitIndex.rows_of gives them.
    deck_rows_snapshot: the stamp of the snapshot those rows number (the
        split index's snapshot_stamp).
    deck_split_key: the split index's key.
    holdout: the run's spec; card tiers come from holdout.tier_of.
    """

    decks: DeckSnapshot
    deck_rows_by_split: Mapping[Split, np.ndarray]
    deck_rows_snapshot: DeckSnapshotStamp
    deck_split_key: str
    holdout: HoldoutSpec

    def __post_init__(self) -> None:
        """Raise ValueError if the rows number another snapshot, split by
        another key, or a split is missing."""
        # Validate inputs: the rows number this snapshot, under this holdout
        if self.deck_rows_snapshot != self.decks.stamp:
            raise ValueError(
                "TeacherInputs: the deck rows number another "
                f"{self.decks.source_game.value} deck snapshot; rebuild the split index"
            )
        expected_key = self.holdout.deck_split_key(self.decks.source_game)
        if self.deck_split_key != expected_key:
            raise ValueError(
                f"TeacherInputs: the deck rows split by {self.deck_split_key}, "
                f"the holdout by {expected_key}"
            )
        missing = sorted(
            split.value for split in Split if split not in self.deck_rows_by_split
        )
        if missing:
            raise ValueError(f"TeacherInputs: no deck rows for splits {missing}")

    def selected_deck_rows(self, deck_splits: frozenset[Split]) -> np.ndarray:
        """The snapshot rows of every deck in deck_splits, ascending.

        Inputs: deck_splits (non-empty).
        Output: np.ndarray int64 (a new array; the inputs stay read-only).
        Side effects: none. Exceptions: none.

        Example:
            >>> inputs.selected_deck_rows(frozenset({Split.TRAIN}))
        """
        chosen = [self.deck_rows_by_split[split] for split in sorted(deck_splits)]
        return np.sort(np.concatenate(chosen).astype(np.int64, copy=False))

    def selected_card_mask(self, card_tiers: frozenset[CardTier]) -> np.ndarray:
        """Which card snapshot rows the teacher keeps: those whose
        holdout.tier_of is in card_tiers, never the unknown sentinel.

        Inputs: card_tiers (non-empty).
        Output: np.ndarray bool (card_snapshot.card_count,).
        Side effects: none (one tier_of per card: ~30k hashes for MTG).
        Exceptions: none.

        Example:
            >>> mask = inputs.selected_card_mask(frozenset({CardTier.TRAIN}))
        """
        cards = self.decks.card_snapshot
        game = self.decks.source_game
        result = np.zeros(cards.card_count, dtype=bool)

        # Keep each card whose tier is chosen
        for row, card_uuid in enumerate(cards.card_uuids):
            result[row] = self.holdout.tier_of(card_uuid, game) in card_tiers

        # The sentinel stands for cards the binder lacks: never a teacher row
        result[cards.unknown_index] = False
        return result
