"""DeckScore and anchor_aucs: one deck's same-deck retrieval AUC per anchor
card. Private to DeckRetrievalAuc (single consumer)."""

from dataclasses import dataclass
from uuid import UUID

import numpy as np

from src.schema.game_id import GameId


# eq=False: numpy fields have no single-bool equality
@dataclass(frozen=True, eq=False)
class DeckScore:
    """One scored deck.

    anchor_aucs: float64 (n_cards,), anchor i's Mann-Whitney AUC.
    anchor_is_train: bool (n_cards,), anchor i's holdout tier is TRAIN.
    n_cards: the deck's cards found in the table (>= 2).

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: ValueError unless anchor_aucs and anchor_is_train are 1-D,
        of equal length, with at least 2 anchors.
    """

    deck_uuid: UUID
    source_game: GameId
    anchor_aucs: np.ndarray
    anchor_is_train: np.ndarray

    def __post_init__(self) -> None:
        shapes = (self.anchor_aucs.shape, self.anchor_is_train.shape)
        if self.anchor_aucs.ndim != 1 or shapes[0] != shapes[1]:
            raise ValueError(f"anchor arrays must be 1-D and aligned, got {shapes}")
        if len(self.anchor_aucs) < 2:
            raise ValueError(f"a deck needs >= 2 anchors, got {len(self.anchor_aucs)}")

    @property
    def n_cards(self) -> int:
        """Inputs: none. Output: int. Side effects: none. Exceptions: none."""
        return len(self.anchor_aucs)

    @property
    def n_train_anchors(self) -> int:
        """Inputs: none. Output: int. Side effects: none. Exceptions: none."""
        return int(np.count_nonzero(self.anchor_is_train))

    @property
    def n_heldout_anchors(self) -> int:
        """Inputs: none. Output: int. Side effects: none. Exceptions: none."""
        return self.n_cards - self.n_train_anchors

    @property
    def auc(self) -> float:
        """Mean anchor AUC.

        Inputs: none. Output: float in [0, 1]. Side effects: none.
        Exceptions: none.

        Example:
            >>> score.auc
            0.81
        """
        return float(self.anchor_aucs.mean())

    def tier_auc(self, train: bool) -> float | None:
        """Mean AUC of the TRAIN anchors (train=True) or the non-TRAIN
        anchors (train=False); None when the deck has no anchor of that tier.

        Inputs: train (bool). Output: float or None. Side effects: none.
        Exceptions: none.

        Example:
            >>> score.tier_auc(train=False)
            0.77
        """
        chosen = self.anchor_aucs[self.anchor_is_train == train]
        return float(chosen.mean()) if len(chosen) else None


def anchor_aucs(
    anchor_cosines: np.ndarray, member_rows: np.ndarray, negative_rows: np.ndarray
) -> np.ndarray:
    """Each deck member's Mann-Whitney AUC as an anchor: over every
    (deck-mate, negative) pair, 1 when the deck-mate's cosine to the anchor
    is higher, 1/2 on a tie, 0 otherwise, divided by the number of pairs.

    Every cosine comes from one product (anchor_cosines), so a deck-mate
    and a negative with the same vector get the same cosine and tie. Per
    anchor, the negatives' cosines are sorted once and each deck-mate's
    wins counted by binary search: about k * n log n for k members and n
    negatives.

    Inputs: anchor_cosines float64 (k, g): row i is member i's cosine to
        each of the game's g cards; member_rows int (k,), distinct: the
        members' columns, in anchor_cosines' row order; negative_rows int
        (n,), n >= 1: the negatives' columns, disjoint from member_rows.
    Output: float64 (k,), each in [0, 1].
    Side effects: none.
    Exceptions: none (the caller guarantees k >= 2, n >= 1, disjoint rows).

    Example:
        >>> cosines = unit(game_vectors)[members] @ unit(game_vectors).T
        >>> anchor_aucs(cosines, members, negatives)
        array([0.91, 0.84, 0.88])
    """
    member_count = len(member_rows)
    result = np.empty(member_count, dtype=np.float64)

    # Split each anchor's cosines into deck-mates and (sorted) negatives
    mate_cosines = anchor_cosines[:, member_rows]
    negative_cosines = np.sort(anchor_cosines[:, negative_rows], axis=1)

    # Count each anchor's wins and ties over its (deck-mate, negative) pairs
    for anchor in range(member_count):
        mates = np.delete(mate_cosines[anchor], anchor)
        result[anchor] = _pair_win_share(mates, negative_cosines[anchor])
    return result


def _pair_win_share(
    mate_cosines: np.ndarray, sorted_negative_cosines: np.ndarray
) -> float:
    """(wins + ties / 2) / pairs for one anchor.

    Inputs: mate_cosines (k - 1,); sorted_negative_cosines (n,), ascending.
    Output: float in [0, 1]. Side effects: none. Exceptions: none.
    """
    less = np.searchsorted(sorted_negative_cosines, mate_cosines, side="left")
    less_or_equal = np.searchsorted(sorted_negative_cosines, mate_cosines, side="right")
    wins = less.sum() + 0.5 * (less_or_equal - less).sum()
    return float(wins / (len(mate_cosines) * len(sorted_negative_cosines)))
