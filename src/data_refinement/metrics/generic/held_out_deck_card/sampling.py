"""HeldOutCardSampling: the knobs one HeldOutDeckCardMetric subclass
fixes per published deck box (how many decks, how many rows per deck,
how many decoys and of which kind, how hard staples are down-weighted).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeldOutCardSampling:
    """How HeldOutDeckCardMetric samples decks, targets and decoys.

    Fields:
        max_decks: how many decks to sample: the first max_decks decks,
            in DeckBox.uuids_ranked_randomly(game, seed) order, that
            have at least two distinct eligible cards (see
            DeckSample.from_deck_box); None samples every such deck.
        targets_per_deck: at most this many distinct target cards (one
            output row each) per sampled deck; >= 1.
        decoy_count: decoys per row (K); >= 1. A row has K + 1
            candidates unless the sample cannot supply K valid decoys.
        cooccurrence_decoy_share: fraction of the K decoys drawn from
            other decks that contain the target (hard negatives); the
            rest are frequency-matched. In [0, 1].
        staple_threshold: t in the staple weight
            w(c) = min(1, sqrt(t / f(c))), f(c) = the card's document
            frequency / sample size. > 0. Smaller t down-weights
            staples (basic lands, Copper) harder.
        seed: seeds the deck ranking and every random draw, so a rerun
            on the same box writes the same file.

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: ValueError (from __post_init__) for any field out of
        range.
    """

    max_decks: int | None
    targets_per_deck: int
    decoy_count: int
    cooccurrence_decoy_share: float = 0.5
    staple_threshold: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject out-of-range fields.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming the first bad field.
        """
        if self.max_decks is not None and self.max_decks < 1:
            raise ValueError(f"max_decks must be >= 1 or None, got {self.max_decks}")
        if self.targets_per_deck < 1:
            raise ValueError(
                f"targets_per_deck must be >= 1, got {self.targets_per_deck}"
            )
        if self.decoy_count < 1:
            raise ValueError(f"decoy_count must be >= 1, got {self.decoy_count}")
        if not 0.0 <= self.cooccurrence_decoy_share <= 1.0:
            raise ValueError(
                "cooccurrence_decoy_share must be in [0, 1], got "
                f"{self.cooccurrence_decoy_share}"
            )
        if self.staple_threshold <= 0.0:
            raise ValueError(
                f"staple_threshold must be > 0, got {self.staple_threshold}"
            )
