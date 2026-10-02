"""CandidateSampler: for one sampled deck, picks the held-out target
cards and, per target, the decoys it is ranked against.

Private to held_out_deck_card/ (single consumer: HeldOutDeckCardMetric).

Rules (the why is in ../README.md's HeldOutDeckCardMetric section):
- Targets: distinct cards of the deck, without replacement, weighted by
  the staple weight w(c) = min(1, sqrt(t / f(c))).
- Decoys: never the target, never a card of the deck, never repeated.
  A share of them are co-occurrence decoys (a card of another sampled
  deck that also holds the target, drawn from that deck with weight
  w(c) so donor staples are down-weighted like targets); the rest, and any co-occurrence
  decoy that cannot be found, are frequency-matched: drawn with
  probability proportional to df(c) * w(c), the same marginal targets
  are drawn from.
- Candidate order is shuffled.
"""

from dataclasses import dataclass
from uuid import UUID

import numpy as np

from src.data_refinement.metrics.generic.held_out_deck_card.deck_sample import (
    DeckSample,
)
from src.data_refinement.metrics.generic.held_out_deck_card.sampling import (
    HeldOutCardSampling,
)

# Rejection-sampling budget per decoy: draws that hit the deck, the
# target or an already-chosen decoy are retried at most this many times
# before that decoy slot is given up.
_MAX_DRAWS_PER_DECOY = 32


@dataclass(frozen=True)
class HeldOutCardRow:
    """One output row: a deck, the card held out of it, and the
    candidates (target + decoys, shuffled) the model ranks.

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: none.
    """

    deck_uuid: UUID
    target_card_uuid: UUID
    candidate_uuids: tuple[UUID, ...]


class CandidateSampler:
    """Draws targets and decoys over one DeckSample (Strategy-free on
    purpose: there is one rule set today; see the module docstring)."""

    def __init__(self, sample: DeckSample, sampling: HeldOutCardSampling) -> None:
        """Precompute per-card staple weights and the frequency-matched
        decoy distribution.

        Inputs:
            sample: the deck sample to draw from; never modified.
            sampling: targets_per_deck, decoy_count,
                cooccurrence_decoy_share, staple_threshold and seed are
                read (max_decks was already applied by the sample).
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if sample has no decks.
        """
        if sample.deck_count == 0:
            raise ValueError("cannot sample candidates from an empty deck sample")
        self._sample = sample
        self._sampling = sampling
        self._rng = np.random.default_rng(sampling.seed)
        document_frequencies = sample.document_frequencies()
        self._staple_weights = _staple_weights(
            document_frequencies, sample.deck_count, sampling.staple_threshold
        )
        self._decoy_cumulative = _cumulative_distribution(
            document_frequencies * self._staple_weights
        )
        self._cooccurrence_decoy_count = round(
            sampling.decoy_count * sampling.cooccurrence_decoy_share
        )

    def rows_for(self, deck_index: int) -> list[HeldOutCardRow]:
        """Every output row for one sampled deck: one per drawn target.

        Inputs: deck_index (int, 0 <= deck_index < sample.deck_count).
        Output: list[HeldOutCardRow], at most targets_per_deck long. A
            target for which not even one decoy can be found yields no
            row.
        Side effects: advances this sampler's random state.
        Exceptions: IndexError for an out-of-range deck_index.

        Example:
            >>> sampler = CandidateSampler(sample, sampling)
            >>> sampler.rows_for(0)[0].candidate_uuids
            (UUID('...'), UUID('...'), ...)
        """
        result: list[HeldOutCardRow] = []
        deck_cards = self._sample.cards_of(deck_index)

        # One row per target: its decoys, shuffled in with it
        for target in self._targets_for(deck_cards):
            decoys = self._decoys_for(deck_index, deck_cards, int(target))
            if not decoys:
                continue
            candidates = self._rng.permutation([int(target), *decoys])
            result.append(
                HeldOutCardRow(
                    deck_uuid=self._sample.deck_uuids[deck_index],
                    target_card_uuid=self._sample.card_uuids[int(target)],
                    candidate_uuids=tuple(
                        self._sample.card_uuids[int(card)] for card in candidates
                    ),
                )
            )
        return result

    def _targets_for(self, deck_cards: np.ndarray) -> np.ndarray:
        """Up to targets_per_deck distinct cards of the deck, drawn
        without replacement, weighted by staple weight.

        Inputs: deck_cards (the deck's distinct card indices).
        Output: np.ndarray of card indices.
        Side effects: advances self._rng. Exceptions: none.
        """
        weights = self._staple_weights[deck_cards]
        count = min(self._sampling.targets_per_deck, len(deck_cards))
        return self._rng.choice(
            deck_cards, size=count, replace=False, p=weights / weights.sum()
        )

    def _decoys_for(
        self, deck_index: int, deck_cards: np.ndarray, target: int
    ) -> list[int]:
        """Up to decoy_count distinct decoys for target: co-occurrence
        decoys first, then frequency-matched ones for the remaining
        slots (including co-occurrence slots that came up empty).

        Inputs: deck_index, deck_cards (sorted, for np.searchsorted
            membership tests), target (a card index in deck_cards).
        Output: list[int] of card indices, none in deck_cards, no
            repeats, possibly shorter than decoy_count.
        Side effects: advances self._rng. Exceptions: none.
        """
        result: list[int] = []

        # Hard negatives: cards that share a deck with the target
        for _ in range(self._cooccurrence_decoy_count):
            decoy = self._cooccurrence_decoy(deck_index, deck_cards, target, result)
            if decoy is not None:
                result.append(decoy)

        # Frequency-matched negatives fill every remaining slot
        while len(result) < self._sampling.decoy_count:
            decoy = self._frequency_decoy(deck_cards, result)
            if decoy is None:
                break
            result.append(decoy)
        return result

    def _cooccurrence_decoy(
        self, deck_index: int, deck_cards: np.ndarray, target: int, chosen: list[int]
    ) -> int | None:
        """One card from a random other sampled deck that contains
        target, drawn from that deck with weight w(c), not in
        deck_cards and not already chosen.

        Inputs: deck_index (excluded as a donor), deck_cards, target,
            chosen (decoys so far).
        Output: a card index, or None after _MAX_DRAWS_PER_DECOY failed
            draws (e.g. the target occurs in no other sampled deck).
        Side effects: advances self._rng. Exceptions: none.
        """
        donors = self._sample.decks_containing(target)
        if len(donors) < 2:
            return None

        # Uniform donor deck, uniform card of it, kept with probability
        # w(c): exactly a w-weighted draw within the donor deck
        for _ in range(_MAX_DRAWS_PER_DECOY):
            donor = int(donors[self._rng.integers(len(donors))])
            if donor == deck_index:
                continue
            donor_cards = self._sample.cards_of(donor)
            card = int(donor_cards[self._rng.integers(len(donor_cards))])
            if self._rng.random() >= self._staple_weights[card]:
                continue
            if _contains(deck_cards, card) or card in chosen:
                continue
            return card
        return None

    def _frequency_decoy(self, deck_cards: np.ndarray, chosen: list[int]) -> int | None:
        """One card drawn proportional to df(c) * w(c), not in
        deck_cards and not already chosen.

        Inputs: deck_cards, chosen.
        Output: a card index, or None after _MAX_DRAWS_PER_DECOY failed
            draws.
        Side effects: advances self._rng. Exceptions: none.
        """
        for _ in range(_MAX_DRAWS_PER_DECOY):
            card = int(
                np.searchsorted(self._decoy_cumulative, self._rng.random(), "right")
            )
            card = min(card, len(self._decoy_cumulative) - 1)  # float round-off
            if _contains(deck_cards, card) or card in chosen:
                continue
            return card
        return None


def _staple_weights(
    document_frequencies: np.ndarray, deck_count: int, staple_threshold: float
) -> np.ndarray:
    """w(c) = min(1, sqrt(t / f(c))), f(c) = df(c) / deck_count.

    Inputs: document_frequencies (int64 per card, each >= 1),
        deck_count (>= 1), staple_threshold (t > 0).
    Output: np.ndarray of float64 in (0, 1], per card.
    Side effects: none. Exceptions: none.
    """
    frequencies = document_frequencies / deck_count
    return np.minimum(1.0, np.sqrt(staple_threshold / frequencies))


def _cumulative_distribution(weights: np.ndarray) -> np.ndarray:
    """Normalized cumulative sum of weights, for np.searchsorted draws.

    Inputs: weights (non-negative float64, positive sum).
    Output: np.ndarray of float64, last entry 1.0.
    Side effects: none. Exceptions: ValueError if weights sum to 0.
    """
    total = weights.sum()
    if total <= 0:
        raise ValueError("decoy weights sum to 0")
    result = np.cumsum(weights) / total
    result[-1] = 1.0
    return result


def _contains(sorted_cards: np.ndarray, card: int) -> bool:
    """Whether card is in sorted_cards (binary search).

    Inputs: sorted_cards (ascending int array), card.
    Output: bool. Side effects: none. Exceptions: none.
    """
    position = int(np.searchsorted(sorted_cards, card))
    return position < len(sorted_cards) and int(sorted_cards[position]) == card
