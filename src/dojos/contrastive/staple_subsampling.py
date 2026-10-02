"""Staple subsampling for contrastive pairs: thin a deck's common cards
before its items are sampled.

Base cards (Copper, Estate, basic lands) sit in most decks of a game, so
they fill many same-deck positive pairs while saying little about the
deck. word2vec's frequent-word subsampling (Mikolov et al. 2013), which
item2vec (Barkan & Koenigstein 2016) applied to item sets, keeps each
occurrence of a card with probability min(1, sqrt(t / df)), where df is
the share of the game's decks containing the card. t = inf keeps every
card (no change); a moderate t thins the staples; a tiny t nearly always
drops them.

df comes from a deck box's TRAIN split (a fixed, seeded sample of decks,
read through its DeckBoxDealer) and is cached as a small JSON file under
data/splits/. The cache key is the box's CardBinder version plus the
sample size, so a rebuilt dojo reuses it. A box re-minted against a new
binder version gets recounted. A box re-ingested against the same version,
or a rebuilt split index, does not: delete the cache file in those cases,
as you would the split index. The deck box itself is only read.

The held-out-card metric uses the same formula for its candidate
sampling (data_refinement/metrics/generic/held_out_deck_card/
candidate_sampler.py). Keep the two in step.
"""

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping
from uuid import UUID

from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.schema.card import GenericDeck
from src.schema.splits import Split

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentFrequency:
    """The share of a sample of one game's decks that contains each card.

    card_binder_version: the CardBinder version the decks were minted
        against (the cache key). deck_count: decks in the sample (> 0).
    shares: card uuid -> share of the sample's decks containing it, in
        (0, 1]; a card absent from the sample has share 0.
    """

    card_binder_version: str
    deck_count: int
    shares: Mapping[UUID, float]

    def __post_init__(self) -> None:
        # Validate inputs: a share is a fraction of a non-empty sample
        if self.deck_count <= 0:
            raise ValueError(f"deck_count must be positive, got {self.deck_count}")
        for card_uuid, share in self.shares.items():
            if not 0 < share <= 1:
                raise ValueError(f"share of {card_uuid} must be in (0, 1], got {share}")

    @classmethod
    def from_decks(
        cls, decks: Iterable[GenericDeck], card_binder_version: str
    ) -> "DocumentFrequency":
        """Count, for each card, the decks that contain it at least once.

        Inputs: decks (a non-empty sample), card_binder_version.
        Output: DocumentFrequency. Side effects: none.
        Exceptions: ValueError if decks is empty.

        Example:
            >>> DocumentFrequency.from_decks([deck_a, deck_b], "v1").share_of(copper)
            1.0
        """
        containing: dict[UUID, int] = {}
        deck_count = 0
        for deck in decks:
            deck_count += 1
            for card_uuid in set(deck.card_nocab_uuids):
                containing[card_uuid] = containing.get(card_uuid, 0) + 1
        if not deck_count:
            raise ValueError("cannot measure document frequency over no decks")
        shares = {card: count / deck_count for card, count in containing.items()}
        return cls(card_binder_version, deck_count, shares)

    def share_of(self, card_uuid: UUID) -> float:
        """card_uuid's share of decks; 0.0 for a card the sample lacks.
        Inputs: card_uuid. Output: float in [0, 1]. Side effects: none.
        Exceptions: none."""
        return self.shares.get(card_uuid, 0.0)

    def to_json(self) -> str:
        """This table as JSON (uuids as strings).
        Inputs: none. Output: str. Side effects: none. Exceptions: none."""
        return json.dumps(
            {
                "card_binder_version": self.card_binder_version,
                "deck_count": self.deck_count,
                "shares": {str(card): share for card, share in self.shares.items()},
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "DocumentFrequency":
        """Parse to_json's output.

        Inputs: text. Output: DocumentFrequency. Side effects: none.
        Exceptions: ValueError (json.JSONDecodeError is one) or KeyError /
            TypeError for malformed text; as __post_init__.

        Example:
            >>> DocumentFrequency.from_json(table.to_json()) == table
            True
        """
        raw = json.loads(text)
        shares = {UUID(card): float(share) for card, share in raw["shares"].items()}
        return cls(str(raw["card_binder_version"]), int(raw["deck_count"]), shares)


@dataclass(frozen=True)
class StapleSubsampling:
    """Keeps each card occurrence with probability min(1, sqrt(threshold /
    df)), df from frequency. A card with df <= threshold (including one the
    sample never saw) is always kept.

    threshold: t, finite and > 0 (t = inf is "no subsampling", which a
        caller expresses by passing no StapleSubsampling at all, so the
        default path never draws from an RNG).
    """

    threshold: float
    frequency: DocumentFrequency

    def __post_init__(self) -> None:
        # Validate inputs: `not t > 0` also rejects NaN
        if not self.threshold > 0 or math.isinf(self.threshold):
            raise ValueError(f"threshold must be finite and > 0, got {self.threshold}")

    def keep_probability(self, card_uuid: UUID) -> float:
        """min(1, sqrt(threshold / df)); 1.0 when df is 0.

        Inputs: card_uuid. Output: float in (0, 1]. Side effects: none.
        Exceptions: none.

        Example:
            >>> StapleSubsampling(0.25, table).keep_probability(copper)  # df 1
            0.5
        """
        share = self.frequency.share_of(card_uuid)
        if share <= self.threshold:
            return 1.0
        return math.sqrt(self.threshold / share)

    def kept(self, card_uuids: list[UUID], rng: random.Random) -> list[UUID]:
        """The occurrences that survive one roll each, in order: a new list
        (card_uuids is only read).

        Inputs: card_uuids (one deck's cards, duplicates as occurrences),
            rng (the caller's sampling RNG).
        Output: list[UUID], possibly empty.
        Side effects: advances rng once per occurrence.
        Exceptions: none.

        Example:
            >>> StapleSubsampling(0.01, table).kept([copper, smithy], rng)
            [smithy]
        """
        return [
            card_uuid
            for card_uuid in card_uuids
            if rng.random() < self.keep_probability(card_uuid)
        ]


def cached_document_frequency(
    dealer: DeckBoxDealer, cache_path: Path, sample_decks: int
) -> DocumentFrequency:
    """dealer's TRAIN document frequency, from cache_path when it matches
    the box's CardBinder version and sample size, else counted and cached.

    The sample is the first sample_decks TRAIN decks in the dealer's fixed
    (seeded) order, or the whole split if smaller, so a recount gives the
    same table.

    Inputs: dealer, cache_path (a JSON file under data/splits/),
        sample_decks (> 0).
    Output: DocumentFrequency.
    Side effects: reads decks through dealer (the deck box is only read);
        writes cache_path (and its parent directory) on a miss; logs one
        INFO line on a recount.
    Exceptions: ValueError if sample_decks <= 0, the box has no recorded
        CardBinder version, or its TRAIN split is empty.

    Example:
        >>> cached_document_frequency(dealer, Path("data/splits/x.df.json"), 20_000)
    """
    if sample_decks <= 0:
        raise ValueError(f"sample_decks must be positive, got {sample_decks}")
    version = dealer.card_binder_version
    if version is None:
        raise ValueError("the deck box has no recorded CardBinder version")

    # Reuse the cached table when it was counted the same way from this box
    cached = _read_cache(cache_path)
    expected_count = min(sample_decks, dealer.deck_count(Split.TRAIN))
    if (
        cached is not None
        and cached.card_binder_version == version
        and cached.deck_count == expected_count
    ):
        return cached

    # Otherwise count a fixed TRAIN sample and cache it
    result = DocumentFrequency.from_decks(
        _first_train_decks(dealer, sample_decks), version
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result.to_json(), encoding="utf-8")
    _logger.info(
        "Counted document frequency over %d TRAIN decks into %s",
        result.deck_count,
        cache_path,
    )
    return result


def _first_train_decks(dealer: DeckBoxDealer, limit: int) -> Iterator[GenericDeck]:
    """The first limit TRAIN decks in the dealer's fixed order (fewer if
    the split is smaller).

    Inputs: dealer, limit (> 0). Output: iterator of GenericDeck.
    Side effects: reads decks through dealer. Exceptions: none.
    """
    # One deck per group: decks_for drops a trailing partial group, and it
    # reads each deck on its own either way
    for count, (deck,) in enumerate(dealer.decks_for(Split.TRAIN, 1, shuffle=False)):
        if count >= limit:
            return
        yield deck


def _read_cache(cache_path: Path) -> DocumentFrequency | None:
    """The table at cache_path, or None if it is absent or unreadable
    (a bad cache is recounted, never trusted).

    Inputs: cache_path. Output: DocumentFrequency | None.
    Side effects: reads cache_path; logs a warning for an unreadable file.
    Exceptions: none.
    """
    if not cache_path.exists():
        return None
    try:
        return DocumentFrequency.from_json(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        _logger.warning("Ignoring unreadable cache %s: %r", cache_path, error)
        return None
