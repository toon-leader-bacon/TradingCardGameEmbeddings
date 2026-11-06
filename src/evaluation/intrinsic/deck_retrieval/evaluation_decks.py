"""EvaluationDeck and sample_evaluation_decks: the seeded VALIDATION decks
that deck-level analyses score every encoder on.

An EmbeddingTable holds cards only, so decks come from here, the way card
labels come from labels/. Precedent: DeckSample.from_deck_box
(src/data_refinement/metrics/generic/held_out_deck_card/deck_sample.py)
also takes a seeded sample of decks with distinct known cards. It ranks
with uuids_ranked_randomly, has no split filter and holds CSR arrays this
sampler does not need; deduplicating deck-sampling code is deferred to one
later pass.
"""

import hashlib
import logging
from dataclasses import dataclass
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_lookup import DeckLookup
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationDeck:
    """One deck as an analysis sees it.

    card_uuids: the deck's distinct cards in deck order, the game's
        Unknown placeholder removed (copies do not matter to a retrieval
        measurement).

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: none.
    """

    deck_uuid: UUID
    source_game: GameId
    card_uuids: tuple[UUID, ...]


@dataclass(frozen=True)
class EvaluationDeckSpec:
    """How many decks to sample, and how.

    count: decks wanted per game (fewer when the game has fewer).
    seed: orders the candidates; the same seed gives the same decks.
    min_cards: a deck needs at least this many distinct known cards.

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: ValueError if count < 1 or min_cards < 2.
    """

    count: int
    seed: int
    min_cards: int = 4

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        if self.min_cards < 2:
            raise ValueError(f"min_cards must be >= 2, got {self.min_cards}")


def sample_evaluation_decks(
    deck_lookup: DeckLookup,
    game: GameId,
    holdout: HoldoutSpec,
    spec: EvaluationDeckSpec,
) -> tuple[EvaluationDeck, ...]:
    """Up to spec.count VALIDATION decks of game, in a seeded order, each
    with at least spec.min_cards distinct known cards.

    The order is the sha256 of (seed, deck uuid), so it is the same across
    box rebuilds and machines and does not depend on SQLite row order. The
    VALIDATION candidates' (key, uuid) pairs are held in memory: a tenth of
    the game's decks at the default 8/1/1 split, but every deck of a game
    in holdout.held_out_games (all its decks are VALIDATION) - about 50 MB
    per million decks. Decks are loaded one at a
    time, in key order, until spec.count qualify. Card tiers are not
    filtered: the VALIDATION view sees every tier.

    Inputs: deck_lookup (only read), game, holdout (the run's spec; its
        deck split picks the decks), spec.
    Output: tuple of EvaluationDeck, in seeded order. Fewer than
        spec.count (logged) when the game has fewer qualifying decks;
        empty when it has none.
    Side effects: reads deck_lookup; logs a shortfall.
    Exceptions: UnknownDeckSplitError if holdout has no deck split (a spec
        from before decks were split by the holdout): its VALIDATION
        decks cannot be known, and guessing would score training decks.

    Example:
        >>> decks = sample_evaluation_decks(
        ...     box, GameId.GWENT, holdout, EvaluationDeckSpec(count=1000, seed=0))
        >>> len(decks)
        1000
    """
    result: list[EvaluationDeck] = []
    placeholder = CardBinder.unknown_card_uuid(game)

    # Validate inputs: refuse an unknown deck split even for a game with no
    # decks (deck_split_key raises UnknownDeckSplitError for one)
    holdout.deck_split_key(game)

    # Rank every VALIDATION deck by its seeded key
    candidates = _ranked_validation_uuids(deck_lookup, game, holdout, spec.seed)

    # Load decks in key order until enough qualify
    for deck_uuid in candidates:
        if len(result) >= spec.count:
            break
        deck = deck_lookup.get_by_uuid(deck_uuid)
        if deck is None:
            continue
        evaluation_deck = _evaluation_deck_of(deck, placeholder)
        if len(evaluation_deck.card_uuids) < spec.min_cards:
            continue
        result.append(evaluation_deck)

    # Report a shortfall; it is not an error
    if len(result) < spec.count:
        _logger.info(
            "%s: %d of %d evaluation decks found", game.value, len(result), spec.count
        )
    return tuple(result)


def _ranked_validation_uuids(
    deck_lookup: DeckLookup, game: GameId, holdout: HoldoutSpec, seed: int
) -> list[UUID]:
    """game's VALIDATION deck uuids, sorted by _ordering_key.

    Inputs: deck_lookup, game, holdout, seed. Output: list of UUID.
    Side effects: streams deck_lookup.all_uuids(game).
    Exceptions: UnknownDeckSplitError if holdout has no deck split.
    """
    keyed = [
        (_ordering_key(seed, deck_uuid), deck_uuid)
        for deck_uuid in deck_lookup.all_uuids(game)
        if holdout.deck_split_of(deck_uuid, game) == Split.VALIDATION
    ]
    return [deck_uuid for _, deck_uuid in sorted(keyed)]


def _ordering_key(seed: int, deck_uuid: UUID) -> bytes:
    """sha256 of f"{seed}:{deck_uuid}": a stable, seeded sort key.

    Inputs: seed, deck_uuid. Output: 32-byte digest. Side effects: none.
    Exceptions: none.
    """
    return hashlib.sha256(f"{seed}:{deck_uuid}".encode()).digest()


def _evaluation_deck_of(deck: GenericDeck, placeholder: UUID) -> EvaluationDeck:
    """deck's distinct cards in first-seen order, placeholder dropped.

    Inputs: deck, placeholder (the game's Unknown card uuid).
    Output: EvaluationDeck. Side effects: none. Exceptions: none.
    """
    distinct = dict.fromkeys(deck.card_nocab_uuids)
    distinct.pop(placeholder, None)
    return EvaluationDeck(deck.nocab_uuid, deck.source_game, tuple(distinct))
