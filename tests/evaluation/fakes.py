"""Small stand-ins for evaluation tests: cards, a card lookup, an embedder."""

from datetime import datetime, timezone
from typing import Sequence
from uuid import uuid4

import torch

from src.encoder_model.precision import Precision
from src.evaluation.card_row import CardRow
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

WIDTH = 4


def make_card(name: str, game: GameId = GameId.MTG) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            DataSource.SCRYFALL, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def make_row(game: GameId = GameId.MTG) -> CardRow:
    return CardRow(uuid4(), game)


class FakeCardLookup:
    """all_cards / version_for over a fixed dict of cards per game."""

    def __init__(self, cards: dict[GameId, list[GenericCard]]) -> None:
        self._cards = cards

    def all_cards(self, source_game: GameId) -> list[GenericCard]:
        return list(self._cards[source_game])

    def version_for(self, source_game: GameId) -> str:
        return f"{source_game.value}-v1"


class FakeEmbedder:
    """Satisfies CardEmbedder. A card's vector is derived from its name, so
    it depends only on that card. fail_calls: 1-based call numbers that
    raise; nan_calls: call numbers whose output contains NaN."""

    def __init__(
        self,
        width: int = WIDTH,
        fail_calls: frozenset[int] = frozenset(),
        nan_calls: frozenset[int] = frozenset(),
    ) -> None:
        self._width = width
        self._fail_calls = fail_calls
        self._nan_calls = nan_calls
        self.calls: list[list[str]] = []

    @property
    def embedding_dim(self) -> int:
        return self._width

    def isolated_embeddings(
        self, cards: Sequence[GenericCard], precision: Precision = "fp32"
    ) -> torch.Tensor:
        self.calls.append([card.name for card in cards])
        call = len(self.calls)
        if call in self._fail_calls:
            raise RuntimeError(f"embedder failed on call {call}")
        rows = [vector_for(card.name, self._width) for card in cards]
        result = torch.tensor(rows, dtype=torch.float32).reshape(
            len(cards), self._width
        )
        if call in self._nan_calls:
            result[0, 0] = float("nan")
        return result


def vector_for(name: str, width: int = WIDTH) -> list[float]:
    """The FakeEmbedder's deterministic vector for a card name."""
    return [float(sum(map(ord, name)) % 97 + index) for index in range(width)]
