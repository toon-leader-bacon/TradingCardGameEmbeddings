import warnings
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.game_data_metrics.card_column_cache import (
    CardColumnCache,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _chunk(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: [] for column in columns})


class TestCardColumns:
    def test_resolves_from_the_first_chunks_header(self) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        cache = CardColumnCache(binder, GameId.MTG)

        result = cache.card_columns(
            _chunk(
                [
                    "deck_Bolt",
                    "opening_hand_Bolt",
                    "drawn_Bolt",
                    "tutored_Bolt",
                    "sideboard_Bolt",
                    "won",
                ]
            )
        )

        assert len(result) == 1
        assert result[0].nocab_uuid == bolt.nocab_uuid

    def test_caches_after_the_first_call(self) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt"))
        cache = CardColumnCache(binder, GameId.MTG)
        first_chunk = _chunk(
            [
                "deck_Bolt",
                "opening_hand_Bolt",
                "drawn_Bolt",
                "tutored_Bolt",
                "sideboard_Bolt",
                "won",
            ]
        )
        first_result = cache.card_columns(first_chunk)

        # A second chunk with a totally different header (as if it
        # would resolve differently) must NOT change the cached result
        # — only the first call's chunk is ever actually consulted.
        second_chunk = _chunk(["deck_Bear", "won"])
        second_result = cache.card_columns(second_chunk)

        assert second_result is first_result

    def test_unresolved_names_captured_and_warned(self) -> None:
        binder = CardBinder()  # nothing registered
        cache = CardColumnCache(binder, GameId.MTG)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cache.card_columns(_chunk(["deck_Bolt", "won"]))

        assert cache.unresolved_column_names == ["Bolt"]
        assert len(caught) == 1
        assert issubclass(caught[0].category, RuntimeWarning)

    def test_unresolved_column_names_empty_before_first_call(self) -> None:
        binder = CardBinder()
        cache = CardColumnCache(binder, GameId.MTG)

        assert cache.unresolved_column_names == []
