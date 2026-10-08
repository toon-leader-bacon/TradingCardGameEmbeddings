from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.sts2_runs.option_pick_data_constructor import (
    OptionPickDataConstructor,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder(*cards: GenericCard) -> CardBinder:
    binder = CardBinder()
    for card in cards:
        binder.create(card)
    return binder


def _chunk(deck: list, offered: list, picked: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "deck_uuids": [[str(c.nocab_uuid) for c in deck]],
            "offered_uuids": [[str(c.nocab_uuid) for c in offered]],
            "picked_uuid": [picked],
        }
    )


class TestBuild:
    def test_a_pick_is_labelled_with_its_index_among_the_offered(self) -> None:
        a, b, c, deck_card = _card("A"), _card("B"), _card("C"), _card("Deck")
        binder = _binder(a, b, c, deck_card)

        result = OptionPickDataConstructor().build(
            _chunk([deck_card], [a, b, c], str(b.nocab_uuid)), binder
        )

        assert result == [([[a, b, c], [deck_card]], 1)]

    def test_a_skip_is_labelled_one_past_the_last_offered_card(self) -> None:
        a, b = _card("A"), _card("B")

        result = OptionPickDataConstructor().build(
            _chunk([], [a, b], None), _binder(a, b)
        )

        assert result == [([[a, b], []], 2)]

    def test_a_skip_read_back_from_parquet_as_nan_is_still_a_skip(self) -> None:
        a = _card("A")
        chunk = _chunk([], [a], None)
        chunk["picked_uuid"] = pd.Series([float("nan")])

        result = OptionPickDataConstructor().build(chunk, _binder(a))

        assert [label for _, label in result] == [1]

    def test_the_offer_is_group_0_and_the_deck_group_1(self) -> None:
        a, deck_card = _card("A"), _card("Deck")

        ((groups, _),) = OptionPickDataConstructor().build(
            _chunk([deck_card], [a], None), _binder(a, deck_card)
        )

        assert groups[0] == [a] and groups[1] == [deck_card]

    def test_a_row_with_an_unknown_offered_card_is_dropped(self) -> None:
        a, stranger = _card("A"), _card("Stranger")

        result = OptionPickDataConstructor().build(
            _chunk([], [a, stranger], None), _binder(a)
        )

        assert result == []

    def test_an_unknown_deck_card_is_dropped_but_the_row_stays(self) -> None:
        a, deck_card, stranger = _card("A"), _card("Deck"), _card("Stranger")

        result = OptionPickDataConstructor().build(
            _chunk([deck_card, stranger], [a], str(a.nocab_uuid)), _binder(a, deck_card)
        )

        assert result == [([[a], [deck_card]], 0)]

    def test_a_picked_card_that_was_not_offered_drops_the_row(self) -> None:
        a, other = _card("A"), _card("Other")

        result = OptionPickDataConstructor().build(
            _chunk([], [a], str(other.nocab_uuid)), _binder(a, other)
        )

        assert result == []

    def test_an_unparseable_pick_is_dropped_not_read_as_a_skip(self) -> None:
        a = _card("A")

        result = OptionPickDataConstructor().build(
            _chunk([], [a], "not-a-uuid"), _binder(a)
        )

        assert result == []

    def test_an_empty_deck_is_valid(self) -> None:
        a = _card("A")

        ((groups, _),) = OptionPickDataConstructor().build(
            _chunk([], [a], str(a.nocab_uuid)), _binder(a)
        )

        assert groups[1] == []
