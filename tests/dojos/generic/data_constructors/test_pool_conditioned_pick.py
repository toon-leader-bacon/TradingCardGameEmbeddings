from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import PoolConditionedPickDataConstructor
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str, source_id: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestPoolConditionedPickDataConstructorBuild:
    def test_builds_option_group_pool_group_and_pick_index(self) -> None:
        binder = CardBinder()
        card_a, card_b = _card("A", "a"), _card("B", "b")
        pool_card = _card("Pool", "pool")
        for card in (card_a, card_b, pool_card):
            binder.create(card)
        chunk = pd.DataFrame(
            {
                "pool_uuids": [[str(pool_card.nocab_uuid)]],
                "pack_option_uuids": [[str(card_a.nocab_uuid), str(card_b.nocab_uuid)]],
                "pick_uuid": [str(card_b.nocab_uuid)],
            }
        )
        constructor = PoolConditionedPickDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [([[card_a, card_b], [pool_card]], 1)]

    def test_empty_pool_produces_a_valid_row_with_an_empty_pool_group(self) -> None:
        binder = CardBinder()
        card_a = _card("A", "a")
        binder.create(card_a)
        chunk = pd.DataFrame(
            {
                "pool_uuids": [[]],
                "pack_option_uuids": [[str(card_a.nocab_uuid)]],
                "pick_uuid": [str(card_a.nocab_uuid)],
            }
        )
        constructor = PoolConditionedPickDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [([[card_a], []], 0)]

    def test_unresolvable_pool_uuid_is_dropped_not_skipped(self) -> None:
        binder = CardBinder()
        card_a = _card("A", "a")
        pool_card = _card("Pool", "pool")
        for card in (card_a, pool_card):
            binder.create(card)
        chunk = pd.DataFrame(
            {
                "pool_uuids": [[str(pool_card.nocab_uuid), str(uuid4())]],
                "pack_option_uuids": [[str(card_a.nocab_uuid)]],
                "pick_uuid": [str(card_a.nocab_uuid)],
            }
        )
        constructor = PoolConditionedPickDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [([[card_a], [pool_card]], 0)]

    def test_skips_row_with_null_pick(self) -> None:
        binder = CardBinder()
        card_a = _card("A", "a")
        binder.create(card_a)
        chunk = pd.DataFrame(
            {
                "pool_uuids": [[]],
                "pack_option_uuids": [[str(card_a.nocab_uuid)]],
                "pick_uuid": [None],
            }
        )
        constructor = PoolConditionedPickDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == []

    def test_skips_whole_row_when_any_option_is_unresolvable(self) -> None:
        binder = CardBinder()
        card_a = _card("A", "a")
        binder.create(card_a)
        chunk = pd.DataFrame(
            {
                "pool_uuids": [[]],
                "pack_option_uuids": [[str(card_a.nocab_uuid), str(uuid4())]],
                "pick_uuid": [str(card_a.nocab_uuid)],
            }
        )
        constructor = PoolConditionedPickDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == []
