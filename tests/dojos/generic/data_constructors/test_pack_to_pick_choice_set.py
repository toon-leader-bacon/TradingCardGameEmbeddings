from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import PackToPickChoiceSetDataConstructor
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


class TestPackToPickChoiceSetDataConstructorBuild:
    def test_builds_option_list_and_pick_index(self) -> None:
        binder = CardBinder()
        card_a, card_b, card_c = _card("A", "a"), _card("B", "b"), _card("C", "c")
        for card in (card_a, card_b, card_c):
            binder.create(card)
        chunk = pd.DataFrame(
            {
                "pack_option_uuids": [
                    [
                        str(card_a.nocab_uuid),
                        str(card_b.nocab_uuid),
                        str(card_c.nocab_uuid),
                    ]
                ],
                "pick_uuid": [str(card_b.nocab_uuid)],
            }
        )
        constructor = PackToPickChoiceSetDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == [([card_a, card_b, card_c], 1)]

    def test_skips_row_with_null_pick(self) -> None:
        binder = CardBinder()
        card_a = _card("A", "a")
        binder.create(card_a)
        chunk = pd.DataFrame(
            {
                "pack_option_uuids": [[str(card_a.nocab_uuid)]],
                "pick_uuid": [None],
            }
        )
        constructor = PackToPickChoiceSetDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == []

    def test_skips_row_when_pick_not_in_pack(self) -> None:
        binder = CardBinder()
        card_a = _card("A", "a")
        binder.create(card_a)
        chunk = pd.DataFrame(
            {
                "pack_option_uuids": [[str(card_a.nocab_uuid)]],
                "pick_uuid": [str(uuid4())],
            }
        )
        constructor = PackToPickChoiceSetDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == []

    def test_skips_whole_row_when_any_option_is_unresolvable(self) -> None:
        binder = CardBinder()
        card_a = _card("A", "a")
        binder.create(card_a)
        chunk = pd.DataFrame(
            {
                "pack_option_uuids": [[str(card_a.nocab_uuid), str(uuid4())]],
                "pick_uuid": [str(card_a.nocab_uuid)],
            }
        )
        constructor = PackToPickChoiceSetDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == []
