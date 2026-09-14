from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import CardCharacterPredictionDataConstructor
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


class TestCardCharacterPredictionDataConstructorBuild:
    def test_single_character_row_builds_single_entry_dict(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "characters": [["CHARACTER.SILENT"]],
                "probabilities": [[1.0]],
                "sample_count": [1],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [(card, {"CHARACTER.SILENT": 1.0})]

    def test_multi_character_row_zips_into_one_dict(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "characters": [["CHARACTER.SILENT", "CHARACTER.REGENT"]],
                "probabilities": [[0.6, 0.4]],
                "sample_count": [5],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [(card, {"CHARACTER.SILENT": 0.6, "CHARACTER.REGENT": 0.4})]

    def test_skips_row_with_unresolvable_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(uuid4())],
                "characters": [["CHARACTER.SILENT"]],
                "probabilities": [[1.0]],
                "sample_count": [1],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == []

    def test_processes_multiple_rows_independently(self) -> None:
        binder = CardBinder()
        card1 = _card("Strike", "strike")
        card2 = _card("Defend", "defend")
        binder.create(card1)
        binder.create(card2)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card1.nocab_uuid), str(card2.nocab_uuid)],
                "characters": [["CHARACTER.SILENT"], ["CHARACTER.REGENT"]],
                "probabilities": [[1.0], [1.0]],
                "sample_count": [1, 1],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == [
            (card1, {"CHARACTER.SILENT": 1.0}),
            (card2, {"CHARACTER.REGENT": 1.0}),
        ]

    def test_empty_chunk_returns_empty_list(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [],
                "characters": [],
                "probabilities": [],
                "sample_count": [],
            }
        )
        constructor = CardCharacterPredictionDataConstructor(binder)

        result = constructor.build(chunk)

        assert result == []
