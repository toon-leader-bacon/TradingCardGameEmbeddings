from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import MaskedFieldDataConstructor
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


class TestMaskedFieldDataConstructorBuild:
    def test_builds_one_datum_per_resolvable_row(self) -> None:
        binder = CardBinder()
        card = _card("Geralt of Rivia", "geralt")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "masked_field": [["faction"]],
                "label": ["northern_realms"],
            }
        )
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == [(card, "northern_realms")]

    def test_skips_row_with_unresolvable_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(uuid4())],
                "masked_field": [["faction"]],
                "label": ["northern_realms"],
            }
        )
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_malformed_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": ["not-a-uuid"],
                "masked_field": [["faction"]],
                "label": ["northern_realms"],
            }
        )
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == []

    def test_empty_chunk_returns_empty_list(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame({"nocab_uuid": [], "masked_field": [], "label": []})
        constructor = MaskedFieldDataConstructor(binder, "label")

        result = constructor.build(chunk)

        assert result == []
