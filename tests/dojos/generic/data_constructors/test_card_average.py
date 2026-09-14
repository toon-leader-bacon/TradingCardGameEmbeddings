from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import CardAverageDataConstructor
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


class TestCardAverageDataConstructorBuild:
    def test_builds_one_datum_per_resolvable_row(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "average_relic_count": [1.5],
                "sample_count": [10],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == [(card, 1.5)]

    def test_skips_row_with_unresolvable_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(uuid4())],
                "average_relic_count": [1.5],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_malformed_uuid(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "nocab_uuid": ["not-a-uuid"],
                "average_relic_count": [1.5],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_skips_row_with_unparseable_label(self) -> None:
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "average_relic_count": ["not-a-float"],
            }
        )
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_reads_bool_label_column_as_float(self) -> None:
        # CardWinRateMetric's label column holds bools (win/loss) - see
        # CardAverageMetric's WIN RATE IS AN AVERAGE docstring note.
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "win_rate": [True],
            }
        )
        constructor = CardAverageDataConstructor(binder, "win_rate")

        result = constructor.build(chunk)

        assert result == [(card, 1.0)]

    def test_empty_chunk_returns_empty_list(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame({"nocab_uuid": [], "average_relic_count": []})
        constructor = CardAverageDataConstructor(binder, "average_relic_count")

        result = constructor.build(chunk)

        assert result == []

    def test_uuid_column_overrides_default_nocab_uuid(self) -> None:
        # TutorTargetPoolMetric's id column is "pool_card_uuid", not
        # "nocab_uuid" - see TutorTargetPoolDojo.
        binder = CardBinder()
        card = _card("Strike", "strike")
        binder.create(card)
        chunk = pd.DataFrame(
            {
                "pool_card_uuid": [str(card.nocab_uuid)],
                "tutored": [True],
            }
        )
        constructor = CardAverageDataConstructor(
            binder, "tutored", uuid_column="pool_card_uuid"
        )

        result = constructor.build(chunk)

        assert result == [(card, 1.0)]
