from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import MaskedFieldMultiLabelDataConstructor
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card() -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="Card",
        raw_content={"name": "Card"},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id="1",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _chunk(uuids: list[str], labels: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nocab_uuid": uuids,
            "masked_field": [["colors"]] * len(uuids),
            "label": labels,
        }
    )


class TestMaskedFieldMultiLabelDataConstructor:
    def test_builds_a_dense_indicator_dict(self) -> None:
        binder = CardBinder()
        card = binder.create(_card())
        constructor = MaskedFieldMultiLabelDataConstructor(["W", "U", "B"])

        result = constructor.build(_chunk([str(card.nocab_uuid)], [["B", "W"]]), binder)

        assert result == [(card, {0: 1.0, 1: 0.0, 2: 1.0})]

    def test_empty_label_is_all_zeros_and_unknown_values_are_ignored(self) -> None:
        binder = CardBinder()
        card = binder.create(_card())
        constructor = MaskedFieldMultiLabelDataConstructor(["W", "U"])

        result = constructor.build(
            _chunk([str(card.nocab_uuid)] * 2, [[], ["G"]]), binder
        )

        assert [label for _, label in result] == [{0: 0.0, 1: 0.0}] * 2

    def test_a_single_label_str_is_rejected(self) -> None:
        binder = CardBinder()
        card = binder.create(_card())
        constructor = MaskedFieldMultiLabelDataConstructor(["W", "U"])
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)],
                "masked_field": [["c"]],
                "label": ["W"],
            }
        )

        with pytest.raises(TypeError):
            constructor.build(chunk, binder)

    def test_skips_rows_whose_card_is_not_found(self) -> None:
        constructor = MaskedFieldMultiLabelDataConstructor(["W"])

        assert constructor.build(_chunk([str(uuid4())], [["W"]]), CardBinder()) == []

    @pytest.mark.parametrize("label_values", [[], ["W", "W"]])
    def test_rejects_an_empty_or_repeating_vocabulary(
        self, label_values: list[str]
    ) -> None:
        with pytest.raises(ValueError):
            MaskedFieldMultiLabelDataConstructor(label_values)
