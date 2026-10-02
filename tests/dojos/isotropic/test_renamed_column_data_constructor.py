import pandas as pd
import pytest

from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.isotropic.renamed_column_data_constructor import (
    RenamedColumnDataConstructor,
)
from tests.dojos.isotropic._fixtures import binder_and_group


class TestRenamedColumnDataConstructor:
    def test_builds_deck_label_data_from_a_renamed_uuid_column(self) -> None:
        binder, box, kingdom = binder_and_group(["Chapel", "Village", "Smithy"])
        constructor = RenamedColumnDataConstructor(
            DeckLabelDataConstructor(box, "winner_turns"),
            {"kingdom_uuid": "deck_uuid"},
        )
        chunk = pd.DataFrame(
            {"kingdom_uuid": [str(kingdom.nocab_uuid)], "winner_turns": [17]}
        )

        result = constructor.build(chunk, binder)

        assert len(result) == 1
        cards, label = result[0]
        assert sorted(card.name for card in cards) == ["Chapel", "Smithy", "Village"]
        assert label == 17.0

    def test_does_not_modify_the_given_chunk(self) -> None:
        binder, box, kingdom = binder_and_group(["Chapel"])
        constructor = RenamedColumnDataConstructor(
            DeckLabelDataConstructor(box, "winner_turns"),
            {"kingdom_uuid": "deck_uuid"},
        )
        chunk = pd.DataFrame(
            {"kingdom_uuid": [str(kingdom.nocab_uuid)], "winner_turns": [17]}
        )

        constructor.build(chunk, binder)

        assert list(chunk.columns) == ["kingdom_uuid", "winner_turns"]

    def test_rejects_empty_renames(self) -> None:
        _, box, _ = binder_and_group(["Chapel"])
        with pytest.raises(ValueError):
            RenamedColumnDataConstructor(DeckLabelDataConstructor(box, "x"), {})

    def test_rejects_two_columns_renamed_onto_one(self) -> None:
        _, box, _ = binder_and_group(["Chapel"])
        with pytest.raises(ValueError):
            RenamedColumnDataConstructor(
                DeckLabelDataConstructor(box, "x"),
                {"kingdom_uuid": "deck_uuid", "partial_deck_uuid": "deck_uuid"},
            )
