from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games import kingdom_ending_type_metric
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.isotropic.deck_label_dojos import (
    FullDeckWinPredictionDojo,
    KingdomEndingTypeDojo,
    KingdomGameLengthDojo,
    NextTurnActionCountDojo,
    build_deck_label_constructor,
    label_capped_at,
)
from src.dojos.generic.renamed_column_data_constructor import (
    RenamedColumnDataConstructor,
)
from src.schema.holdout import HoldoutSpec
from tests.dojos.isotropic._fixtures import binder_and_group

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")

_UUID = "00000000-0000-0000-0000-000000000000"

# (dojo class, generic cell, group column, label column, a label value)
_CASES = [
    (
        FullDeckWinPredictionDojo,
        MultiCardBinaryClassificationDojo,
        "deck_uuid",
        "won",
        True,
    ),
    (
        KingdomGameLengthDojo,
        MultiCardRegressionDojo,
        "kingdom_uuid",
        "winner_turns",
        18,
    ),
    (
        NextTurnActionCountDojo,
        MultiCardRegressionDojo,
        "partial_deck_uuid",
        "next_turn_action_count",
        2,
    ),
    (
        KingdomEndingTypeDojo,
        MultiCardFixedClassificationDojo,
        "kingdom_uuid",
        "ending_type",
        "province",
    ),
]


def _write_source(path: Path, group_column: str, label_column: str, label) -> None:
    df = pd.DataFrame({group_column: [_UUID] * 10, label_column: [label] * 10})
    df.to_parquet(path, index=False)


def _build(dojo_cls, source: Path, deck_box: DeckBox):
    return dojo_cls(
        CardBinder(),
        HoldoutSpec.no_holdout(),
        deck_box,
        card_embedding_size=4,
        path_to_training_data=source,
        rng_seed=0,
        strict_version_check=False,
    )


class TestIsotropicDeckLabelDojos:
    @pytest.mark.parametrize("dojo_cls,cell,group_column,label_column,label", _CASES)
    def test_is_its_generic_cell(
        self, dojo_cls, cell, group_column, label_column, label, tmp_path: Path
    ) -> None:
        source = tmp_path / "isotropic_deck_label_source.parquet"
        _write_source(source, group_column, label_column, label)

        dojo = _build(dojo_cls, source, DeckBox())

        assert isinstance(dojo, cell)

    @pytest.mark.parametrize("dojo_cls,cell,group_column,label_column,label", _CASES)
    def test_constructor_reads_the_metrics_own_columns(
        self, dojo_cls, cell, group_column, label_column, label, tmp_path: Path
    ) -> None:
        source = tmp_path / "isotropic_deck_label_source.parquet"
        _write_source(source, group_column, label_column, label)
        binder, box, group = binder_and_group(["Chapel", "Village"])
        dojo = _build(dojo_cls, source, box)
        chunk = pd.DataFrame(
            {group_column: [str(group.nocab_uuid)], label_column: [label]}
        )

        result = dojo.data_constructor.build(chunk, binder)

        assert len(result) == 1
        cards, built_label = result[0]
        assert sorted(card.name for card in cards) == ["Chapel", "Village"]
        expected = label if isinstance(label, str) else float(label)
        assert built_label == expected

    def test_kingdom_ending_type_label_values_are_the_metrics_own(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "isotropic_deck_label_source.parquet"
        _write_source(source, "kingdom_uuid", "ending_type", "colony")

        dojo = _build(KingdomEndingTypeDojo, source, DeckBox())

        assert dojo.label_values == list(kingdom_ending_type_metric.LABEL_VALUES)


class TestBuildDeckLabelConstructor:
    def test_deck_uuid_column_needs_no_rename(self) -> None:
        result = build_deck_label_constructor(DeckBox(), "deck_uuid", "won")

        assert isinstance(result, DeckLabelDataConstructor)

    def test_other_group_column_is_renamed(self) -> None:
        result = build_deck_label_constructor(DeckBox(), "kingdom_uuid", "winner_turns")

        assert isinstance(result, RenamedColumnDataConstructor)


class TestLabelCaps:
    def test_label_capped_at_clips_only_above_the_cap(self) -> None:
        caster = label_capped_at(30.0)

        assert caster(112) == 30.0
        assert caster(2) == 2.0

    @pytest.mark.parametrize(
        "dojo_cls,group_column,label_column,raw,expected",
        [
            (KingdomGameLengthDojo, "kingdom_uuid", "winner_turns", 323, 50.0),
            (
                NextTurnActionCountDojo,
                "partial_deck_uuid",
                "next_turn_action_count",
                112,
                30.0,
            ),
        ],
    )
    def test_heavy_tailed_labels_are_capped(
        self, dojo_cls, group_column, label_column, raw, expected, tmp_path: Path
    ) -> None:
        source = tmp_path / "isotropic_deck_label_source.parquet"
        _write_source(source, group_column, label_column, raw)
        binder, box, group = binder_and_group(["Chapel"])
        dojo = _build(dojo_cls, source, box)
        chunk = pd.DataFrame(
            {group_column: [str(group.nocab_uuid)], label_column: [raw]}
        )

        [(_, label)] = dojo.data_constructor.build(chunk, binder)

        assert label == expected
