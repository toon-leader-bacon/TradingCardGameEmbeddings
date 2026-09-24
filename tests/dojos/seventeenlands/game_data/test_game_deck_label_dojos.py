from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.seventeenlands.game_data.game_deck_label_dojos import (
    DeckGameLengthPredictionDojo,
    DeckRankTierPredictionDojo,
    DeckWinPredictionDojo,
)
from src.schema.holdout import HoldoutSpec


def _write_source(path: Path, label_column: str, value) -> None:
    df = pd.DataFrame({"deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df[label_column] = value
    df.to_parquet(path, index=False)


class TestDeckGameLengthPredictionDojo:
    def test_wires_data_constructor_to_label_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, DeckGameLengthPredictionMetric.LABEL_COLUMN, 8)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = DeckGameLengthPredictionDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, MultiCardRegressionDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert (
            dojo.data_constructor._label_column
            == DeckGameLengthPredictionMetric.LABEL_COLUMN
        )


class TestDeckWinPredictionDojo:
    def test_wires_data_constructor_to_won_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, DeckWinPredictionMetric.LABEL_COLUMN, True)
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = DeckWinPredictionDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, MultiCardBinaryClassificationDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert (
            dojo.data_constructor._label_column == DeckWinPredictionMetric.LABEL_COLUMN
        )


class TestDeckRankTierPredictionDojo:
    def test_wires_data_constructor_to_rank_column_with_str_caster(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, DeckRankTierPredictionMetric.LABEL_COLUMN, "gold")
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = DeckRankTierPredictionDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, MultiCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert (
            dojo.data_constructor._label_column
            == DeckRankTierPredictionMetric.LABEL_COLUMN
        )
        assert dojo.data_constructor._label_caster is str

    def test_label_values_is_metrics_own_label_values(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, DeckRankTierPredictionMetric.LABEL_COLUMN, "gold")
        card_binder = CardBinder()
        deck_box = DeckBox()

        dojo = DeckRankTierPredictionDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert dojo.label_values == list(DeckRankTierPredictionMetric.LABEL_VALUES)
