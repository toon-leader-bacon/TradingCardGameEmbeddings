from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts_gg.card_character_prediction_metric import (
    CardCharacterPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
)
from src.dojos.generic.data_constructors import CardCharacterPredictionDataConstructor
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.loss.soft_classification_loss import SoftClassificationLoss
from src.dojos.sts_gg.card_character_prediction_dojo import CardCharacterPredictionDojo


def _write_source(path: Path) -> None:
    df = pd.DataFrame(
        {
            "nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
            "characters": [["CHARACTER.SILENT"]] * 10,
            "probabilities": [[1.0]] * 10,
            "sample_count": [1] * 10,
        }
    )
    df.to_parquet(path, index=False)


class TestCardCharacterPredictionDojo:
    def test_wires_data_constructor_and_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = CardCharacterPredictionDojo(
            card_binder,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, SingleCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, CardCharacterPredictionDataConstructor)
        assert isinstance(dojo.loss_calculator, SoftClassificationLoss)

    def test_label_values_is_character_prediction_metrics_own_label_values(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = CardCharacterPredictionDojo(
            card_binder,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert dojo.label_values == list(CharacterPredictionMetric.LABEL_VALUES)

    def test_defaults_to_metrics_own_output_path(self) -> None:
        card_binder = CardBinder()

        with pytest.raises(FileNotFoundError) as exc_info:
            CardCharacterPredictionDojo(card_binder, card_embedding_size=4)

        assert str(CardCharacterPredictionMetric.DEFAULT_OUTPUT_PATH) in str(
            exc_info.value
        )
