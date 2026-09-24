from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pick_number_decay_curve_metric import (  # noqa: E501
    PickNumberDecayCurveMetric,
)
from src.dojos.generic.data_constructors import PickNumberDecayCurveDataConstructor
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss
from src.dojos.seventeenlands.draft_data.pick_number_decay_curve_dojo import (
    MIN_SAMPLE_COUNT,
    PickNumberDecayCurveDojo,
)
from src.schema.holdout import HoldoutSpec


def _write_source(path: Path) -> None:
    bucket_count = PickNumberDecayCurveMetric.MAX_BUCKET_COUNT
    df = pd.DataFrame({"nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 2})
    df["take_rate_by_pick_number"] = [[0.5] * bucket_count] * 2
    df["sample_count_by_pick_number"] = [[10] * bucket_count] * 2
    df.to_parquet(path, index=False)


class TestPickNumberDecayCurveDojo:
    def test_wires_data_constructor_and_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = PickNumberDecayCurveDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, SingleCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, PickNumberDecayCurveDataConstructor)
        assert dojo.data_constructor._min_sample_count == MIN_SAMPLE_COUNT
        assert isinstance(dojo.loss_calculator, MaskedVectorRegressionLoss)
        assert len(dojo.label_values) == PickNumberDecayCurveMetric.MAX_BUCKET_COUNT
