from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_pool_metric import (
    TutorTargetPoolMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.seventeenlands.game_data.tutor_target_pool_dojo import (
    TutorTargetPoolDojo,
)


def _write_source(path: Path) -> None:
    df = pd.DataFrame({"pool_card_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df["tutored"] = True
    df.to_parquet(path, index=False)


class TestTutorTargetPoolDojo:
    def test_wires_data_constructor_to_pool_card_uuid_column(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = TutorTargetPoolDojo(
            card_binder,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, SingleCardRegressionDojo)
        assert isinstance(dojo.data_constructor, CardAverageDataConstructor)
        assert dojo.data_constructor._label_column == "tutored"
        assert dojo.data_constructor._uuid_column == "pool_card_uuid"

    def test_defaults_to_metrics_own_output_path(self) -> None:
        card_binder = CardBinder()

        with pytest.raises(FileNotFoundError) as exc_info:
            TutorTargetPoolDojo(card_binder, card_embedding_size=4)

        assert str(TutorTargetPoolMetric.DEFAULT_OUTPUT_PATH) in str(exc_info.value)
