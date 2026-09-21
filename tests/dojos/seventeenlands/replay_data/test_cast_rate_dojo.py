from pathlib import Path

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.seventeenlands.replay_data.cast_rate_dojo import CastRateDojo
from src.schema.holdout import HoldoutSpec


def _write_source(path: Path) -> None:
    df = pd.DataFrame({"nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df["cast_rate"] = 0.7
    df["sample_count"] = 10
    df.to_parquet(path, index=False)


class TestCastRateDojo:
    def test_wires_data_constructor_to_label_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source)
        card_binder = CardBinder()

        dojo = CastRateDojo(
            card_binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, SingleCardRegressionDojo)
        assert isinstance(dojo.data_constructor, CardAverageDataConstructor)
        assert dojo.data_constructor._label_column == "cast_rate"
