from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.isotropic.card_rate_dojos import (
    AverageCopiesBoughtDojo,
    OpeningBuyRateDojo,
    PileExhaustionRateDojo,
    TurnCountAssociationDojo,
    VetoRateDojo,
)
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")

# (dojo class, label column, metric output file stem)
_CASES = [
    (AverageCopiesBoughtDojo, "average_copies_bought", "average_copies_bought"),
    (TurnCountAssociationDojo, "turn_count_delta", "turn_count_association"),
    (OpeningBuyRateDojo, "opening_buy_rate", "opening_buy_rate"),
    (PileExhaustionRateDojo, "pile_exhaustion_rate", "pile_exhaustion_rate"),
    (VetoRateDojo, "veto_rate", "veto_rate"),
]


def _write_source(path: Path, label_column: str) -> None:
    df = pd.DataFrame({"nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df[label_column] = 1.0
    df["sample_count"] = 100
    df.to_parquet(path, index=False)


class TestIsotropicCardRateDojos:
    @pytest.mark.parametrize("dojo_cls,label_column,output_stem", _CASES)
    def test_wires_data_constructor_to_metrics_label_column(
        self, dojo_cls, label_column, output_stem, tmp_path: Path
    ) -> None:
        source = tmp_path / "isotropic_card_rate_source.parquet"
        _write_source(source, label_column)

        dojo = dojo_cls(
            CardBinder(),
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, SingleCardRegressionDojo)
        assert isinstance(dojo.data_constructor, CardAverageDataConstructor)
        assert dojo.data_constructor._label_column == label_column

    @pytest.mark.parametrize("dojo_cls,label_column,output_stem", _CASES)
    def test_default_output_path_is_the_metrics_own(
        self, dojo_cls, label_column, output_stem
    ) -> None:
        assert dojo_cls.METRIC.DEFAULT_OUTPUT_PATH == Path(
            f"data/metrics/isotropic/{output_stem}.parquet"
        )
