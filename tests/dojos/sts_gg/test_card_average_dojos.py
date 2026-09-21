from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts_gg.card_average_metrics import (
    CardDeckSizeMetric,
    CardElitesKilledMetric,
    CardFloorsClearedMetric,
    CardRelicCountMetric,
    CardTotalCardsPickedMetric,
    CardTotalCombatsMetric,
    CardTotalDamageTakenMetric,
    CardTotalTurnsMetric,
    CardWinRateMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.sts_gg.card_average_dojos import (
    CardDeckSizeDojo,
    CardElitesKilledDojo,
    CardFloorsClearedDojo,
    CardRelicCountDojo,
    CardTotalCardsPickedDojo,
    CardTotalCombatsDojo,
    CardTotalDamageTakenDojo,
    CardTotalTurnsDojo,
    CardWinRateDojo,
)
from src.schema.holdout import HoldoutSpec

_CASES = [
    (CardRelicCountDojo, CardRelicCountMetric),
    (CardTotalDamageTakenDojo, CardTotalDamageTakenMetric),
    (CardDeckSizeDojo, CardDeckSizeMetric),
    (CardTotalCardsPickedDojo, CardTotalCardsPickedMetric),
    (CardTotalTurnsDojo, CardTotalTurnsMetric),
    (CardElitesKilledDojo, CardElitesKilledMetric),
    (CardFloorsClearedDojo, CardFloorsClearedMetric),
    (CardTotalCombatsDojo, CardTotalCombatsMetric),
    (CardWinRateDojo, CardWinRateMetric),
]


def _write_source(path: Path, label_column: str) -> None:
    df = pd.DataFrame({"nocab_uuid": ["00000000-0000-0000-0000-000000000000"] * 10})
    df[label_column] = 1.0
    df.to_parquet(path, index=False)


class TestCardAverageDojoWrappers:
    @pytest.mark.parametrize("dojo_cls,metric_cls", _CASES)
    def test_wires_data_constructor_to_metrics_label_column(
        self, dojo_cls, metric_cls, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, metric_cls.LABEL_COLUMN)
        card_binder = CardBinder()

        dojo = dojo_cls(
            card_binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
        )

        assert isinstance(dojo, SingleCardRegressionDojo)
        assert isinstance(dojo.data_constructor, CardAverageDataConstructor)
        assert dojo.data_constructor._label_column == metric_cls.LABEL_COLUMN
