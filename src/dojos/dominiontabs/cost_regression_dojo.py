"""Thin per-metric wrapper over CostRegressionMetric
(src/data_refinement/metrics/dominiontabs/cost_regression_metric.py) -
this project's first MaskedFieldRegressionMetric consumer wired into a
dojo, pairing SingleCardRegressionDojo with
MaskedFieldRegressionDataConstructor (../generic/data_constructors/masked_field_regression.py).

MASKING MOD, train_only=False: cost genuinely lives in raw_content
(unlike SetMaskMetric's cardset_tags - see
dominiontabs/masked_field_dojos.py's SetMaskDojo docstring), so it must
stay masked on test/validation input too, or the model could just read
the answer off raw_content instead of predicting it - same rationale
as every MaskedFieldMetric-family wrapper.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.dominiontabs.cost_regression_metric import (
    CostRegressionMetric,
)
from src.dojos.generic.data_constructors import MaskedFieldRegressionDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec


class CostRegressionDojo(SingleCardRegressionDojo):
    """Card, cost masked -> predicted coin cost (CostRegressionMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or CostRegressionMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldRegressionDataConstructor("label"),
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=CostRegressionMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
            strict_version_check=strict_version_check,
        )
