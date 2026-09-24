"""Thin wrapper over PickNumberDecayCurveMetric
(src/data_refinement/metrics/seventeenlands/draft_data/pick_number_decay_curve_metric.py).

Unlike every other wrapper in this package, this one is NOT a plain
SingleCardRegressionDojo/SingleCardFixedClassificationDojo consumer:
its target (take_rate_by_pick_number) is a fixed-width vector of
independent per-bucket probabilities, not one scalar or one
mutually-exclusive class. It still reuses
SingleCardFixedClassificationDojo mechanically though - single card in,
MAX_BUCKET_COUNT raw logits out is exactly FixedClassificationDecoderHead's
shape - by injecting two new collaborators via that cell's existing
extension seams (see plans/seventeen_lands_dojos.md):
    - data_constructor=PickNumberDecayCurveDataConstructor, which reads
      the metric's two parallel list columns into a sparse
      bucket_index -> take_rate dict, masking out any bucket with fewer
      than min_sample_count samples.
    - loss_factory=MaskedVectorRegressionLoss, which sigmoids each
      output position independently and scores it against that sparse
      dict (masked MSE) instead of FixedClassificationLoss's default
      softmax cross-entropy - see that class's own module docstring for
      why softmax would be wrong here.
label_values=[str(i) for i in range(MAX_BUCKET_COUNT)] is a degenerate
vocabulary (bucket indices, not real category names) - required by
SingleCardFixedClassificationDojo's constructor shape, but only its
length is ever read (by MaskedVectorRegressionLoss - see that class's
own docstring).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pick_number_decay_curve_metric import (  # noqa: E501
    PickNumberDecayCurveMetric,
)
from src.dojos.generic.data_constructors import PickNumberDecayCurveDataConstructor
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss
from src.schema.holdout import HoldoutSpec

MIN_SAMPLE_COUNT = 10


class PickNumberDecayCurveDojo(SingleCardFixedClassificationDojo):
    """Card -> per-bucket P(picked | in pack, at pick_number bucket i)
    (PickNumberDecayCurveMetric). See module docstring for why this
    wrapper injects MaskedVectorRegressionLoss instead of this cell's
    default FixedClassificationLoss."""

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
            or PickNumberDecayCurveMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=PickNumberDecayCurveDataConstructor(
                min_sample_count=MIN_SAMPLE_COUNT
            ),
            label_values=[
                str(bucket)
                for bucket in range(PickNumberDecayCurveMetric.MAX_BUCKET_COUNT)
            ],
            card_embedding_size=card_embedding_size,
            loss_factory=MaskedVectorRegressionLoss,
            rng_seed=rng_seed,
            strict_version_check=strict_version_check,
        )
