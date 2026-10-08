import math

import pytest

from src.training.loss_weighting import DEFAULT_BASELINE_FLOOR, LossWeighting
from tests.training.fakes import FakeBatch, FakeDojo


def _batch() -> FakeBatch:
    return FakeBatch(inputs=None)


class TestScaleFor:
    def test_is_the_weight_over_the_baseline(self) -> None:
        weighting = LossWeighting({"a": 2.0}, baseline_floor=0.01)
        assert weighting.scale_for(FakeDojo("a", baseline=4.0), _batch()) == 0.5

    def test_an_unnamed_dojo_counts_one(self) -> None:
        weighting = LossWeighting({"a": 2.0})
        assert weighting.scale_for(FakeDojo("b", baseline=0.5), _batch()) == 2.0

    def test_a_baseline_below_the_floor_is_treated_as_the_floor(self) -> None:
        weighting = LossWeighting({"a": 0.5}, baseline_floor=0.1)
        assert weighting.scale_for(FakeDojo("a", baseline=1e-6), _batch()) == 5.0

    def test_a_batch_with_no_usable_baseline_gives_none(self) -> None:
        dojo = FakeDojo("a", unusable_baseline=True)
        assert LossWeighting({}).scale_for(dojo, _batch()) is None

    def test_equalizes_dojos_on_different_scales(self) -> None:
        # Untrained losses equal to their baselines both scale to the weight
        weighting = LossWeighting({})
        small, large = FakeDojo("s", baseline=0.69), FakeDojo("l", baseline=6.9)
        scaled_small = 0.69 * (weighting.scale_for(small, _batch()) or 0.0)
        scaled_large = 6.9 * (weighting.scale_for(large, _batch()) or 0.0)
        assert scaled_small == pytest.approx(1.0)
        assert scaled_large == pytest.approx(1.0)


class TestConstruction:
    def test_the_floor_defaults(self) -> None:
        assert LossWeighting({}).baseline_floor == DEFAULT_BASELINE_FLOOR

    @pytest.mark.parametrize("weight", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_a_bad_weight(self, weight: float) -> None:
        with pytest.raises(ValueError, match="loss weight for 'a'"):
            LossWeighting({"a": weight})

    @pytest.mark.parametrize("floor", [0.0, -0.1, math.nan, math.inf])
    def test_rejects_a_bad_floor(self, floor: float) -> None:
        with pytest.raises(ValueError, match="baseline_floor"):
            LossWeighting({}, baseline_floor=floor)

    def test_weights_are_a_read_only_copy(self) -> None:
        source = {"a": 2.0}
        weighting = LossWeighting(source)
        source["a"] = 9.0
        assert weighting.weights["a"] == 2.0
        with pytest.raises(TypeError):
            weighting.weights["a"] = 3.0  # type: ignore[index]
