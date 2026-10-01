import math

import pytest

from src.dojos.loss.label_stats import LabelStats


class TestFromLabels:
    def test_population_mean_and_std(self) -> None:
        assert LabelStats.from_labels([1.0, 3.0]) == LabelStats(mean=2.0, std=1.0)

    def test_standardized_labels_have_unit_mean_square(self) -> None:
        labels = [450.0, 300.0, 610.0, 512.0]
        stats = LabelStats.from_labels(labels)
        standardized = stats.standardize(labels)
        assert math.isclose(sum(standardized), 0.0, abs_tol=1e-9)
        assert math.isclose(sum(z * z for z in standardized) / len(labels), 1.0)

    @pytest.mark.parametrize("labels", [[], [1.0, math.nan], [math.inf, 1.0]])
    def test_rejects_empty_or_non_finite_labels(self, labels: list[float]) -> None:
        with pytest.raises(ValueError):
            LabelStats.from_labels(labels)

    @pytest.mark.parametrize("labels", [[5.0], [0.1, 0.1, 0.1]])
    def test_a_constant_label_is_an_error(self, labels: list[float]) -> None:
        with pytest.raises(ValueError, match="zero standard deviation"):
            LabelStats.from_labels(labels)


class TestConversions:
    def test_standardize(self) -> None:
        assert LabelStats(2.0, 1.0).standardize([1.0, 3.0]) == [-1.0, 1.0]

    def test_to_label_units_inverts_standardize(self) -> None:
        stats = LabelStats(mean=450.0, std=25.0)
        assert stats.to_label_units(-1.0) == 425.0
        assert stats.to_label_units(stats.standardize([500.0])[0]) == 500.0

    def test_standardize_of_nothing_is_nothing(self) -> None:
        assert LabelStats(0.0, 1.0).standardize([]) == []
