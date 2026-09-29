import pytest

from src.evaluation.chart_theme import REFERENCE_CATEGORICAL, ChartTheme


def test_defaults_are_the_reference_eight() -> None:
    assert ChartTheme().categorical == REFERENCE_CATEGORICAL
    assert len(REFERENCE_CATEGORICAL) == 8


@pytest.mark.parametrize("overrides", [{"categorical": ()}, {"dpi": 0}])
def test_invalid_fields_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        ChartTheme(**overrides)
