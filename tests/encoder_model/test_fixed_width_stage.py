"""FixedWidthStage: the widths every pipeline stage reports."""

import pytest

from src.encoder_model.fixed_width_stage import FixedWidthStage


def test_a_stage_reports_the_widths_it_was_built_with() -> None:
    stage = FixedWidthStage(768, 256)
    assert (stage.input_dim, stage.output_dim) == (768, 256)


@pytest.mark.parametrize("widths", [(0, 1), (1, 0), (-3, 4)])
def test_a_width_below_one_is_refused(widths: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="widths must be >= 1"):
        FixedWidthStage(*widths)
