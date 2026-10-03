"""Tests for data_slice.py's SeventeenLandsSlice."""

import pytest

from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.partition import (
    SeventeenLandsPartition,
)
from src.data_retrieval.seventeenlands.refs import DataType, Expansion, format_code


def _partition(expansion: Expansion, fmt: format_code) -> SeventeenLandsPartition:
    return SeventeenLandsPartition(DataType.GAME, "m", expansion, fmt)


@pytest.mark.parametrize(
    "data_slice, name",
    [
        (SeventeenLandsSlice(), "all"),
        (
            SeventeenLandsSlice(formats=frozenset({format_code.PremierDraft})),
            "formats-PremierDraft",
        ),
        (
            SeventeenLandsSlice(
                expansions=frozenset({Expansion.MSH, Expansion.KTK}),
                formats=frozenset({format_code.TradDraft}),
            ),
            "sets-KTK+MSH_formats-TradDraft",
        ),
    ],
)
def test_name(data_slice: SeventeenLandsSlice, name: str) -> None:
    assert data_slice.name == name


def test_everything_slice_includes_every_partition() -> None:
    assert SeventeenLandsSlice().includes(_partition(Expansion.KTK, format_code.Sealed))


def test_a_slice_must_pass_both_filters() -> None:
    data_slice = SeventeenLandsSlice(
        expansions=frozenset({Expansion.KTK}),
        formats=frozenset({format_code.Sealed}),
    )

    assert data_slice.includes(_partition(Expansion.KTK, format_code.Sealed))
    assert not data_slice.includes(_partition(Expansion.KTK, format_code.TradDraft))
    assert not data_slice.includes(_partition(Expansion.MSH, format_code.Sealed))


@pytest.mark.parametrize("field", ["expansions", "formats"])
def test_an_empty_filter_is_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="is empty"):
        SeventeenLandsSlice(**{field: frozenset()})
