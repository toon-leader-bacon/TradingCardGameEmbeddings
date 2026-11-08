import pytest

from src.data_refinement.deck_box.deck_sizes import DeckSizes


class TestDeckSizes:
    def test_bounds_are_inclusive(self) -> None:
        sizes = DeckSizes(50, 55)
        assert [sizes.fits(n) for n in (49, 50, 55, 56)] == [False, True, True, False]

    def test_no_upper_bound(self) -> None:
        sizes = DeckSizes(60, None)
        assert not sizes.fits(59) and sizes.fits(60) and sizes.fits(80)

    def test_an_exact_size(self) -> None:
        sizes = DeckSizes(51, 51)
        assert [sizes.fits(n) for n in (50, 51, 52)] == [False, True, False]

    @pytest.mark.parametrize("bounds", [(0, 10), (60, 59)])
    def test_impossible_bounds_raise(self, bounds: tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            DeckSizes(*bounds)
