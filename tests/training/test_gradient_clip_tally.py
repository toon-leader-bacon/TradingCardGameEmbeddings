import logging

import pytest

from src.training.gradient_clip_tally import GradientClipTally


def _logged(tally: GradientClipTally, caplog: pytest.LogCaptureFixture) -> str:
    with caplog.at_level(logging.INFO):
        tally.log_and_reset("joint", 2)
    return caplog.text


def test_logs_the_clipped_share_and_the_most_clipped_dojos(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tally = GradientClipTally()
    for norm in (3.0, 2.0, 0.5):
        tally.record("a", norm, max_norm=1.0)
    tally.record("b", 5.0, max_norm=1.0)
    tally.record("c", 0.1, max_norm=1.0)

    text = _logged(tally, caplog)

    assert (
        "[joint] round 2: gradients clipped on 3 of 5 step(s); most: a 2/3, b 1/1"
        in text
    )
    assert "c 0/1" not in text


def test_ties_on_clipped_count_go_to_the_higher_clip_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tally = GradientClipTally()
    tally.record("often", 2.0, max_norm=1.0)
    tally.record("rarely", 2.0, max_norm=1.0)
    tally.record("rarely", 0.5, max_norm=1.0)
    assert "most: often 1/1, rarely 1/2" in _logged(tally, caplog)


def test_lists_at_most_max_listed_dojos(caplog: pytest.LogCaptureFixture) -> None:
    tally = GradientClipTally(max_listed_dojos=1)
    tally.record("a", 2.0, max_norm=1.0)
    tally.record("a", 2.0, max_norm=1.0)
    tally.record("b", 2.0, max_norm=1.0)
    text = _logged(tally, caplog)
    assert "most: a 2/2" in text
    assert "b 1/1" not in text


def test_a_round_with_no_clips_names_no_dojo(caplog: pytest.LogCaptureFixture) -> None:
    tally = GradientClipTally()
    tally.record("a", 0.5, max_norm=1.0)
    text = _logged(tally, caplog)
    assert "gradients clipped on 0 of 1 step(s)" in text
    assert "most:" not in text


def test_a_non_finite_norm_is_not_counted(caplog: pytest.LogCaptureFixture) -> None:
    tally = GradientClipTally()
    tally.record("a", float("inf"), max_norm=1.0)
    tally.record("a", float("nan"), max_norm=1.0)
    assert "gradients clipped" not in _logged(tally, caplog)


def test_counts_reset_after_each_round(caplog: pytest.LogCaptureFixture) -> None:
    tally = GradientClipTally()
    tally.record("a", 2.0, max_norm=1.0)
    _logged(tally, caplog)
    caplog.clear()
    assert "gradients clipped" not in _logged(tally, caplog)


def test_max_listed_dojos_must_be_positive() -> None:
    with pytest.raises(ValueError):
        GradientClipTally(max_listed_dojos=0)
