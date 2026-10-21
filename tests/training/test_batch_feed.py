"""BatchFeed: the direct feed follows the round's schedule and refuses
calls out of it; the factory picks the variant by depth."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.training.diet.batch_feed import DirectBatchFeed, open_batch_feed
from src.training.diet.batch_feed_errors import ScheduleMismatchError
from src.training.diet.dojo_batch_stream import DojoBatchStream
from src.training.diet.prefetching_batch_streams import (
    NO_BATCH_WARMING,
    PrefetchingBatchStreams,
)
from src.training.diet.round_schedule import RoundSchedule


class _NumberedStream:
    """Fake DojoBatchStream: numbered batches forever."""

    def __init__(self) -> None:
        self.calls = 0

    def next_batch(self) -> Any:
        self.calls += 1
        return SimpleNamespace(inputs=[], tag=self.calls)


def _streams(*names: str) -> dict[str, DojoBatchStream]:
    return {name: cast(DojoBatchStream, _NumberedStream()) for name in names}


def _schedule(*names: str) -> RoundSchedule:
    return RoundSchedule(names)


class TestDirectBatchFeed:
    def test_next_batch_takes_each_scheduled_dojo_s_batches_in_order(self) -> None:
        feed = DirectBatchFeed(_streams("a", "b"))
        feed.begin_round(_schedule("a", "a", "b", "a"))
        tags = [feed.next_batch(name).tag for name in ("a", "a", "b", "a")]
        assert tags == [1, 2, 1, 3]

    def test_a_call_out_of_schedule_is_refused(self) -> None:
        feed = DirectBatchFeed(_streams("a", "b"))
        with pytest.raises(ScheduleMismatchError, match="no round open"):
            feed.next_batch("a")
        feed.begin_round(_schedule("a"))
        with pytest.raises(ScheduleMismatchError, match="next scheduled dojo is 'a'"):
            feed.next_batch("b")
        feed.next_batch("a")
        with pytest.raises(ScheduleMismatchError, match="past the round"):
            feed.next_batch("a")
        with pytest.raises(ScheduleMismatchError, match="last round ended"):
            feed.begin_round(_schedule("a"))

    def test_reschedule_replaces_the_rest_of_the_round(self) -> None:
        feed = DirectBatchFeed(_streams("a", "b"))
        feed.begin_round(_schedule("a", "a", "a"))
        feed.next_batch("a")
        feed.reschedule(_schedule("b", "b"))
        assert [feed.next_batch("b").tag for _ in range(2)] == [1, 2]

    def test_end_round_opens_the_way_for_the_next_and_builds_nothing(self) -> None:
        streams = _streams("a")
        feed = DirectBatchFeed(streams)
        feed.end_round()  # no round: nothing to do
        feed.begin_round(_schedule("a", "a"))
        feed.end_round()
        with pytest.raises(ScheduleMismatchError, match="reschedule"):
            feed.reschedule(_schedule("a"))
        feed.begin_round(_schedule("a"))
        assert feed.next_batch("a").tag == 1
        assert cast(_NumberedStream, streams["a"]).calls == 1
        feed.close()
        feed.close()


class TestOpenBatchFeed:
    def test_depth_zero_builds_on_the_calling_thread(self) -> None:
        feed = open_batch_feed(_streams("a"), NO_BATCH_WARMING, 0)
        assert isinstance(feed, DirectBatchFeed)

    def test_a_positive_depth_starts_a_prefetching_feed(self) -> None:
        feed = open_batch_feed(_streams("a"), NO_BATCH_WARMING, 2)
        try:
            assert isinstance(feed, PrefetchingBatchStreams)
            feed.begin_round(_schedule("a"))  # started: no PrefetchStoppedError
            assert feed.next_batch("a").tag == 1
        finally:
            feed.close()

    def test_a_negative_depth_is_refused(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            open_batch_feed(_streams("a"), NO_BATCH_WARMING, -1)
