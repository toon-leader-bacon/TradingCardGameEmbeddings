import threading
from typing import Iterator

import pytest

from src.utils.stage_clock import ProfileStage, StageClock, StageTotals


@pytest.fixture
def clock() -> Iterator[StageClock]:
    result = StageClock()
    result.enable(synchronize_cuda=False)
    yield result


class TestStageTotals:
    def test_zero_has_every_stage(self) -> None:
        totals = StageTotals.zero()
        assert all(totals.seconds_in(stage) == 0.0 for stage in ProfileStage)
        assert totals.live_texts == 0

    def test_a_missing_stage_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="lacks"):
            StageTotals.of({ProfileStage.MODS: 1.0}, 0)

    def test_minus_gives_the_difference(self) -> None:
        later = StageTotals.of({stage: 3.0 for stage in ProfileStage}, 10)
        earlier = StageTotals.of({stage: 1.0 for stage in ProfileStage}, 4)
        difference = later.minus(earlier)
        assert difference.seconds_in(ProfileStage.MODS) == 2.0
        assert difference.live_texts == 6


class TestStageClock:
    def test_disabled_times_nothing(self) -> None:
        clock = StageClock()
        with clock.timed(ProfileStage.MODS):
            pass
        clock.count_live_texts(5)
        assert clock.totals() == StageTotals.zero()

    def test_times_a_stage(self, clock: StageClock) -> None:
        with clock.timed(ProfileStage.MODS):
            sum(range(10_000))
        assert clock.totals().seconds_in(ProfileStage.MODS) > 0
        assert clock.totals().seconds_in(ProfileStage.SERIALIZATION) == 0

    def test_a_nested_stage_counts_once(self, clock: StageClock) -> None:
        with clock.timed(ProfileStage.MODS):
            with clock.timed(ProfileStage.MODS):
                sum(range(10_000))
            inner_only = clock.totals().seconds_in(ProfileStage.MODS)
        # Nothing was added when the inner call closed
        assert inner_only == 0.0
        assert clock.totals().seconds_in(ProfileStage.MODS) > 0

    def test_time_is_added_when_the_body_raises(self, clock: StageClock) -> None:
        with pytest.raises(RuntimeError):
            with clock.timed(ProfileStage.LIVE_ENCODING, queues_gpu_work=True):
                raise RuntimeError("boom")
        assert clock.totals().seconds_in(ProfileStage.LIVE_ENCODING) > 0

    def test_counts_live_texts(self, clock: StageClock) -> None:
        clock.count_live_texts(3)
        clock.count_live_texts(4)
        assert clock.totals().live_texts == 7

    def test_totals_are_per_thread(self, clock: StageClock) -> None:
        clock.count_live_texts(3)
        seen: list[int] = []
        worker = threading.Thread(target=lambda: seen.append(clock.totals().live_texts))
        worker.start()
        worker.join()
        assert seen == [0]

    def test_totals_are_a_copy(self, clock: StageClock) -> None:
        before = clock.totals()
        with clock.timed(ProfileStage.MODS):
            sum(range(10_000))
        assert before.seconds_in(ProfileStage.MODS) == 0.0
