"""PrefetchingBatchStreams: scheduled batches arrive in schedule and stream
order, errors in their place; the worker never builds past depth or the
round; reservations are released as steps move on; a reschedule keeps
what matches and saves the rest as leftovers; the lifecycle never blocks
forever; build_model_warmer refuses a misconfigured model."""

import logging
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, cast
from uuid import uuid4

import pytest
import torch

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.pooled_feature_cache import (
    FeatureReservation,
    PooledFeatureCache,
)
from src.encoder_model.feature_cache.slot_policy import LeastRecentlyUsed
from src.encoder_model.feature_cache.tensor_feature_slab import (
    SlabPlacement,
    TensorFeatureSlab,
)
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore
from src.encoder_model.precision import Precision
from src.encoder_model.single_card_model import SingleCardModel
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.training.diet.batch_feed_errors import (
    PrefetchStoppedError,
    ScheduleMismatchError,
)
from src.training.diet.dojo_batch_stream import DojoBatchStream
from src.training.diet.prefetching_batch_streams import (
    NO_BATCH_WARMING,
    PrefetchingBatchStreams,
    _Lifecycle,
    build_model_warmer,
)
from src.training.diet.round_schedule import RoundSchedule
from tests.encoder_model.fakes import HIDDEN, fake_card_pipeline
from tests.encoder_model.feature_cache.cache_fakes import fingerprint, memory_cache


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            DataSource.SCRYFALL, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


class _ScriptedStream:
    """Fake DojoBatchStream: hands out its script in order (an Exception
    entry is raised in its place), then numbered batches forever. Counts
    its calls."""

    def __init__(self, name: str, script: list[Any] | None = None) -> None:
        self.name = name
        self._script = list(script or [])
        self._count = 0
        self.calls = 0
        self.gate: threading.Event | None = None  # set to block next_batch
        self.entered = threading.Event()  # set once next_batch is running

    def next_batch(self) -> Any:
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            self.gate.wait()
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        self._count += 1
        return SimpleNamespace(
            inputs=[_card(f"{self.name}{self._count}")], tag=self._count
        )


class _RecordingWarmer:
    """Fake BatchWarmer: a reservation per call, named after its cards;
    tracks which are outstanding and refuses a double release."""

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self._lock = threading.Lock()
        self.reserved: list[str] = []
        self.outstanding: dict[int, FeatureReservation] = {}

    def reserve(self, cards: list[GenericCard]) -> FeatureReservation:
        if self._fail:
            raise RuntimeError("out of memory")
        result = FeatureReservation(tuple(CardTextKey.of_text(c.name) for c in cards))
        with self._lock:
            self.reserved.extend(card.name for card in cards)
            self.outstanding[id(result)] = result
        return result

    def release(self, reservation: FeatureReservation) -> None:
        if not reservation.keys:
            return
        with self._lock:
            if self.outstanding.pop(id(reservation), None) is None:
                raise ValueError("released twice")


class _Boom(BaseException):
    """Not an Exception: escapes the build and kills the worker loop."""


def _streams(*streams: _ScriptedStream) -> dict[str, DojoBatchStream]:
    return {stream.name: cast(DojoBatchStream, stream) for stream in streams}


def _schedule(*names: str) -> RoundSchedule:
    return RoundSchedule(names)


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.01)


def _ready_count(prefetch: PrefetchingBatchStreams) -> int:
    return len(prefetch._ready)  # noqa: SLF001 - observing the worker


def _idle(prefetch: PrefetchingBatchStreams) -> bool:
    return prefetch._in_flight is None  # noqa: SLF001 - observing the worker


def _blocked_consumer(
    prefetch: PrefetchingBatchStreams, name: str
) -> tuple[threading.Thread, list[BaseException]]:
    """A thread blocked in next_batch(name), and where its error lands."""
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            prefetch.next_batch(name)
        except BaseException as error:  # noqa: B036 - the test inspects it
            errors.append(error)

    thread = threading.Thread(target=consume)
    thread.start()
    return thread, errors


class TestScheduledBuilding:
    def test_batches_arrive_in_schedule_and_stream_order_and_are_reserved(
        self,
    ) -> None:
        warmer = _RecordingWarmer()
        a, b = _ScriptedStream("a"), _ScriptedStream("b")
        with PrefetchingBatchStreams(_streams(a, b), warmer, 2) as prefetch:
            prefetch.begin_round(_schedule("a", "b", "a", "a"))
            tags = [prefetch.next_batch(n).tag for n in ("a", "b", "a", "a")]
            assert tags == [1, 1, 2, 3]
            prefetch.end_round()
        assert warmer.reserved == ["a1", "b1", "a2", "a3"]

    def test_the_worker_stays_within_depth_and_the_round(self) -> None:
        a = _ScriptedStream("a")
        with PrefetchingBatchStreams(_streams(a), NO_BATCH_WARMING, 2) as prefetch:
            prefetch.begin_round(_schedule("a", "a", "a"))
            _wait_until(lambda: _ready_count(prefetch) == 2)
            time.sleep(0.05)
            assert a.calls == 2  # depth 2: no third build yet
            prefetch.next_batch("a")
            _wait_until(lambda: _ready_count(prefetch) == 2)
            prefetch.next_batch("a")
            prefetch.next_batch("a")
            time.sleep(0.05)
            assert a.calls == 3  # never past the round's three steps

    def test_a_stream_error_arrives_in_its_place_and_the_stream_goes_on(self) -> None:
        first = SimpleNamespace(inputs=[_card("x")], tag="first")
        streams = _streams(_ScriptedStream("a", [first, ValueError("bad row")]))
        with PrefetchingBatchStreams(streams, NO_BATCH_WARMING, 3) as prefetch:
            prefetch.begin_round(_schedule("a", "a", "a"))
            assert prefetch.next_batch("a").tag == "first"
            with pytest.raises(ValueError, match="bad row"):
                prefetch.next_batch("a")
            assert prefetch.next_batch("a").tag == 1

    def test_calls_out_of_schedule_are_refused(self) -> None:
        with PrefetchingBatchStreams(
            _streams(_ScriptedStream("a"), _ScriptedStream("b")), NO_BATCH_WARMING, 1
        ) as prefetch:
            with pytest.raises(ScheduleMismatchError, match="no round open"):
                prefetch.next_batch("a")
            with pytest.raises(ScheduleMismatchError, match="reschedule"):
                prefetch.reschedule(_schedule("a"))
            prefetch.begin_round(_schedule("a"))
            with pytest.raises(ScheduleMismatchError, match="last round ended"):
                prefetch.begin_round(_schedule("a"))
            with pytest.raises(ScheduleMismatchError, match="is 'a'"):
                prefetch.next_batch("b")
            prefetch.next_batch("a")
            with pytest.raises(ScheduleMismatchError, match="past the round"):
                prefetch.next_batch("a")


class TestReservations:
    def test_each_step_s_pins_are_released_when_the_next_is_taken(self) -> None:
        warmer = _RecordingWarmer()
        with PrefetchingBatchStreams(
            _streams(_ScriptedStream("a")), warmer, 1
        ) as prefetch:
            prefetch.begin_round(_schedule("a", "a", "a"))
            for _ in range(3):
                prefetch.next_batch("a")
                # The step just taken, plus at most depth built ahead
                assert len(warmer.outstanding) <= 2
            prefetch.end_round()
            assert warmer.outstanding == {}

    def test_close_releases_everything_held(self) -> None:
        warmer = _RecordingWarmer()
        prefetch = PrefetchingBatchStreams(_streams(_ScriptedStream("a")), warmer, 3)
        prefetch.start()
        prefetch.begin_round(_schedule("a", "a", "a", "a"))
        prefetch.next_batch("a")
        _wait_until(lambda: _ready_count(prefetch) == 3)
        prefetch.close()
        assert warmer.outstanding == {}

    def test_a_warming_failure_is_counted_and_the_batch_still_delivered(
        self,
    ) -> None:
        with PrefetchingBatchStreams(
            _streams(_ScriptedStream("a")), _RecordingWarmer(fail=True), 1
        ) as prefetch:
            prefetch.begin_round(_schedule("a", "a"))
            assert prefetch.next_batch("a").tag == 1
            assert prefetch.next_batch("a").tag == 2
            assert prefetch.warm_failures == 2


class TestRescheduleAndLeftovers:
    def test_unmatched_built_steps_are_used_later_in_stream_order(self) -> None:
        warmer = _RecordingWarmer()
        a, b = _ScriptedStream("a"), _ScriptedStream("b")
        with PrefetchingBatchStreams(_streams(a, b), warmer, 3) as prefetch:
            prefetch.begin_round(_schedule("a", "b", "a", "b"))
            _wait_until(lambda: _ready_count(prefetch) == 3)  # a1 b1 a2
            # The trainer redraws the whole rest: only "b" steps now
            prefetch.reschedule(_schedule("b", "b", "b"))
            assert [prefetch.next_batch("b").tag for _ in range(3)] == [1, 2, 3]
            prefetch.end_round()
            assert warmer.outstanding == {}
            # a1 and a2 wait as leftovers and come back first, in order
            prefetch.begin_round(_schedule("a", "a", "a"))
            assert [prefetch.next_batch("a").tag for _ in range(3)] == [1, 2, 3]
            prefetch.end_round()
        assert (a.calls, b.calls) == (3, 3)
        # A leftover is reserved again before it is used
        assert warmer.reserved.count("a1") == 2

    def test_a_matching_prefix_is_kept(self) -> None:
        a, b = _ScriptedStream("a"), _ScriptedStream("b")
        with PrefetchingBatchStreams(_streams(a, b), NO_BATCH_WARMING, 2) as prefetch:
            prefetch.begin_round(_schedule("a", "b", "b"))
            _wait_until(lambda: _ready_count(prefetch) == 2)  # a1 b1
            prefetch.reschedule(_schedule("a", "a"))
            assert [prefetch.next_batch("a").tag for _ in range(2)] == [1, 2]
            prefetch.end_round()
            prefetch.begin_round(_schedule("b"))
            assert prefetch.next_batch("b").tag == 1  # the dropped b1
            prefetch.end_round()

    @pytest.mark.parametrize(
        ("new_schedule", "in_flight_kept"),
        [(("a", "b"), True), (("b", "b"), False)],
    )
    def test_a_build_in_flight_is_kept_only_if_it_still_matches(
        self, new_schedule: tuple[str, ...], in_flight_kept: bool
    ) -> None:
        warmer = _RecordingWarmer()
        a, b = _ScriptedStream("a"), _ScriptedStream("b")
        b.gate = threading.Event()
        prefetch = PrefetchingBatchStreams(_streams(a, b), warmer, 2)
        prefetch.start()
        prefetch.begin_round(_schedule("a", "b", "a"))
        assert b.entered.wait(timeout=5)  # a1 built, b1 in flight
        prefetch.reschedule(RoundSchedule(new_schedule))
        b.gate.set()
        for name in new_schedule:
            prefetch.next_batch(name)
        prefetch.end_round()
        # Kept or saved as a leftover, b1 is built once and never skipped
        assert b.calls == (1 if in_flight_kept else 2)
        assert warmer.reserved.count("b1") == (1 if in_flight_kept else 2)
        assert warmer.outstanding == {}
        prefetch.close()

    def test_a_queued_build_failure_keeps_its_place_as_a_leftover(self) -> None:
        a = _ScriptedStream("a", [ValueError("bad row")])
        b = _ScriptedStream("b")
        with PrefetchingBatchStreams(_streams(a, b), NO_BATCH_WARMING, 3) as prefetch:
            prefetch.begin_round(_schedule("a", "a", "b"))
            _wait_until(lambda: _ready_count(prefetch) == 3)  # error, a1, b1
            prefetch.reschedule(_schedule("b"))
            prefetch.next_batch("b")
            prefetch.end_round()
            prefetch.begin_round(_schedule("a", "a"))
            with pytest.raises(ValueError, match="bad row"):
                prefetch.next_batch("a")
            assert prefetch.next_batch("a").tag == 1
            prefetch.end_round()

    def test_end_round_waits_for_the_build_in_flight_and_keeps_it(self) -> None:
        stream = _ScriptedStream("a")
        stream.gate = threading.Event()
        prefetch = PrefetchingBatchStreams(_streams(stream), NO_BATCH_WARMING, 1)
        prefetch.start()
        prefetch.begin_round(_schedule("a"))
        assert stream.entered.wait(timeout=5)
        ender = threading.Thread(target=prefetch.end_round)
        ender.start()
        ender.join(timeout=0.2)
        assert ender.is_alive()  # the worker is still inside next_batch
        stream.gate.set()
        ender.join(timeout=5)
        assert not ender.is_alive()
        assert _idle(prefetch)
        prefetch.begin_round(_schedule("a"))
        assert prefetch.next_batch("a").tag == 1
        assert stream.calls == 1  # the leftover, not a new build
        prefetch.close()


class TestLifecycle:
    def test_begin_round_before_start_or_after_close_raises(self) -> None:
        prefetch = PrefetchingBatchStreams(
            _streams(_ScriptedStream("a")), NO_BATCH_WARMING, 1
        )
        with pytest.raises(PrefetchStoppedError, match="not_started"):
            prefetch.begin_round(_schedule("a"))
        prefetch.start()
        with pytest.raises(RuntimeError, match="cannot start"):
            prefetch.start()
        prefetch.close()
        prefetch.close()  # idempotent
        with pytest.raises(PrefetchStoppedError, match="closed"):
            prefetch.begin_round(_schedule("a"))

    def test_a_dead_worker_wakes_the_consumer_with_its_cause(self) -> None:
        streams = _streams(_ScriptedStream("a", [_Boom()]))
        with PrefetchingBatchStreams(streams, NO_BATCH_WARMING, 1) as prefetch:
            prefetch.begin_round(_schedule("a"))
            with pytest.raises(PrefetchStoppedError, match="failed") as raised:
                prefetch.next_batch("a")
            assert isinstance(raised.value.__cause__, _Boom)
            prefetch.end_round()  # returns: nothing is in flight

    def test_a_dead_worker_refuses_the_next_round_with_its_cause(self) -> None:
        prefetch = PrefetchingBatchStreams(
            _streams(_ScriptedStream("a", [_Boom()])), NO_BATCH_WARMING, 1
        )
        prefetch.start()
        prefetch.begin_round(_schedule("a"))
        _wait_until(lambda: prefetch._state is _Lifecycle.FAILED)  # noqa: SLF001
        prefetch.end_round()
        with pytest.raises(PrefetchStoppedError, match="failed") as raised:
            prefetch.begin_round(_schedule("a"))
        assert isinstance(raised.value.__cause__, _Boom)
        prefetch.close()

    def test_close_wakes_a_blocked_consumer(self) -> None:
        stream = _ScriptedStream("a")
        stream.gate = threading.Event()
        prefetch = PrefetchingBatchStreams(_streams(stream), NO_BATCH_WARMING, 1)
        prefetch.start()
        prefetch.begin_round(_schedule("a"))
        consumer, errors = _blocked_consumer(prefetch, "a")
        assert stream.entered.wait(timeout=5)
        closer = threading.Thread(target=prefetch.close)
        closer.start()
        consumer.join(timeout=5)
        assert not consumer.is_alive()
        assert isinstance(errors[0], PrefetchStoppedError)
        assert "closed" in str(errors[0])
        # close waits for the build in flight, whose batch is then dropped
        assert closer.is_alive()
        stream.gate.set()
        closer.join(timeout=5)
        assert not closer.is_alive()

    def test_a_stopped_error_without_a_failure_hides_no_context(self) -> None:
        prefetch = PrefetchingBatchStreams(
            _streams(_ScriptedStream("a")), NO_BATCH_WARMING, 1
        )
        with pytest.raises(PrefetchStoppedError) as raised:
            prefetch.begin_round(_schedule("a"))
        assert raised.value.__cause__ is None
        assert not raised.value.__suppress_context__

    def test_bad_arguments_are_refused(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            PrefetchingBatchStreams(_streams(_ScriptedStream("a")), NO_BATCH_WARMING, 0)
        with pytest.raises(ValueError, match="at least one stream"):
            PrefetchingBatchStreams({}, NO_BATCH_WARMING, 1)


def _cached_pipeline(
    precision: Precision = "fp32", cache: PooledFeatureCache | None = None
) -> CardEmbeddingPipeline:
    """The fake card pipeline with a small in-memory feature cache."""
    pipeline = fake_card_pipeline(4)
    return CardEmbeddingPipeline(
        pipeline.text_encoder,
        None,
        pipeline.pooling,
        pipeline.embedding_head,
        cache or memory_cache(HIDDEN, precision=precision),
    )


def _one_row_lru_cache() -> PooledFeatureCache:
    slab = TensorFeatureSlab(
        "vram",
        1,
        FeatureStorageDtype.FP32,
        LeastRecentlyUsed(1),
        SlabPlacement.FOLLOW_FIRST_PUT,
    )
    return PooledFeatureCache(TieredFeatureStore([slab]), fingerprint(HIDDEN), False)


class TestModelWarmer:
    def test_it_refuses_a_model_without_a_cache_or_at_another_precision(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="feature cache"):
            build_model_warmer(SingleCardModel(fake_card_pipeline(4)), "fp32")
        cached = _cached_pipeline("fp16")
        with pytest.raises(ValueError, match="built for fp16, not fp32"):
            build_model_warmer(SingleCardModel(cached), "fp32")

    def test_it_pins_and_releases_on_the_cpu(self) -> None:
        model = SingleCardModel(_cached_pipeline())
        warmer = build_model_warmer(model, "fp32")
        reservation = warmer.reserve([_card("a"), _card("b")])
        stats = model.feature_cache_stats()
        assert stats is not None
        assert (stats.tiers[0].rows_used, stats.tiers[0].pinned_rows) == (2, 2)
        warmer.release(reservation)
        stats = model.feature_cache_stats()
        assert stats is not None
        assert stats.tiers[0].pinned_rows == 0
        assert warmer.reserve([]).keys == ()

    def test_it_warns_once_when_pinned_rows_fill_a_tier(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = SingleCardModel(_cached_pipeline(cache=_one_row_lru_cache()))
        warmer = build_model_warmer(model, "fp32")
        with caplog.at_level(logging.WARNING):
            warmer.reserve([_card("a")])
            warmer.reserve([_card("b")])  # the one row is pinned by a
            warmer.reserve([_card("c")])
        assert caplog.text.count("full of pinned rows") == 1

    def test_no_batch_warming_reserves_nothing(self) -> None:
        reservation = NO_BATCH_WARMING.reserve([_card("x")])
        assert reservation.keys == ()
        NO_BATCH_WARMING.release(reservation)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs a CUDA (or ROCm) device"
    )
    def test_it_fills_the_cache_on_cuda_from_a_worker_thread(self) -> None:
        model = SingleCardModel(_cached_pipeline()).to("cuda")
        warmer = build_model_warmer(model, "fp32")
        reservations: list[FeatureReservation] = []
        worker = threading.Thread(
            target=lambda: reservations.append(warmer.reserve([_card("a"), _card("b")]))
        )
        worker.start()
        worker.join(timeout=30)
        stats = model.feature_cache_stats()
        assert stats is not None
        assert stats.tiers[0].rows_used == 2
        # The training thread now reads those rows on the default stream
        assert model([_card("a")])[0].device.type == "cuda"
        warmer.release(reservations[0])
