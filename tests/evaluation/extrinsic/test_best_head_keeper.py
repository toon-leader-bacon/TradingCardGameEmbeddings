import pytest
import torch

from src.evaluation.extrinsic.best_head_keeper import BestHeadKeeper
from src.training.recording.reports import RoundReport
from tests.training.fakes import FakeDojo


def _report(round_index: int, losses: dict[str, float]) -> RoundReport:
    return RoundReport(
        "extrinsic", round_index, round_index, 0.0, losses, {}, frozenset()
    )


def _set_head(dojo: FakeDojo, value: float) -> None:
    with torch.no_grad():
        for parameter in dojo.trainable_parameters():
            parameter.fill_(value)


def _head_values(dojo: FakeDojo) -> set[float]:
    return {float(v) for p in dojo.trainable_parameters() for v in p.detach().flatten()}


def test_restores_each_head_to_its_own_best_round() -> None:
    a, b = FakeDojo("a"), FakeDojo("b")
    keeper = BestHeadKeeper([a, b])
    # Round 0: both at their best; round 1: a improves, b regresses
    _set_head(a, 1.0)
    _set_head(b, 1.0)
    keeper.on_round_end(_report(0, {"a": 0.9, "b": 0.5}))
    _set_head(a, 2.0)
    _set_head(b, 2.0)
    keeper.on_round_end(_report(1, {"a": 0.7, "b": 0.8}))
    _set_head(a, 3.0)
    _set_head(b, 3.0)

    keeper.restore_best_heads()

    assert _head_values(a) == {2.0}
    assert _head_values(b) == {1.0}
    assert keeper.best_rounds == {"a": 1, "b": 0}


def test_a_tie_keeps_the_earlier_snapshot() -> None:
    a = FakeDojo("a")
    keeper = BestHeadKeeper([a])
    _set_head(a, 1.0)
    keeper.on_round_end(_report(0, {"a": 0.5}))
    _set_head(a, 2.0)
    keeper.on_round_end(_report(1, {"a": 0.5}))
    keeper.restore_best_heads()
    assert _head_values(a) == {1.0}


def test_a_non_finite_loss_never_becomes_the_best() -> None:
    a = FakeDojo("a")
    keeper = BestHeadKeeper([a])
    _set_head(a, 1.0)
    keeper.on_round_end(_report(0, {"a": 0.5}))
    _set_head(a, 2.0)
    keeper.on_round_end(_report(1, {"a": float("nan")}))
    keeper.restore_best_heads()
    assert _head_values(a) == {1.0}
    assert keeper.best_rounds == {"a": 0}


def test_unscored_and_unknown_dojos_are_left_alone() -> None:
    a = FakeDojo("a")
    keeper = BestHeadKeeper([a])
    keeper.on_round_end(_report(0, {"held_out": 0.1}))
    _set_head(a, 4.0)
    keeper.restore_best_heads()
    assert _head_values(a) == {4.0}
    assert keeper.best_rounds == {}


def test_a_snapshot_is_a_copy_not_a_view() -> None:
    a = FakeDojo("a")
    keeper = BestHeadKeeper([a])
    _set_head(a, 1.0)
    keeper.on_round_end(_report(0, {"a": 0.5}))
    _set_head(a, 9.0)  # in-place training updates must not reach the snapshot
    keeper.restore_best_heads()
    assert _head_values(a) == {1.0}


def test_a_head_that_changed_shape_is_refused() -> None:
    a = FakeDojo("a")
    keeper = BestHeadKeeper([a])
    keeper.on_round_end(_report(0, {"a": 0.5}))
    a._head = torch.nn.Linear(3, 1)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="changed shape"):
        keeper.restore_best_heads()
