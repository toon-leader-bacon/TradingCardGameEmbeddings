import pytest

from src.training.diet.dojo_batch_stream import DojoBatchStream
from tests.training.fakes import BUDGET, FakeDojo


def test_restarts_the_pass_when_exhausted() -> None:
    stream = DojoBatchStream(FakeDojo("a", n_batches=2), BUDGET)  # type: ignore[arg-type]
    assert len([stream.next_batch() for _ in range(5)]) == 5


def test_raises_if_the_dojo_yields_nothing() -> None:
    stream = DojoBatchStream(FakeDojo("a", n_batches=0), BUDGET)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        stream.next_batch()


def test_a_mid_pass_error_propagates_and_the_next_call_starts_a_new_pass() -> None:
    dojo = FakeDojo("a", n_batches=3, fail_at_batch=1)
    stream = DojoBatchStream(dojo, BUDGET)  # type: ignore[arg-type]
    stream.next_batch()
    with pytest.raises(RuntimeError, match="bad row"):
        stream.next_batch()
    stream.next_batch()  # fresh pass, batch 0 again
