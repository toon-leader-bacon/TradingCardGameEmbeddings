"""RowwiseMetric - an Adapter (PATTERNS.md) that lets a not-yet-ported
row metric (Metric[dict]) run inside a family that scans chunks
(see game_data/README.md).

Transitional: a family wraps its row metrics in this only while it has
any, and each family's wrapping is deleted once its last row metric is
ported. game_data, the first family ported, no longer uses it;
replay_data is planned to wrap its row metrics here first (plan slice
4). Each family supplies its own frame_of to get a pandas DataFrame out
of its chunk type.

Per-row failure isolation lives here, as under the old row scanner: a
row that raises is skipped for that metric only, and the rest of the
chunk still reaches it. Rather than one traceback per bad row (a single
wiring bug would log thousands), accumulate() finishes the chunk and
then raises one RowFailures naming how many rows failed, chained to the
first row's exception. The scanner logs that once per chunk and counts
it in its end-of-CSV summary.
"""

from pathlib import Path
from typing import Callable, Generic, TypeVar

import pandas as pd

from src.data_refinement.metrics.metric import Metric

ChunkT = TypeVar("ChunkT")


class RowFailures(Exception):
    """Some rows of one chunk raised inside a wrapped row metric; the
    other rows were still accumulated. Chained (__cause__) to the first
    row's exception."""


class RowwiseMetric(Generic[ChunkT]):
    """A Metric[ChunkT] that feeds each chunk's rows, one dict at a time,
    to an inner Metric[dict].
    """

    def __init__(
        self, inner: Metric[dict], frame_of: Callable[[ChunkT], pd.DataFrame]
    ) -> None:
        """Wrap inner so it can be driven chunk by chunk.

        Inputs:
            inner: the row metric to drive.
            frame_of: gets a chunk's rows as a DataFrame; the family
                that wraps row metrics supplies it for its chunk type.
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        self._inner = inner
        self._frame_of = frame_of

    def accumulate(self, chunk: ChunkT) -> None:
        """Feed every row of chunk to the inner metric, in order.

        Inputs: chunk. Output: none.
        Side effects: inner.accumulate(row) once per row; a row that
            raises is skipped (see module docstring).
        Exceptions: RowFailures, after the whole chunk, if any row
            raised; whatever frame_of raises.

        Example:
            >>> RowwiseMetric(WinMetric(...), frame_of).accumulate(chunk)
        """
        # Same dicts the old pandas row scanner produced
        rows = self._frame_of(chunk).to_dict(orient="records")

        failed_rows = 0
        first_failure: Exception | None = None

        # Per-row isolation: a bad row is skipped, the rest still count
        for row in rows:
            try:
                self._inner.accumulate(row)
            except Exception as failure:
                failed_rows += 1
                first_failure = first_failure or failure

        # One summarizing error per chunk, not one traceback per row
        if first_failure is not None:
            raise RowFailures(
                f"{failed_rows} of {len(rows)} rows raised; the first is chained"
            ) from first_failure

    def finalize(self) -> Path:
        """Finalize the inner metric.

        Inputs: none. Output: the inner metric's output path.
        Side effects: inner.finalize()'s. Exceptions: inner.finalize()'s.

        Example:
            >>> RowwiseMetric(inner, frame_of).finalize()
        """
        return self._inner.finalize()

    def __repr__(self) -> str:
        """Names the wrapped metric, for scanner log lines.

        Inputs: none. Output: str. Side effects: none. Exceptions: none.
        """
        return f"RowwiseMetric({self._inner!r})"
