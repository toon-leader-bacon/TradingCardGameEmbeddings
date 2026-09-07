"""The shared, generic contract every metric class in this container
satisfies structurally.

See plans/sts_gg_metrics.md for the design rationale: each raw source
feeds a different row shape into accumulate() (sts_gg's JSONL rows are
dict, 17lands' CSV chunks are pd.DataFrame - see
legacy/seventeenlands/metric.py's own, separate Metric Protocol, not
migrated onto this one), so this Protocol is generic over that row
type rather than fixing one concrete shape. Streaming metrics do their
real work inside accumulate() and treat finalize() as a near-no-op
(closing a writer); accumulation metrics buffer state in accumulate()
and do the real computation inside finalize().
"""

from pathlib import Path
from typing import Protocol, TypeVar

RawRowT = TypeVar("RawRowT", contravariant=True)


class Metric(Protocol[RawRowT]):
    """One metric, driven one raw row (or chunk) at a time.

    Structural, not a base class - any object with this shape
    satisfies it without inheriting from it (e.g. CardUpgradeRateMetric,
    CardWinRateAtAct2Metric in metrics/sts_gg/).
    """

    def accumulate(self, row: RawRowT) -> None:
        """Fold one raw row into this metric's own state.

        Inputs:
            row: one unit of raw data, in whatever shape this metric's
                raw source produces (e.g. one parsed JSON dict for a
                JSONL source).
        Output: none.
        Side effects: implementation-defined - a streaming metric
            writes output immediately; an accumulation metric only
            updates in-memory state.
        Exceptions: implementation-defined.
        """
        ...

    def finalize(self) -> Path:
        """Write this metric's output to its own file and return that
        path.

        Inputs: none.
        Output: the path written to.
        Side effects: implementation-defined - a streaming metric only
            closes an already-open writer; an accumulation metric
            performs its real computation and writes here.
        Exceptions: implementation-defined.
        """
        ...
