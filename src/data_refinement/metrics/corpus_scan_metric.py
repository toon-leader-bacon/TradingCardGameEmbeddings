"""The shared, generic contract every masking metric in this project
satisfies structurally.

See plans/masking_metrics.md for the design rationale: unlike
metric.py's Metric[RawRowT] Protocol (driven per-row by an external
scanner over a large raw file it can't hold in memory all at once,
e.g. sts_gg/scanner.py's scan_runs_jsonl), a masking metric's source
is a CardBinder that's already fully loaded by the time it runs -
there's nothing to stream, so there's no accumulate()/finalize()
split. One scan() call walks the whole corpus and writes the result.
"""

from pathlib import Path
from typing import Protocol


class CorpusScanMetric(Protocol):
    """One metric, driven over its own already-held corpus in a single
    call.

    Structural, not a base class - any object with this shape
    satisfies it without inheriting from it (e.g. MaskedFieldMetric,
    masked_field_metric.py).
    """

    def scan(self) -> Path:
        """Scan this metric's full corpus once, write one output row
        per eligible unit found, and return the path written.

        Inputs: none - the corpus/target is already held by whatever
            concrete object satisfies this Protocol, supplied at its
            own construction time (e.g. a CardLookup for a
            card-corpus metric).
        Output: the path written to.
        Side effects: implementation-defined - expected to create the
            output path's parent directories and write exactly one
            file.
        Exceptions: implementation-defined.

        Example:
            >>> metric = FactionMaskMetric(card_lookup)
            >>> metric.scan()
            PosixPath('data/metrics/gwent_one/faction_mask.parquet')
        """
        ...
