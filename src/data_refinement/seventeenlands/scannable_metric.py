"""The subset of Metric/DraftMetric/ReplayMetric that the shared checkpoint
helper (metric_checkpoint.py) and results-writer (metric_writer.py) need —
everything except accumulate(), whose signature genuinely differs per
pipeline (see each pipeline's own Metric/DraftMetric/ReplayMetric
protocol: each takes a differently-shaped `resolved` second argument)
and which neither shared piece ever calls.

Every concrete Metric/DraftMetric/ReplayMetric implementation already
satisfies this Protocol structurally (typing.Protocol, matching this
codebase's existing convention) — no inheritance, registration, or change
to those three Protocols is needed for this to work.
"""

from typing import Protocol
from uuid import UUID

from src.data_refinement.seventeenlands.metric_result import MetricResult


class ScannableMetric(Protocol):
    """Strategy subset: whatever a *MetricScanner needs to checkpoint and
    finalize a metric, independent of how that metric accumulates data.
    """

    name: str

    def finalize(self) -> dict[UUID, MetricResult]: ...
    def save_state(self) -> dict: ...  # Memento: serializable snapshot
    def load_state(self, state: dict) -> None: ...  # Memento: restore from snapshot
