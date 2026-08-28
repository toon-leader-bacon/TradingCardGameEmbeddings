"""The shared output row shape every 17lands metric pipeline produces.

Promoted out of game_data_metrics/card_column_set.py once draft_data_metrics
became a second real consumer — MetricResult has no game_data-specific
assumptions baked into it (it's just "one card, one metric, one value,
one sample size, tagged with which expansion/format it came from"), so
keeping it private to game_data_metrics/ would have forced
draft_data_metrics to either duplicate it or reach into a cousin
directory's private module (PRINCIPLES.md section 3's "heavy cross-
cousin-directory imports" smell). Lives at this shared seventeenlands/
level, not the more general src/schema/ — expansion/format are
17lands-specific framing, not universal across every future source.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class MetricResult:
    """One (card, metric) row of a finished metric run's output.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: UUID
    metric_name: str
    value: float
    sample_size: int
    expansion: str
    format: str
