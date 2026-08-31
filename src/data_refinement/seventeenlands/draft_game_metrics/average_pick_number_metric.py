"""
Each row in the csv is a pick in a MTG draft.
For this metric, we care about teh pick_number column (int), and the pick column
(str name of card picked).
"""

from collections import defaultdict
from typing import ClassVar
from pathlib import Path
import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.metric import metric_output_path
from src.data_refinement.seventeenlands.name_lookup import find_uuid_by_name
from src.schema.game_id import GameId


class AveragePickNumberMetric:
    """
    Accumulator Metric: running per-card pick_number sum/count,
    written out once, in finalize().

    Satisfies the Metric Protocol (src/data_refinement/seventeenlands/
    metric.py) structurally.
    """

    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/metrics/17lands/draft")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{expansion}.{format_code}.average_pick_number.parquet"

    def __init__(self, card_binder: CardBinder,
                 expansion: str,
                 format_code: str,
                 output_dir: Path | None = None,
                 output_name: str | None = None) -> None:
        """
        Inputs:
            card_binder: registry to resolve pick names against.
            expansion: 17lands expansion code (e.g. "MSH"), used only
                to fill in the default output filename.
            format_code: 17lands format code (e.g. "PremierDraft"),
                used only to fill in the default output filename.
            output_dir: overrides DEFAULT_OUTPUT_DIR when given.
            output_name: overrides DEFAULT_OUTPUT_NAME when given.
        Output: none (constructor).
        Side effects: none — no I/O happens until finalize() is called.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._expansion = expansion
        self._format_code = format_code
        self._output_path = self.output_path(expansion, format_code, output_dir, output_name)

        # Accumulator state
        self._name_to_pick_sum: defaultdict[str, int] = defaultdict(int)
        self._name_to_pick_count: defaultdict[str, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def output_path(expansion: str, format_code: str,
                     output_dir: Path | None = None,
                     output_name: str | None = None) -> Path:
        return metric_output_path(
            AveragePickNumberMetric.DEFAULT_OUTPUT_DIR,
            AveragePickNumberMetric.DEFAULT_OUTPUT_NAME,
            expansion, format_code, output_dir, output_name,
        )

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """Fold one chunk of draft data into this metric's running
        per-card pick_number sum/count.

        Inputs:
            chunk: a chunk of raw 17lands draft-data rows, must carry
                "pick_number" and "pick" columns.
        Output: none.
        Side effects: updates self._name_to_pick_sum/_name_to_pick_count
            in place.
        Exceptions: none expected.
        """
        for _, row in chunk.iterrows():
            pick_number = row["pick_number"]
            pick_name = row["pick"]
            if pd.isna(pick_number) or pd.isna(pick_name):
                continue

            self._name_to_pick_sum[pick_name] += pick_number
            self._name_to_pick_count[pick_name] += 1

    def finalize(self) -> Path:
        """Resolve every accumulated pick name to a nocab_uuid and write
        a parquet file with columns:
        - name: str
        - nocab_uuid: str
        - average_pick_number: float

        Inputs: none (uses accumulated state).
        Output: the path written to.
        Side effects: creates self._output_path's parent directories if
            missing, writes self._output_path.
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric = AveragePickNumberMetric(card_binder, "MSH", "PremierDraft")
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
            PosixPath('data/final/metrics/17lands/draft/MSH.PremierDraft.average_pick_number.parquet')
        """
        rows = []
        for name, pick_sum in self._name_to_pick_sum.items():
            nocab_uuid = find_uuid_by_name(self._card_binder, GameId.MTG, name)
            if nocab_uuid is None or self._name_to_pick_count[name] == 0:
                continue
            rows.append({
                "name": name,
                "nocab_uuid": str(nocab_uuid),
                "average_pick_number": pick_sum / self._name_to_pick_count[name],
            })

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(self._output_path, index=False)
        return self._output_path
