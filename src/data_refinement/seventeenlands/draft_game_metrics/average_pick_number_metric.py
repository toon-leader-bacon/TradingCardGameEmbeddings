"""
Each row in the csv is a pick in a MTG draft.
For this metric, we care about teh pick_number column (int), and the pick column
(str name of card picked).
"""

from typing import ClassVar
import csv
from pathlib import Path
from typing import Dict, Set
import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.name_lookup import find_uuid_by_name
from src.schema.game_id import GameId


class DictWithDefault(dict):
    def __init__(self, default_value=None) -> None:
        super().__init__()
        self._default_value = default_value

    def __getitem__(self, key: str):
        if key not in self:
            return self._default_value
        return super().__getitem__(key)


class AveragePickNumberMetric:
    """
    Accumulator Metric: running per-card pick_number sum/count,
    written out once, in finalize().

    Satisfies v2's Metric Protocol (src/data_refinement/seventeenlands/
    v2/metric.py) structurally.
    """

    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/metrics/17lands/v2/draft")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{expansion}.{format_code}.average_pick_number.jsonl"

    def __init__(self, card_binder: CardBinder,
                 expansion: str,
                 format_code: str,
                 output_path: Path | None = None,
                 output_name: str | None = None) -> None:
        self._card_binder = card_binder
        self._expansion = expansion
        self._format_code = format_code
        self._output_path = self.output_path(expansion, format_code, output_path, output_name)

        # Accumulator state
        self._name_to_pick_sum: DictWithDefault = DictWithDefault(0)
        self._name_to_pick_count: DictWithDefault = DictWithDefault(0)

    @staticmethod
    def output_path(expansion: str, format_code: str,
                    dir: Path | None = None,
                    name: str | None = None) -> Path:
        if dir is None:
            dir = AveragePickNumberMetric.DEFAULT_OUTPUT_DIR
        if name is None:
            name = AveragePickNumberMetric.DEFAULT_OUTPUT_NAME
        return dir / name.format(expansion=expansion, format_code=format_code)

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """Given a chunk of draft data from 17lands, extract the pick_number and
        the associated pick name. Accumulate the sum of the pick_number for
        each pick name.
        """
        for _, row in chunk.iterrows():
            pick_number = row["pick_number"]
            pick_name = row["pick"]
            # Validate data
            if pick_number is None or pick_name is None:
                continue

            # Accumulate data
            self._name_to_pick_sum[pick_name] += pick_number
            self._name_to_pick_count[pick_name] += 1

    def finalize(self) -> Path:
        """Output a CSV with the following columns:
        - name: str
        - nocab_uuid: str
        - average_pick_number: float
        """
        invalid_cards: set[str] = set()

        with open(self._output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "nocab_uuid", "average_pick_number"])
            for name, pick_sum in self._name_to_pick_sum.items():
                nocab_uuid = find_uuid_by_name(
                    self._card_binder, GameId.MTG, name
                )
                if nocab_uuid is None:
                    invalid_cards.add(name)
                    continue
                if self._name_to_pick_count[name] == 0:
                    invalid_cards.add(name)
                    continue
                average_pick_number = pick_sum / self._name_to_pick_count[name]
                writer.writerow([name, nocab_uuid, average_pick_number])
        return self._output_path
