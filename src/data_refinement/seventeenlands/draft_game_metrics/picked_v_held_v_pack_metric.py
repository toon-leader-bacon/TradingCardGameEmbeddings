import csv
from pathlib import Path
from typing import ClassVar
import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.name_lookup import find_uuid_by_name


class PickedVHeldVPackMetric:
    """Metric: The cards that are currently picked, vs the cards that are avaliable
    in the pack to pick from. The true label of the metric is the actual card
    that was picked from the pack.
    """

    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/metrics/17lands/v3/draft")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{expansion}.{format_code}.picked_v_held_v_pack.csv"

    def __init__(self, card_binder: CardBinder,
                 expansion: str,
                 format_code: str,
                 output_dir: Path | None,
                 output_name: str | None,) -> None:
        self._card_binder = card_binder
        self._expansion = expansion
        self._format_code = format_code
        self._output_path = self.output_path(expansion, format_code, output_dir, output_name)

    @staticmethod
    def output_path(expansion: str, format_code: str, dir: Path | None = None, name: str | None = None) -> Path:
        if dir is None:
            dir = PickedVHeldVPackMetric.DEFAULT_OUTPUT_DIR
        if name is None:
            name = PickedVHeldVPackMetric.DEFAULT_OUTPUT_NAME
        return dir / name.format(expansion=expansion, format_code=format_code)

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """
        Each row in the chunk is a single pick event in a draft. 
        There are 2 types of columns we care about:
        - "pack_card_*": The cards in the pack that were avaliable to pick from.
        - "pool_*": Cards that have already been picked are are in the player's pool.
        - "pick": The name of the card that was actually picked from the pack.

        Each * is the name of a card (and may contain space characters or even 
        the comma character). 

        This metric will extract all the cards from both the pack and pool and
        write those nocab_uuids to a CSV, along with the nocab_uuid of the true
        picked card.
        """
        self._set_up_output_file()
        with open(self._output_path, "w") as f:
            writer = csv.writer(f)

            for _, row in chunk.iterrows():
                pack_card_columns = [col for col in chunk.columns if col.startswith("pack_card_")]
                pool_card_columns = [col for col in chunk.columns if col.startswith("pool_")]

                pack_cards_names = [row[col] for col in pack_card_columns]
                pool_cards_names = [row[col] for col in pool_card_columns]
                picked_card_name = row["pick"]

                pack_cards_uuids = [find_uuid_by_name(name) for name in pack_cards_names]
                pool_cards_uuids = [find_uuid_by_name(name) for name in pool_cards_names]
                picked_card_uuid = find_uuid_by_name(picked_card_name)

                # Validate for non null values
                # Filter out any null values
                pack_cards_uuids = [uuid for uuid in pack_cards_uuids if uuid is not None]
                pool_cards_uuids = [uuid for uuid in pool_cards_uuids if uuid is not None]

                if picked_card_uuid is None:
                    # This is the true label, can't be null so skip if it is
                    continue

                # Write the data out to the CSV
                writer.writerow([picked_card_uuid, pack_cards_uuids, pool_cards_uuids])

    def finalize(self) -> None:
        """
        For this type of metric, the accumulate methods writes the results to 
        the output file directly. No need to finalize, but we will return the 
        path to the output file.
        """
        return self._output_path

    def _set_up_output_file(self) -> None:
        """ IF the file doesn't exist, create it and write the header.

        This is useful because the accumlate method is called for each chunk, 
        and we want to make sure the file is set up correctly the first time
        we process data (not every chunk)
        """
        if not self._output_path.exists():
            with open(self._output_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["picked_card_uuid", "pack_cards_uuids", "pool_cards_uuids"])
        # Else if the file is empty, write the header
        else:
            with open(self._output_path, "r") as f:
                reader = csv.reader(f)
                if len(reader) == 0:
                    writer = csv.writer(f)
                    writer.writerow(["picked_card_uuid", "pack_cards_uuids", "pool_cards_uuids"])
        return self._output_path
