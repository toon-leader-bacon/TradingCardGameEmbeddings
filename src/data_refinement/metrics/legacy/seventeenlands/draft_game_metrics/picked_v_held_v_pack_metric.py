from typing import ClassVar
from pathlib import Path
import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.metric import metric_output_path
from src.data_refinement.seventeenlands.name_lookup import find_uuid_by_name
from src.schema.game_id import GameId


class PickedVHeldVPackMetric:
    """Metric: The cards that are currently picked, vs the cards that are avaliable
    in the pack to pick from. The true label of the metric is the actual card
    that was picked from the pack.

    Streaming in spirit (each row of raw data resolves to one output row on
    its own), but buffers resolved rows in memory across accumulate() calls
    and writes them out as a single parquet file in finalize() — parquet has
    no cheap per-chunk append, unlike CSV.
    """

    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/metrics/17lands/draft")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = (
        "{expansion}.{format_code}.picked_v_held_v_pack.parquet"
    )

    def __init__(
        self,
        card_binder: CardBinder,
        expansion: str,
        format_code: str,
        output_dir: Path | None = None,
        output_name: str | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve pack/pool/pick card names against.
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
        self._output_path = self.output_path(
            expansion, format_code, output_dir, output_name
        )
        self._rows: list[dict] = []

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def output_path(
        expansion: str,
        format_code: str,
        output_dir: Path | None = None,
        output_name: str | None = None,
    ) -> Path:
        return metric_output_path(
            PickedVHeldVPackMetric.DEFAULT_OUTPUT_DIR,
            PickedVHeldVPackMetric.DEFAULT_OUTPUT_NAME,
            expansion,
            format_code,
            output_dir,
            output_name,
        )

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """
        Each row in the chunk is a single pick event in a draft.
        There are 2 types of columns we care about:
        - "pack_card_<name>": how many copies of <name> were available
          to pick from in the pack (0 or more).
        - "pool_<name>": how many copies of <name> are already in the
          player's pool (previously picked, 0 or more).
        - "pick": the name of the card that was actually picked from
          the pack.

        The card's name is encoded in the column header, not the cell
        value — the cell value is a per-row copy count. <name> may
        contain space characters or even the comma character.

        Resolves every card name to a nocab_uuid, repeats each one by
        its copy count, and buffers one row per pick event
        (picked_card_uuid, pack_cards_uuids, pool_cards_uuids) in
        memory, to be written out by finalize().

        Inputs:
            chunk: a chunk of raw 17lands draft-data rows, must carry a
                "pick" column plus "pack_card_*"/"pool_*" columns.
        Output: none.
        Side effects: appends to self._rows.
        Exceptions: none expected.
        """
        pack_card_columns = [
            col for col in chunk.columns if col.startswith("pack_card_")
        ]
        pool_card_columns = [col for col in chunk.columns if col.startswith("pool_")]

        pack_card_names = {
            col: col.removeprefix("pack_card_") for col in pack_card_columns
        }
        pool_card_names = {col: col.removeprefix("pool_") for col in pool_card_columns}

        for _, row in chunk.iterrows():
            picked_card_uuid = find_uuid_by_name(
                self._card_binder, GameId.MTG, row["pick"]
            )
            if picked_card_uuid is None:
                # This is the true label, can't be null so skip if it is
                continue

            self._rows.append(
                {
                    "picked_card_uuid": str(picked_card_uuid),
                    "pack_cards_uuids": self._card_uuids_by_count(row, pack_card_names),
                    "pool_cards_uuids": self._card_uuids_by_count(row, pool_card_names),
                }
            )

    def _card_uuids_by_count(
        self, row: pd.Series, column_to_name: dict[str, str]
    ) -> list[str]:
        """
        The cell data is a number, typically <4 of the number of copies of the
        card. The column header contains the name of the card. So this helper
        function will iterate over the columns, and for each column, it will
        get the name of the card, and the number of copies of that card. It will
        then convert the card name to a nocab_uuid, and add that uuid to the list
        of uuids for that row.

        One nocab_uuid string per copy, per column, for every
        column whose cell holds a positive copy count.
        """
        uuids = []
        for column, name in column_to_name.items():
            count = row[column]
            if pd.isna(count) or count <= 0:
                continue
            uuid = find_uuid_by_name(self._card_binder, GameId.MTG, name)
            if uuid is None:
                continue
            uuids.extend([str(uuid)] * int(count))
        return uuids

    def finalize(self) -> Path:
        """Write every buffered row out to a single parquet file.

        Inputs: none (uses buffered self._rows).
        Output: the path written to.
        Side effects: creates self._output_path's parent directories if
            missing, writes self._output_path.
        Exceptions: whatever pandas.DataFrame.to_parquet raises.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(self._output_path, index=False)
        return self._output_path
