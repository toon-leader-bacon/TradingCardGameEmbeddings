"""Shared fixtures for the draft_data chunk tests: a small binder, a
draft_data CSV header and rows, and a scan that returns a metric's
finished output."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    DraftDataChunk,
)
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk_parser import (
    DraftDataChunkParser,
)
from src.data_refinement.metrics.seventeenlands.draft_data.scanner import (
    scan_draft_csv,
)
from src.data_refinement.metrics.seventeenlands.slice_file import finished_count_table
from src.data_refinement.metrics.seventeenlands.sliced_metric import is_count_table
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

OWLBEAR = "Owlbear"
MORNINGSTAR = "Goblin Morningstar"
BOLT = "Lightning Bolt"
VERSION = MetricVersionMetadata(game=GameId.MTG, card_binder_version="test-version")

HEADER = [
    "draft_id",
    "pack_number",
    "pick_number",
    "pick",
    "rank",
    f"pack_card_{OWLBEAR}",
    f"pack_card_{MORNINGSTAR}",
    f"pool_{OWLBEAR}",
    f"pool_{BOLT}",
]
PICK_TWO_HEADER = [*HEADER[:4], "pick_2", *HEADER[4:]]


def binder_with_cards(names: list[str]) -> CardBinder:
    """A CardBinder holding one MTG card per name."""
    binder = CardBinder()
    for name in names:
        binder.create(
            GenericCard(
                nocab_uuid=uuid4(),
                source_game=GameId.MTG,
                name=name,
                raw_content={},
                provenance=Provenance(
                    data_source=DataSource.SCRYFALL,
                    source_id=name,
                    fetched_at=datetime.now(timezone.utc),
                ),
            )
        )
    return binder


def uuid_for(binder: CardBinder, name: str) -> UUID:
    """The one card named name's uuid."""
    (card,) = binder.get_by_name(GameId.MTG, name)
    return card.nocab_uuid


def row(
    pick: str,
    owlbear: int = 0,
    morningstar: int = 0,
    pack_number: int = 0,
    pick_number: int = 0,
    rank: str = "gold",
    owlbear_pool: int = 0,
    bolt_pool: int = 0,
    pick_2: str = "",
    draft_id: str = "draft1",
) -> dict:
    """One draft_data row over PICK_TWO_HEADER's columns (HEADER drops
    pick_2)."""
    return {
        "draft_id": draft_id,
        "pack_number": pack_number,
        "pick_number": pick_number,
        "pick": pick,
        "pick_2": pick_2,
        "rank": rank,
        f"pack_card_{OWLBEAR}": owlbear,
        f"pack_card_{MORNINGSTAR}": morningstar,
        f"pool_{OWLBEAR}": owlbear_pool,
        f"pool_{BOLT}": bolt_pool,
    }


def write_csv(path: Path, rows: list[dict], header: list[str] = HEADER) -> Path:
    """rows as a CSV with header's columns (empty rows: header only)."""
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    return path


def parser_for(binder: CardBinder, header: list[str] = HEADER) -> DraftDataChunkParser:
    """A parser built the way the driver builds one."""
    return DraftDataChunkParser.from_header(header, binder, GameId.MTG)


def scan_into_frame(
    tmp_path: Path,
    binder: CardBinder,
    rows: list[dict],
    metric: Metric[DraftDataChunk],
    header: list[str] = HEADER,
    block_size: int = 1 << 20,
) -> pd.DataFrame:
    """Write rows, scan them through metric, and return its output: a
    count table's finished (labelled) table, else the file as written."""
    csv_path = write_csv(tmp_path / "picks.csv", rows, header)
    scan_draft_csv(csv_path, [metric], parser_for(binder, header), block_size)
    output_path = metric.finalize()
    metric_class = type(metric)
    if is_count_table(metric_class):
        return finished_count_table(
            metric_class, [pq.read_table(output_path)]
        ).to_pandas()
    return pd.read_parquet(output_path)
