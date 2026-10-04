"""Shared fixtures for the replay_data chunk tests: a tiny MTG binder
with Arena aliases, a CSV written from row dicts, and helpers that parse
or scan it the way the driver does."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayDataChunk,
    ReplayField,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk_parser import (
    ReplayDataChunkParser,
)
from src.data_refinement.metrics.seventeenlands.replay_data.scanner import (
    scan_replay_csv,
)
from src.data_refinement.metrics.seventeenlands.slice_file import finished_count_table
from src.data_refinement.metrics.seventeenlands.sliced_metric import is_count_table
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

OWLBEAR = "Owlbear"
MORNINGSTAR = "Goblin Morningstar"
# Arena ids: OWLBEAR_ID and MORNINGSTAR_ID are in the binder, UNKNOWN_ID is not
OWLBEAR_ID = "1"
MORNINGSTAR_ID = "2"
UNKNOWN_ID = "999"

VERSION = MetricVersionMetadata(game=GameId.MTG, card_binder_version="test-version")

TURNS = (1, 2)


def field_column(actor: Actor, turn: int, field: ReplayField) -> str:
    """The header column of one actor's turn's field."""
    return f"{actor.label}_turn_{turn}_{field.value}"


def _field_header() -> list[str]:
    """Every field column for TURNS, as the real CSVs lay them out (no
    oppo cards_tutored column), plus one per-turn column no metric
    reads."""
    result = []
    for actor in Actor:
        for turn in TURNS:
            result.append(f"{actor.label}_turn_{turn}_lands_played")
            for field in ReplayField:
                if actor is Actor.OPPO and field is ReplayField.CARDS_TUTORED:
                    continue
                result.append(field_column(actor, turn, field))
    return result


HEADER = [
    "draft_id",
    "match_number",
    "game_number",
    "num_turns",
    f"deck_{OWLBEAR}",
    f"deck_{MORNINGSTAR}",
    "deck_Unmatched Card",
    f"sideboard_{OWLBEAR}",
    *_field_header(),
]


def binder_with_cards() -> CardBinder:
    """A CardBinder holding OWLBEAR and MORNINGSTAR, each with its Arena
    alias."""
    binder = CardBinder()
    for name, arena_id in ((OWLBEAR, OWLBEAR_ID), (MORNINGSTAR, MORNINGSTAR_ID)):
        card = GenericCard(
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
        binder.create(card)
        binder.register_alias(GameId.MTG, DataSource.ARENA, arena_id, card.nocab_uuid)
    return binder


def uuid_for(binder: CardBinder, name: str) -> UUID:
    """The one card named name in binder."""
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return cards[0].nocab_uuid


def row(
    cells: dict[str, str] | None = None,
    num_turns: int = 8,
    owlbear_deck: int = 0,
    morningstar_deck: int = 0,
    draft_id: str = "draft1",
    match_number: int = 1,
    game_number: int = 1,
) -> dict:
    """One replay_data row over HEADER: cells maps field columns to
    their "|"-joined Arena ids; every other field cell is empty."""
    result: dict = {column: "" for column in HEADER}
    result.update(
        {
            "draft_id": draft_id,
            "match_number": match_number,
            "game_number": game_number,
            "num_turns": num_turns,
            f"deck_{OWLBEAR}": owlbear_deck,
            f"deck_{MORNINGSTAR}": morningstar_deck,
            "deck_Unmatched Card": 0,
            f"sideboard_{OWLBEAR}": 0,
        }
    )
    result.update(cells or {})
    return result


def write_csv(path: Path, rows: list[dict], header: list[str] = HEADER) -> Path:
    """rows written as a CSV with header's columns."""
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    return path


def parser_for(binder: CardBinder, header: list[str] = HEADER) -> ReplayDataChunkParser:
    """A parser built the way the driver builds one."""
    return ReplayDataChunkParser.from_header(header, binder, GameId.MTG)


def read_batches(
    csv_path: Path, parser: ReplayDataChunkParser, block_size: int = 1 << 20
) -> list[pa.RecordBatch]:
    """csv_path's record batches, read with parser's options."""
    reader = pa_csv.open_csv(
        csv_path,
        read_options=pa_csv.ReadOptions(block_size=block_size),
        convert_options=pa_csv.ConvertOptions(
            include_columns=parser.needed_columns(),
            column_types=parser.column_types(),
        ),
    )
    return list(reader)


def parse_rows(tmp_path: Path, binder: CardBinder, rows: list[dict]) -> ReplayDataChunk:
    """rows written, read and parsed as one chunk."""
    csv_path = write_csv(tmp_path / "replays.csv", rows)
    parser = parser_for(binder)
    batches = read_batches(csv_path, parser)
    assert len(batches) == 1
    return parser.parse(batches[0])


def scan_into_frame(
    tmp_path: Path,
    binder: CardBinder,
    rows: list[dict],
    metric: Metric[ReplayDataChunk],
    index: str | None = "nocab_uuid",
) -> pd.DataFrame:
    """Write rows, scan them through metric, and return its output,
    indexed by index (None: unindexed). A count-table metric's output is
    its finished (labelled) table, as a one-partition slice would give."""
    csv_path = write_csv(tmp_path / "replays.csv", rows)
    scan_replay_csv(csv_path, [metric], parser_for(binder))
    output_path = metric.finalize()
    metric_class = type(metric)
    if is_count_table(metric_class):
        frame = finished_count_table(
            metric_class, [pq.read_table(output_path)]
        ).to_pandas()
    else:
        frame = pd.read_parquet(output_path)
    return frame if index is None else frame.set_index(index)
