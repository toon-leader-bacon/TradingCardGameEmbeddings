"""Shared fixtures for the game_data chunk tests: a tiny MTG binder, a
CSV written from row dicts, and helpers that parse or scan it the way
the driver does."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.seventeenlands.game_data.chunk_decks import (
    build_chunk_decks,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameKeys,
    GameZone,
    ZoneCounts,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import scan_game_csv
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

OWLBEAR = "Owlbear"
MORNINGSTAR = "Goblin Morningstar"

HEADER = [
    "draft_id",
    "match_number",
    "game_number",
    "won",
    "on_play",
    "num_turns",
    "rank",
    f"opening_hand_{OWLBEAR}",
    f"drawn_{OWLBEAR}",
    f"tutored_{OWLBEAR}",
    f"deck_{OWLBEAR}",
    f"sideboard_{OWLBEAR}",
    f"opening_hand_{MORNINGSTAR}",
    f"drawn_{MORNINGSTAR}",
    f"deck_{MORNINGSTAR}",
    "deck_Unmatched Card",
]

VERSION = MetricVersionMetadata(game=GameId.MTG, card_binder_version="test-version")


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
    """The one card named name in binder."""
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return cards[0].nocab_uuid


def row(
    won: bool,
    num_turns: int = 8,
    on_play: bool = True,
    owlbear_deck: int = 0,
    owlbear_opening_hand: int = 0,
    owlbear_drawn: int = 0,
    owlbear_tutored: int = 0,
    owlbear_sideboard: int = 0,
    morningstar_deck: int = 0,
    rank: str = "gold",
    draft_id: str = "draft1",
    match_number: int = 1,
    game_number: int = 1,
) -> dict:
    """One game_data row over HEADER; unlisted counts are 0."""
    result: dict = {column: 0 for column in HEADER}
    result.update(
        {
            "draft_id": draft_id,
            "match_number": match_number,
            "game_number": game_number,
            "won": won,
            "on_play": on_play,
            "num_turns": num_turns,
            "rank": rank,
            f"opening_hand_{OWLBEAR}": owlbear_opening_hand,
            f"drawn_{OWLBEAR}": owlbear_drawn,
            f"tutored_{OWLBEAR}": owlbear_tutored,
            f"deck_{OWLBEAR}": owlbear_deck,
            f"sideboard_{OWLBEAR}": owlbear_sideboard,
            f"deck_{MORNINGSTAR}": morningstar_deck,
        }
    )
    return result


def write_csv(path: Path, rows: list[dict], header: list[str] = HEADER) -> Path:
    """rows written as a CSV with header's columns (empty rows: header
    only)."""
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    return path


def parser_for(binder: CardBinder, header: list[str] = HEADER) -> GameDataChunkParser:
    """A parser built the way the driver builds one."""
    return GameDataChunkParser.from_header(header, binder, GameId.MTG)


def read_batches(csv_path: Path, parser: GameDataChunkParser) -> list[pa.RecordBatch]:
    """csv_path's record batches, read with parser's options."""
    reader = pa_csv.open_csv(
        csv_path,
        convert_options=pa_csv.ConvertOptions(
            include_columns=parser.needed_columns(),
            column_types=parser.column_types(),
        ),
    )
    return list(reader)


def parse_one(csv_path: Path, parser: GameDataChunkParser) -> GameDataChunk:
    """The single chunk of a small CSV."""
    batches = read_batches(csv_path, parser)
    assert len(batches) == 1
    return parser.parse(batches[0])


def chunk_with_zones(
    zones: Mapping[GameZone, ZoneCounts],
    won: list[bool],
    on_play: list[bool] | None = None,
) -> GameDataChunk:
    """A chunk built directly from zones (zones not given are empty),
    with default scalars and its decks identified as the parser would."""
    rows = len(won)
    all_zones = {
        zone: zones.get(zone, ZoneCounts((), np.zeros((rows, 0), np.int16)))
        for zone in GameZone
    }
    keys = GameKeys(
        draft_id=np.array([f"draft{i}" for i in range(rows)], object),
        match_number=np.ones(rows, np.int64),
        game_number=np.ones(rows, np.int64),
    )
    return GameDataChunk(
        zones=all_zones,
        won=np.array(won, np.bool_),
        on_play=np.array(on_play if on_play is not None else [True] * rows, np.bool_),
        num_turns=np.full(rows, 8, np.int32),
        keys=keys,
        rank=np.full(rows, "", object),
        decks=build_chunk_decks(all_zones[GameZone.DECK], keys, GameId.MTG),
    )


def scan_into_frame(
    tmp_path: Path,
    binder: CardBinder,
    rows: list[dict],
    metric: Metric[GameDataChunk],
    block_size: int = 1 << 20,
    index: str | None = "nocab_uuid",
) -> pd.DataFrame:
    """Write rows, scan them through metric, and return its output,
    indexed by index (None: unindexed)."""
    csv_path = write_csv(tmp_path / "games.csv", rows)
    scan_game_csv(csv_path, [metric], parser_for(binder), block_size=block_size)
    frame = pd.read_parquet(metric.finalize())
    return frame if index is None else frame.set_index(index)
