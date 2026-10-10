"""After the rebuild: every deck_uuid the game_data deck metrics wrote
(all sets, all formats) must be a deck in data/final/decks/mtg.db.
Prints counts; exits 1 on any miss."""

import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.slice_file import SeventeenLandsSliceFile
from src.dojos.seventeenlands.sliced_dojos import ALL_DATA
from src.schema.game_id import GameId

slice_path = SeventeenLandsSliceFile(DeckWinPredictionMetric).build(ALL_DATA)
deck_column = pq.read_table(slice_path, columns=["deck_uuid"]).column(0)
metric_uuids = set(deck_column.to_pylist())

box = DeckBox.load([Path("data/final/decks/mtg.db")])
box_uuids = {str(deck_uuid) for deck_uuid in box.all_uuids(GameId.MTG)}

missing = metric_uuids - box_uuids
print(f"game rows: {len(deck_column):,}")
print(f"distinct metric decks: {len(metric_uuids):,}")
print(f"decks in mtg.db: {len(box_uuids):,}")
print(f"metric decks missing from mtg.db: {len(missing):,}")
sys.exit(1 if missing else 0)
