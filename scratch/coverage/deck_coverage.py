"""How many game_data metric deck_uuids (per game row) exist in a canonical
deck box built by the extraction stage from the same CSV."""
import sys
from collections import Counter
from pathlib import Path

import pyarrow.csv as pa_csv

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.seventeenlands_game_data.extraction_stage import (
    SeventeenLandsGameDataDeckExtractionStage,
)
from src.data_refinement.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)
from src.schema.game_id import GameId

csv_path = Path(sys.argv[1])
binder = CardBinder.load([CardBinder.default_output_path(GameId.MTG)])
box = DeckBox.load([])
canonical = set(SeventeenLandsGameDataDeckExtractionStage().extract(csv_path, box, binder))
canonical_str = {str(u) for u in canonical}

header = pa_csv.open_csv(csv_path).schema.names
parser = GameDataChunkParser.from_header(header, binder, GameId.MTG)
reader = pa_csv.open_csv(csv_path, convert_options=pa_csv.ConvertOptions(
    include_columns=parser.needed_columns(), column_types=parser.column_types()))
rows = hit = 0
distinct_rows: Counter = Counter()
for batch in reader:
    chunk = parser.parse(batch)
    for u in chunk.decks.row_deck_uuids():
        rows += 1
        found = str(u) in canonical_str
        hit += found
        distinct_rows[found] += 1
print(f"{csv_path.name}: canonical decks={len(canonical)}  game rows={rows}  rows whose deck is in the canonical box={hit} ({hit/rows:.1%})")
