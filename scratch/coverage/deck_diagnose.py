"""Split the coverage gap: within each draft, does the metric deck of the
canonical game (lowest build_index/match/game) equal the box's deck? And how
often do a draft's games use more than one decklist?"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.csv as pa_csv

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.seventeenlands_game_data.extraction_stage import (
    SeventeenLandsGameDataDeckExtractionStage,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)
from src.schema.game_id import GameId

csv_path = Path(sys.argv[1])
binder = CardBinder.load([CardBinder.default_output_path(GameId.MTG)])
box = DeckBox.load([])
SeventeenLandsGameDataDeckExtractionStage().extract(csv_path, box, binder)
box_by_draft = {}
for u in box.all_uuids(GameId.MTG):
    deck = box.get_by_uuid(u)
    box_by_draft[deck.provenance.source_id.split(":")[0]] = (str(u), sorted(map(str, deck.card_nocab_uuids)))

header = pa_csv.open_csv(csv_path).schema.names
parser = GameDataChunkParser.from_header(header, binder, GameId.MTG)
reader = pa_csv.open_csv(csv_path, convert_options=pa_csv.ConvertOptions(
    include_columns=parser.needed_columns(), column_types=parser.column_types()))
builds = pd.read_csv(csv_path, usecols=["draft_id", "build_index", "match_number", "game_number"])
recs = []
for batch in reader:
    c = parser.parse(batch)
    for i, u in enumerate(c.decks.row_deck_uuids()):
        d = c.decks.decks[c.decks.row_deck[i]]
        recs.append((str(c.keys.draft_id[i]), str(u), len(d.card_nocab_uuids)))
m = pd.DataFrame(recs, columns=["draft_id", "deck_uuid", "n_cards"])
m["build_index"] = builds["build_index"].values
per_draft = m.groupby("draft_id")["deck_uuid"].nunique()
print("drafts:", len(per_draft), " drafts with >1 metric decklist:", int((per_draft > 1).sum()))
canon_rows = m.sort_values(["draft_id", "build_index"]).groupby("draft_id").head(1)
same = sum(box_by_draft[r.draft_id][0] == r.deck_uuid for r in canon_rows.itertuples())
print("canonical-game rows whose metric deck == box deck:", same, "of", len(canon_rows))
# size difference hints at unmatched columns dropped by the metric parser
diff = [(len(box_by_draft[r.draft_id][1]) - r.n_cards) for r in canon_rows.itertuples() if box_by_draft[r.draft_id][0] != r.deck_uuid]
print("for mismatches, box deck size minus metric deck size (first 15):", diff[:15])
unknown = [u for u in box.all_uuids(GameId.MTG) if any(binder.get_by_uuid(c) is None or binder.get_by_uuid(c).name.lower().startswith("unknown") for c in box.get_by_uuid(u).card_nocab_uuids)]
print("box decks containing the Unknown sentinel:", len(unknown), "of", len(box_by_draft))
