"""Where replay_data's metrics-private DeckBox lives.

scripts/run_metrics.py's replay_data run writes it (combat aggression
profile stores each game's deck there); that metric's dojo reads it back.
game_data has no private box: its deck dojos read the canonical MTG box
(data/final/decks/mtg.db). draft_data has no decks.
"""

from pathlib import Path

REPLAY_DATA_DECK_BOX_PATH = Path("data/metrics/seventeenlands/replay_data/deck_box.db")
