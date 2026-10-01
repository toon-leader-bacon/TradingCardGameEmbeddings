"""Where the isotropic metrics' private DeckBox lives.

scripts/run_metrics.py's isotropic runs write it (partial decks, final
decks and kingdoms); deck-level isotropic dojos read it back to verify
each metric was built from the same box.
"""

from pathlib import Path

ISOTROPIC_DECK_BOX_PATH = Path("data/metrics/isotropic/deck_box.db")
