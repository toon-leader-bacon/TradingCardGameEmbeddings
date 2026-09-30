"""Where the sts_gg metrics' private DeckBox lives.

scripts/run_metrics.py's run_sts_gg writes it; deck-level sts_gg dojos
read it back to verify each metric was built from the same box (see
sts_gg/README.md's "Deck references").
"""

from pathlib import Path

STS_GG_DECK_BOX_PATH = Path("data/metrics/sts_gg/deck_box.db")
