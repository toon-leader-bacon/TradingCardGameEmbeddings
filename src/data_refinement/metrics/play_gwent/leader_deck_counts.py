"""Counts how many playgwent.com guide decks use each leader.

RAW SHAPE: src/data_retrieval/play_gwent/downloader.py's guides.jsonl —
one playgwent.com "guide" JSON object per line. Each guide carries its
own top-level "leaderId" (int, gwent.one's card template id for the
deck's leader) and a full embedded "leader" object with a "name" field
(e.g. "Pincer Maneuver") — no card_lookup/CardBinder join is needed to
read a guide's leader identity, unlike deck_box's own extraction stage
(src/data_refinement/deck_box/play_gwent/extraction_stage.py), which
resolves every OTHER deck slot via gwent.one aliases.

PURPOSE: this is metric-design tooling, not a metric itself — it exists
to answer "how many distinct leaders are there, and how skewed is their
frequency" before committing to a fixed LABEL_VALUES tuple (or a
dynamically-derived one) for a future leader-identity classification
metric in this container (see BRAINSTORM.md's "Leader Prediction from
Deck" entry).

STREAMING: guides.jsonl is large (observed ~5GB, ~60k lines) — counting
reads it one line at a time, matching every other guides.jsonl reader
in this project (never loads the whole file into memory).
"""

import json
from collections import Counter
from pathlib import Path

from src.data_retrieval.play_gwent.downloader import PlayGwentDownloader

DEFAULT_RAW_PATH: Path = PlayGwentDownloader.DEFAULT_RAW_DATA_DIR / "guides.jsonl"


def count_decks_per_leader(raw_path: Path | None = None) -> dict[str, int]:
    """Count guide decks per leader name across a guides.jsonl file.

    Inputs:
        raw_path: path to a guides.jsonl file (one JSON guide object
            per line, each carrying "leader"."name"). Defaults to
            DEFAULT_RAW_PATH when None.
    Output: a dict mapping each leader's human-readable card name
        (e.g. "Pincer Maneuver") to the number of guide decks in
        raw_path that use that leader. Keys cover exactly the distinct
        leader names seen; a leader with zero guides is absent, not
        zero.
    Side effects: reads raw_path, one line at a time (see module
        docstring's STREAMING section).
    Exceptions: raises if raw_path doesn't exist or is missing
        "leader"."name" on a line that does parse. A line that isn't
        valid JSON is silently skipped rather than raising.

    Example:
        >>> counts = count_decks_per_leader()
        >>> counts["Pincer Maneuver"]
        413
    """
    path = raw_path or DEFAULT_RAW_PATH

    counts: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as raw_file:
        for line in raw_file:
            if not line.strip():
                continue
            try:
                guide = json.loads(line)
                counts[guide["leader"]["name"]] += 1
            except json.JSONDecodeError:
                continue
    return dict(counts)
