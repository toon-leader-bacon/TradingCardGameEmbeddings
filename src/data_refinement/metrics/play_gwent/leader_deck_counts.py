"""Counts how many playgwent.com guide decks use each leader.

RAW SHAPE: src/data_retrieval/play_gwent/downloader.py's guides.jsonl —
one playgwent.com "guide" JSON object per line. Each guide carries its
own top-level "leaderId" (int, gwent.one's card template id for the
deck's leader). Matching LeaderMaskedFromDeckMetric's own resolution
path (leader_masked_from_deck_metric.py's _target_card_uuid_for_row/
_label_for_card), a guide's leader identity here is the gwent.one
CardBinder's canonical name for that "leaderId" - not the guide's own
embedded "leader"."name" text. The two can disagree (playgwent.com and
gwent.one have independently misspelled/renamed at least one leader
each, e.g. "Reckless Fury" vs. "Reckless Flurry") - counting by guide
text alone previously produced a LEADER_NAMES vocabulary that didn't
match what the metric checks card names against.

PURPOSE: this is metric-design tooling, not a metric itself - it exists
to answer "how many distinct leaders are there, and how skewed is their
frequency" before committing to a fixed LEADER_NAMES tuple for
leader_masked_from_deck_metric.py, using the exact same name source
that metric validates against.

STREAMING: guides.jsonl is large (observed ~5GB, ~60k lines) - counting
reads it one line at a time, matching every other guides.jsonl reader
in this project (never loads the whole file into memory).
"""

import json
from collections import Counter
from pathlib import Path

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_retrieval.play_gwent.downloader import PlayGwentDownloader
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

DEFAULT_RAW_PATH: Path = PlayGwentDownloader.DEFAULT_RAW_DATA_DIR / "guides.jsonl"


def count_decks_per_leader(
    card_lookup: CardLookup, raw_path: Path | None = None
) -> dict[str, int]:
    """Count guide decks per leader's canonical gwent.one card name.

    Inputs:
        card_lookup: used to look up each guide's "leaderId" against
            gwent.one's registered aliases (DataSource.GWENT_ONE) - the
            same lookup LeaderMaskedFromDeckMetric itself uses.
        raw_path: path to a guides.jsonl file (one JSON guide object
            per line, each carrying "leaderId"). Defaults to
            DEFAULT_RAW_PATH when None.
    Output: a dict mapping each leader's canonical CardBinder name
        (card.raw_content["name"]) to the number of guide decks in
        raw_path whose "leaderId" resolves to that card. A guide whose
        "leaderId" is missing or doesn't resolve via card_lookup is
        skipped, not counted - mirrors
        LeaderMaskedFromDeckMetric._target_card_uuid_for_row()'s own
        skip policy, since an unresolvable leader is a real,
        expected data-quality edge case here too.
    Side effects: reads raw_path, one line at a time (see module
        docstring's STREAMING section).
    Exceptions: raises if raw_path doesn't exist. A line that isn't
        valid JSON is silently skipped rather than raising.

    Example:
        >>> counts = count_decks_per_leader(card_binder)
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
            except json.JSONDecodeError:
                continue
            leader_id = guide.get("leaderId")
            if leader_id is None:
                continue
            card = card_lookup.get_by_alias(
                GameId.GWENT, DataSource.GWENT_ONE, str(leader_id)
            )
            if card is None:
                continue
            counts[card.raw_content["name"]] += 1
    return dict(counts)
