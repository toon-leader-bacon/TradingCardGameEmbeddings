import json
from pathlib import Path

from src.data_refinement.metrics.play_gwent.leader_deck_counts import (
    DEFAULT_RAW_PATH,
    count_decks_per_leader,
)


def _guides_jsonl(guides: list[dict], tmp_path: Path) -> Path:
    path = tmp_path / "guides.jsonl"
    with open(path, "w", encoding="utf-8") as raw_file:
        for guide in guides:
            raw_file.write(json.dumps(guide) + "\n")
    return path


def _guide(guide_id: int, leader_name: str) -> dict:
    return {"id": guide_id, "leader": {"name": leader_name}}


def test_counts_one_guide_per_leader(tmp_path: Path) -> None:
    path = _guides_jsonl(
        [_guide(1, "Pincer Maneuver"), _guide(2, "Rain of Fire")], tmp_path
    )

    counts = count_decks_per_leader(path)

    assert counts == {"Pincer Maneuver": 1, "Rain of Fire": 1}


def test_counts_repeated_leader_across_guides(tmp_path: Path) -> None:
    path = _guides_jsonl(
        [
            _guide(1, "Pincer Maneuver"),
            _guide(2, "Pincer Maneuver"),
            _guide(3, "Rain of Fire"),
        ],
        tmp_path,
    )

    counts = count_decks_per_leader(path)

    assert counts == {"Pincer Maneuver": 2, "Rain of Fire": 1}


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "guides.jsonl"
    path.write_text(
        json.dumps(_guide(1, "Pincer Maneuver")) + "\n\n\n", encoding="utf-8"
    )

    counts = count_decks_per_leader(path)

    assert counts == {"Pincer Maneuver": 1}


def test_default_raw_path() -> None:
    assert DEFAULT_RAW_PATH == Path("data/raw/play_gwent/guides.jsonl")
