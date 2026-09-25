import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.play_gwent.leader_deck_counts import (
    DEFAULT_RAW_PATH,
    count_decks_per_leader,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _gwent_one_card(card_id: int, name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=str(card_id),
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _card_binder(cards: dict[int, str]) -> CardBinder:
    binder = CardBinder()
    for card_id, name in cards.items():
        card = _gwent_one_card(card_id, name)
        binder.create(card)
        binder.register_alias(
            GameId.GWENT, DataSource.GWENT_ONE, str(card_id), card.nocab_uuid
        )
    return binder


def _guides_jsonl(guides: list[dict], tmp_path: Path) -> Path:
    path = tmp_path / "guides.jsonl"
    with open(path, "w", encoding="utf-8") as raw_file:
        for guide in guides:
            raw_file.write(json.dumps(guide) + "\n")
    return path


def _guide(guide_id: int, leader_id: int | None) -> dict:
    row: dict = {"id": guide_id}
    if leader_id is not None:
        row["leaderId"] = leader_id
    return row


def test_counts_one_guide_per_leader(tmp_path: Path) -> None:
    binder = _card_binder({1: "Pincer Maneuver", 2: "Rain of Fire"})
    path = _guides_jsonl([_guide(1, 1), _guide(2, 2)], tmp_path)

    counts = count_decks_per_leader(binder, path)

    assert counts == {"Pincer Maneuver": 1, "Rain of Fire": 1}


def test_counts_repeated_leader_across_guides(tmp_path: Path) -> None:
    binder = _card_binder({1: "Pincer Maneuver", 2: "Rain of Fire"})
    path = _guides_jsonl([_guide(1, 1), _guide(2, 1), _guide(3, 2)], tmp_path)

    counts = count_decks_per_leader(binder, path)

    assert counts == {"Pincer Maneuver": 2, "Rain of Fire": 1}


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    binder = _card_binder({1: "Pincer Maneuver"})
    path = tmp_path / "guides.jsonl"
    path.write_text(json.dumps(_guide(1, 1)) + "\n\n\n", encoding="utf-8")

    counts = count_decks_per_leader(binder, path)

    assert counts == {"Pincer Maneuver": 1}


def test_missing_leader_id_is_skipped(tmp_path: Path) -> None:
    binder = _card_binder({1: "Pincer Maneuver"})
    path = _guides_jsonl([_guide(1, None), _guide(2, 1)], tmp_path)

    counts = count_decks_per_leader(binder, path)

    assert counts == {"Pincer Maneuver": 1}


def test_unresolvable_leader_id_is_skipped(tmp_path: Path) -> None:
    binder = _card_binder({1: "Pincer Maneuver"})
    path = _guides_jsonl([_guide(1, 999999), _guide(2, 1)], tmp_path)

    counts = count_decks_per_leader(binder, path)

    assert counts == {"Pincer Maneuver": 1}


def test_default_raw_path() -> None:
    assert DEFAULT_RAW_PATH == Path("data/raw/play_gwent/guides.jsonl")
