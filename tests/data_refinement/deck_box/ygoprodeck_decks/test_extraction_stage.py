import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.ygoprodeck_decks.extraction_stage import (
    PasscodeMatcher,
    YgoprodeckDecksExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_YGO = GameId.YUGIOH
_COLUMNS = ["deck_num", "deck_name", "format", "main_deck", "extra_deck", "side_deck"]


def _binder() -> tuple[CardBinder, dict[str, UUID]]:
    binder = CardBinder()
    uuids: dict[str, UUID] = {}
    for passcode, name in [
        ("1", "Main Card"),
        ("2", "Other Card"),
        ("3", "Extra Card"),
    ]:
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=_YGO,
            name=name,
            raw_content={"name": name},
            provenance=Provenance(
                data_source=DataSource.TCGJSON,
                source_id=passcode,
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder.create(card)
        binder.register_alias(_YGO, DataSource.YGOPRODECK, passcode, card.nocab_uuid)
        uuids[passcode] = card.nocab_uuid
    binder.ensure_unknown_card(_YGO)
    return binder, uuids


def _row(
    deck_num: str,
    main: list[str],
    extra: list[str] = [],
    side: list[str] = [],
    name: str = "A Deck",
) -> dict[str, str]:
    return {
        "deck_num": deck_num,
        "deck_name": name,
        "format": "Tournament Meta Decks",
        "main_deck": json.dumps(main),
        "extra_deck": json.dumps(extra) if extra else "",
        "side_deck": json.dumps(side) if side else "",
    }


def _write_mirror(raw_dir: Path, shards: dict[str, list[dict[str, str]]]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for shard_name, rows in shards.items():
        with open(raw_dir / shard_name, "w", encoding="utf-8", newline="") as shard:
            writer = csv.DictWriter(shard, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    (raw_dir / "revision.json").write_text("{}", encoding="utf-8")
    return raw_dir


_MAIN_40 = ["1"] * 3 + ["2"] * 37


class TestExtract:
    def test_stores_main_plus_extra_and_drops_the_side_deck(
        self, tmp_path: Path
    ) -> None:
        binder, uuids = _binder()
        raw = _write_mirror(
            tmp_path,
            {"00000001.csv": [_row("10", _MAIN_40, extra=["3"] * 2, side=["1"])]},
        )
        box = DeckBox.load([])

        created = YgoprodeckDecksExtractionStage().extract(raw, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        assert Counter(deck.card_nocab_uuids) == Counter(
            {uuids["1"]: 3, uuids["2"]: 37, uuids["3"]: 2}
        )
        assert deck.name == "A Deck"
        assert deck.provenance is not None
        assert deck.provenance.data_source == DataSource.YGOPRODECK_DECKS
        assert deck.provenance.source_id == "10"

    def test_an_identical_list_is_stored_once_and_a_rerun_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _binder()
        raw = _write_mirror(
            tmp_path,
            {
                "00000001.csv": [_row("10", _MAIN_40)],
                "00000002.csv": [
                    _row("20", list(reversed(_MAIN_40)), name="Copy of A Deck")
                ],
            },
        )
        box = DeckBox.load([])

        first = YgoprodeckDecksExtractionStage().extract(raw, box, binder)
        second = YgoprodeckDecksExtractionStage().extract(raw, box, binder)

        assert len(first) == 1 and second == []
        deck = box.get_by_uuid(first[0])
        assert deck is not None and deck.name == "A Deck"

    @pytest.mark.parametrize(
        ("main", "extra"),
        [(["1"] * 39, []), (["1"] * 61, []), (_MAIN_40, ["3"] * 16), ([], [])],
    )
    def test_skips_decks_outside_the_construction_rules(
        self, tmp_path: Path, main: list[str], extra: list[str]
    ) -> None:
        binder, _ = _binder()
        raw = _write_mirror(tmp_path, {"00000001.csv": [_row("10", main, extra=extra)]})

        assert (
            YgoprodeckDecksExtractionStage().extract(raw, DeckBox.load([]), binder)
            == []
        )

    def test_accepts_the_size_boundaries(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        raw = _write_mirror(
            tmp_path,
            {
                "00000001.csv": [
                    _row("10", ["1"] * 40),
                    _row("11", ["1"] * 60, extra=["3"] * 15),
                ]
            },
        )

        created = YgoprodeckDecksExtractionStage().extract(
            raw, DeckBox.load([]), binder
        )

        assert len(created) == 2

    def test_skips_malformed_rows_and_keeps_going(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        bad_json = _row("10", _MAIN_40) | {"main_deck": "[not json"}
        bad_passcode = _row("11", ["x"] * 40)
        no_num = _row("", _MAIN_40)
        raw = _write_mirror(
            tmp_path,
            {"00000001.csv": [bad_json, bad_passcode, no_num, _row("12", ["1"] * 40)]},
        )

        created = YgoprodeckDecksExtractionStage().extract(
            raw, DeckBox.load([]), binder
        )

        assert len(created) == 1

    def test_unmatched_passcodes_become_the_unknown_card(self, tmp_path: Path) -> None:
        binder, uuids = _binder()
        raw = _write_mirror(
            tmp_path, {"00000001.csv": [_row("10", ["1"] * 38 + ["0999", "999"])]}
        )
        box = DeckBox.load([])

        created = YgoprodeckDecksExtractionStage().extract(raw, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        unknown = CardBinder.unknown_card_uuid(_YGO)
        assert Counter(deck.card_nocab_uuids) == Counter({uuids["1"]: 38, unknown: 2})

    def test_a_blank_name_falls_back_to_the_deck_num(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        raw = _write_mirror(
            tmp_path, {"00000001.csv": [_row("10", _MAIN_40, name=" ")]}
        )
        box = DeckBox.load([])

        created = YgoprodeckDecksExtractionStage().extract(raw, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "10"

    def test_raises_on_an_incomplete_mirror_or_no_shard(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        with pytest.raises(FileNotFoundError, match="revision.json"):
            YgoprodeckDecksExtractionStage().extract(tmp_path, DeckBox.load([]), binder)
        (tmp_path / "revision.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="shard"):
            YgoprodeckDecksExtractionStage().extract(tmp_path, DeckBox.load([]), binder)

    def test_raises_without_the_unknown_sentinel(self, tmp_path: Path) -> None:
        raw = _write_mirror(tmp_path, {"00000001.csv": [_row("10", _MAIN_40)]})
        with pytest.raises(RuntimeError, match="Unknown"):
            YgoprodeckDecksExtractionStage().extract(
                raw, DeckBox.load([]), CardBinder()
            )


class TestPasscodeMatcher:
    def test_caches_and_counts_misses_per_copy(self) -> None:
        binder, uuids = _binder()
        unknown = CardBinder.unknown_card_uuid(_YGO)
        matcher = PasscodeMatcher(binder, unknown)

        result = matcher.match_card_nocab_uuids(("1", "999", "999"))

        assert result == [uuids["1"], unknown, unknown]
        assert matcher.unmatched_counts == Counter({"999": 2})


class TestLongFields:
    def test_a_description_over_csvs_default_limit_is_read(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _binder()
        raw = tmp_path / "decks"
        raw.mkdir()
        columns = _COLUMNS + ["deck_description"]
        with open(raw / "00000001.csv", "w", encoding="utf-8", newline="") as shard:
            writer = csv.DictWriter(shard, fieldnames=columns)
            writer.writeheader()
            writer.writerow(_row("10", _MAIN_40) | {"deck_description": "x" * 200_000})
        (raw / "revision.json").write_text("{}", encoding="utf-8")
        limit_before = csv.field_size_limit()

        created = YgoprodeckDecksExtractionStage().extract(
            raw, DeckBox.load([]), binder
        )

        assert len(created) == 1
        assert csv.field_size_limit() == limit_before

    def test_ignores_files_that_are_not_shards(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        raw = _write_mirror(tmp_path, {"00000001.csv": [_row("10", _MAIN_40)]})
        (raw / "0abc.csv").write_text("not,a,shard", encoding="utf-8")

        created = YgoprodeckDecksExtractionStage().extract(
            raw, DeckBox.load([]), binder
        )

        assert len(created) == 1
