import csv
from pathlib import Path
from uuid import UUID

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.ygoprodeck.ingestion_stage import (
    YgoprodeckCardIngestionStage,
    _lean_card_content,
    _link_arrows,
    _stat,
    _variants_by_original,
    _whole_number,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_YGO = GameId.YUGIOH
_COLUMNS = (
    "id", "name", "type", "desc", "atk", "def", "level", "race",
    "attribute", "scale", "linkval", "linkmarkers", "archetype", "image_url",
)  # fmt: skip


def _row(
    passcode: str, name: str, card_type: str = "Effect Monster", **cells: str
) -> dict[str, str]:
    defaults = {
        **dict.fromkeys(_COLUMNS, ""),
        "id": passcode,
        "name": name,
        "type": card_type,
        "desc": f"{name} text.",
        "race": "Warrior",
        "attribute": "LIGHT",
        "level": "4.0",
        "atk": "1800.0",
        "def": "1000.0",
        "archetype": "Some Archetype",
        "image_url": "https://images.ygoprodeck.com/x.jpg",
    }
    return {**defaults, **cells}


def _write_mirror(
    raw_dir: Path,
    rows: list[dict[str, str]],
    variants: tuple[tuple[str, str], ...] = (),
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "cards.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    (raw_dir / "cards_variants.csv").write_text(
        "original,variant\n" + "".join(f"{o},{v}\n" for o, v in variants),
        encoding="utf-8",
    )
    (raw_dir / "revision.json").write_text("{}", encoding="utf-8")
    return raw_dir


def _uuid(binder: CardBinder, passcode: str) -> UUID | None:
    card = binder.get_by_alias(_YGO, DataSource.YGOPRODECK, passcode)
    return card.nocab_uuid if card else None


class TestIngest:
    def test_one_card_per_row_aliased_by_passcode(self, tmp_path: Path) -> None:
        binder = CardBinder.load([])
        raw = _write_mirror(
            tmp_path,
            [_row("46986414", "Dark Magician"), _row("08512558", "Pot of Greed")],
        )

        result = YgoprodeckCardIngestionStage().ingest(raw, binder)

        assert len(result) == 2
        magician = binder.get_by_alias(_YGO, DataSource.YGOPRODECK, "46986414")
        assert magician is not None and magician.name == "Dark Magician"
        assert magician.provenance.data_source is DataSource.YGOPRODECK
        assert magician.provenance.source_id == "46986414"
        assert _uuid(binder, "8512558") is not None  # leading zero dropped

    def test_tokens_and_skill_cards_are_not_cards(self, tmp_path: Path) -> None:
        binder = CardBinder.load([])
        raw = _write_mirror(
            tmp_path,
            [
                _row("1", "Kept"),
                _row("2", "Sheep Token", "Token"),
                _row("300302022", "Ancient Fusion", "Skill Card"),
            ],
        )

        YgoprodeckCardIngestionStage().ingest(raw, binder)

        assert _uuid(binder, "1") is not None
        assert _uuid(binder, "2") is None and _uuid(binder, "300302022") is None

    def test_variants_alias_to_their_original_with_the_policy_edge_cases(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder.load([])
        raw = _write_mirror(
            tmp_path,
            [
                _row("10", "Original"),
                _row("20", "Other"),
                _row("30", "A Token", "Token"),
            ],
            variants=(
                ("10", "11"),  # an alternate artwork
                ("20", "11"),  # listed again under another original: first wins
                ("10", "20"),  # another row's own id: skipped
                ("30", "31"),  # original not kept: unused
                ("99", "98"),  # original not in cards.csv: unused
            ),
        )

        YgoprodeckCardIngestionStage().ingest(raw, binder)

        assert _uuid(binder, "11") == _uuid(binder, "10")
        assert _uuid(binder, "20") != _uuid(binder, "10")
        assert _uuid(binder, "31") is None and _uuid(binder, "98") is None

    def test_a_former_variant_listed_first_never_takes_its_originals_uuid(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder.load([])
        stage = YgoprodeckCardIngestionStage()
        stage.ingest(
            _write_mirror(tmp_path / "a", [_row("10", "Original")], (("10", "11"),)),
            binder,
        )
        original = _uuid(binder, "10")

        stage.ingest(
            _write_mirror(
                tmp_path / "b", [_row("11", "Now Its Own Card"), _row("10", "Original")]
            ),
            binder,
        )

        assert _uuid(binder, "10") == original
        own_card = binder.get_by_alias(_YGO, DataSource.YGOPRODECK, "11")
        assert own_card is not None and own_card.nocab_uuid != original
        assert own_card.name == "Now Its Own Card"

    def test_a_rerun_keeps_uuids_changes_edited_cards_and_drops_removed_ones(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder.load([])
        stage = YgoprodeckCardIngestionStage()
        first = _write_mirror(
            tmp_path / "a", [_row("1", "Same"), _row("2", "Edited"), _row("3", "Gone")]
        )
        stage.ingest(first, binder)
        same, edited, gone = (_uuid(binder, p) for p in ("1", "2", "3"))

        second = _write_mirror(
            tmp_path / "b",
            [_row("1", "Same"), _row("2", "Edited", atk="2000.0")],
        )
        result = stage.ingest(second, binder)

        assert result == [edited]
        assert (_uuid(binder, "1"), _uuid(binder, "2")) == (same, edited)
        assert gone is not None and binder.get_by_uuid(gone) is None
        assert set(binder.all_uuids(_YGO)) == {same, edited}

    def test_a_table_without_deck_cards_raises(self, tmp_path: Path) -> None:
        raw = _write_mirror(tmp_path, [_row("2", "Sheep Token", "Token")])
        with pytest.raises(ValueError, match="no deck cards"):
            YgoprodeckCardIngestionStage().ingest(raw, CardBinder.load([]))

    def test_an_incomplete_mirror_raises(self, tmp_path: Path) -> None:
        raw = _write_mirror(tmp_path, [_row("1", "Kept")])
        (raw / "revision.json").unlink()
        with pytest.raises(FileNotFoundError):
            YgoprodeckCardIngestionStage().ingest(raw, CardBinder.load([]))

    def test_a_header_without_a_needed_column_raises(self, tmp_path: Path) -> None:
        raw = _write_mirror(tmp_path, [_row("1", "Kept")])
        (raw / "cards.csv").write_text("id,name\n1,Kept\n", encoding="utf-8")
        with pytest.raises(ValueError, match="column"):
            YgoprodeckCardIngestionStage().ingest(raw, CardBinder.load([]))

    def test_a_bad_cell_names_its_row(self, tmp_path: Path) -> None:
        raw = _write_mirror(tmp_path, [_row("1", "Kept", level="4.5")])
        with pytest.raises(ValueError, match="id '1'"):
            YgoprodeckCardIngestionStage().ingest(raw, CardBinder.load([]))


class TestLeanCardContent:
    def test_an_xyz_monster_has_a_rank_and_metadata_is_dropped(self) -> None:
        content = _lean_card_content(
            _row("84013237", "Number 39: Utopia", "XYZ Monster", desc="2 Level 4\r\nx ")
        )
        assert content == {
            "name": "Number 39: Utopia",
            "type": "XYZ Monster",
            "attribute": "LIGHT",
            "monster_type": "Warrior",
            "rank": 4,
            "atk": 1800,
            "def": 1000,
            "text": "2 Level 4\nx",
        }

    def test_a_link_monster_has_rating_and_arrows_but_no_level(self) -> None:
        content = _lean_card_content(
            _row(
                "1861629",
                "Decode Talker",
                "Link Monster",
                level="0.0",
                linkval="3.0",
                linkmarkers="['Top', 'Bottom-Left']",
                **{"def": ""},
            )
        )
        assert list(content) == [
            "name", "type", "attribute", "monster_type",
            "link_rating", "link_arrows", "atk", "text",
        ]  # fmt: skip
        assert content["link_rating"] == 3
        assert content["link_arrows"] == ["Top", "Bottom-Left"]

    def test_a_pendulum_keeps_level_scale_and_its_sections(self) -> None:
        desc = "[ Pendulum Effect ] \nA.\n\n[ Monster Effect ] \nB."
        content = _lean_card_content(
            _row("1", "Odd-Eyes", "Pendulum Effect Monster", scale="4.0", desc=desc)
        )
        assert (content["level"], content["scale"], content["text"]) == (4, 4, desc)

    def test_a_spell_has_a_property_and_no_stats(self) -> None:
        content = _lean_card_content(
            _row(
                "55144522", "Pot of Greed", "Spell Card",
                race="Quick-Play", attribute="", level="", atk="", desc="Draw 2 cards.",
                **{"def": ""},
            )  # fmt: skip
        )
        assert content == {
            "name": "Pot of Greed",
            "type": "Spell Card",
            "property": "Quick-Play",
            "text": "Draw 2 cards.",
        }

    def test_an_unknown_stat_reads_as_a_question_mark_and_zero_stays(self) -> None:
        content = _lean_card_content(_row("1", "X", atk="-1.0", **{"def": "0.0"}))
        assert (content["atk"], content["def"]) == ("?", 0)


class TestCells:
    def test_numbers(self) -> None:
        assert _whole_number("4.0") == 4
        assert _whole_number(" ") is None
        assert _stat("-1.0") == "?" and _stat("") is None and _stat("0.0") == 0
        with pytest.raises(ValueError):
            _whole_number("4.5")
        with pytest.raises(ValueError):
            _whole_number("four")

    def test_link_arrows(self) -> None:
        assert _link_arrows("['Top', 'Left']") == ["Top", "Left"]
        assert _link_arrows("") == []
        for bad in ("Top", "[1, 2]", "__import__('os')"):
            with pytest.raises(ValueError):
                _link_arrows(bad)

    def test_variants_by_original_reads_only_kept_originals(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "cards_variants.csv"
        path.write_text("original,variant\n01,02\n01,03\n05,06\n", encoding="utf-8")
        assert _variants_by_original(path, {"1"}) == {"1": ("2", "3")}
