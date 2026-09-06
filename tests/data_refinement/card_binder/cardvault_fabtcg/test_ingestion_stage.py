import csv
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.cardvault_fabtcg.ingestion_stage import (
    CardVaultFabtcgCardIngestionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_FIELDNAMES = [
    "print_id",
    "card_id",
    "print_language",
    "rarity",
    "set_code",
    "face_1_true_name",
    "face_2_true_name",
]

_REAL_ROW = {
    "print_id": "MST131",
    "card_id": "10000-year-reunion-1",
    "print_language": "en",
    "rarity": "majestic",
    "set_code": "MST",
    "face_1_true_name": "10,000 Year Reunion",
    "face_2_true_name": "",
}


def _write_csv(path: Path, rows: list[dict]) -> Path:
    csv_path = path / "public_card_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TestIngest:
    def test_creates_one_card_per_card_id(self, tmp_path: Path) -> None:
        csv_path = _write_csv(
            tmp_path,
            [
                {
                    **_REAL_ROW,
                    "print_id": "MST131",
                    "card_id": "card-a",
                    "face_1_true_name": "A",
                },
                {
                    **_REAL_ROW,
                    "print_id": "MST132",
                    "card_id": "card-b",
                    "face_1_true_name": "B",
                },
            ],
        )
        binder = CardBinder()

        changed = CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        assert len(changed) == 2
        assert set(binder.all_uuids(GameId.FLESH_AND_BLOOD)) == set(changed)

    def test_maps_card_id_to_provenance_source_id(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card is not None
        assert card.provenance.source_id == "10000-year-reunion-1"
        assert card.provenance.data_source == DataSource.CARDVAULT_FABTCG

    def test_print_id_alias_resolves_to_same_card(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        by_card_id = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        by_print_id = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "MST131"
        )
        assert by_print_id is not None
        assert by_print_id.nocab_uuid == by_card_id.nocab_uuid

    def test_raw_content_is_the_entire_row_unmodified(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card.raw_content == _REAL_ROW

    def test_name_is_plain_face_1_true_name_when_no_second_face(
        self, tmp_path: Path
    ) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card.name == "10,000 Year Reunion"

    def test_dfc_name_is_joined_with_double_slash(self, tmp_path: Path) -> None:
        row = {
            **_REAL_ROW,
            "card_id": "a-drop-in-the-ocean-3--inner-chi-3",
            "face_1_true_name": "A Drop in the Ocean",
            "face_2_true_name": "Inner Chi",
        }
        csv_path = _write_csv(tmp_path, [row])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD,
            DataSource.CARDVAULT_FABTCG,
            "a-drop-in-the-ocean-3--inner-chi-3",
        )
        assert card.name == "A Drop in the Ocean // Inner Chi"

    def test_non_english_row_does_not_create_a_separate_card(
        self, tmp_path: Path
    ) -> None:
        csv_path = _write_csv(
            tmp_path,
            [
                _REAL_ROW,
                {
                    **_REAL_ROW,
                    "print_id": "DE_MST131",
                    "print_language": "de",
                    "face_1_true_name": "10,000 Year Reunion",
                },
            ],
        )
        binder = CardBinder()

        changed = CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        assert len(changed) == 1
        assert len(list(binder.all_uuids(GameId.FLESH_AND_BLOOD))) == 1

    def test_non_english_row_print_id_alias_resolves_to_same_card(
        self, tmp_path: Path
    ) -> None:
        csv_path = _write_csv(
            tmp_path,
            [
                _REAL_ROW,
                {
                    **_REAL_ROW,
                    "print_id": "DE_MST131",
                    "print_language": "de",
                    "face_1_true_name": "10,000 Year Reunion",
                },
            ],
        )
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        english_card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        german_print = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "DE_MST131"
        )
        assert german_print is not None
        assert german_print.nocab_uuid == english_card.nocab_uuid

    def test_non_english_row_never_becomes_canonical_content(
        self, tmp_path: Path
    ) -> None:
        # A German row with a longer serialized raw_content than the
        # English row must NOT win merge_strategies.keep_longer_content
        # — pass 2 never builds/merges a candidate at all.
        csv_path = _write_csv(
            tmp_path,
            [
                _REAL_ROW,
                {
                    **_REAL_ROW,
                    "print_id": "DE_MST131",
                    "print_language": "de",
                    "face_1_true_name": "Zehntausendjähriges Wiedersehen (much longer text)",
                },
            ],
        )
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card.raw_content == _REAL_ROW

    def test_multiple_prints_of_same_card_id_merge_via_keep_longer_content(
        self, tmp_path: Path
    ) -> None:
        sparse_row = {
            **_REAL_ROW,
            "print_id": "ENG131",
            "set_code": "ENG",
            "rarity": "",
        }
        csv_path = _write_csv(tmp_path, [sparse_row, _REAL_ROW])
        binder = CardBinder()

        changed = CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        assert len(list(binder.all_uuids(GameId.FLESH_AND_BLOOD))) == 1
        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card.raw_content == _REAL_ROW
        assert changed == [card.nocab_uuid, card.nocab_uuid]

    def test_raises_if_raw_path_does_not_exist(self, tmp_path: Path) -> None:
        binder = CardBinder()
        with pytest.raises(OSError):
            CardVaultFabtcgCardIngestionStage().ingest(tmp_path / "missing.csv", binder)


class TestReIngestDuplicates:
    def test_unchanged_row_is_a_noop_and_preserves_uuid(self, tmp_path: Path) -> None:
        binder = CardBinder()
        stage = CardVaultFabtcgCardIngestionStage()
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        stage.ingest(csv_path, binder)
        original = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )

        (tmp_path / "second").mkdir()
        second_path = _write_csv(tmp_path / "second", [_REAL_ROW])
        changed = stage.ingest(second_path, binder)

        reloaded = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert reloaded.nocab_uuid == original.nocab_uuid
        assert changed == []

    def test_richer_row_updates_content_in_place(self, tmp_path: Path) -> None:
        binder = CardBinder()
        stage = CardVaultFabtcgCardIngestionStage()
        sparse_row = {**_REAL_ROW, "rarity": ""}
        (tmp_path / "sparse").mkdir()
        sparse_path = _write_csv(tmp_path / "sparse", [sparse_row])
        stage.ingest(sparse_path, binder)
        original = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )

        (tmp_path / "richer").mkdir()
        richer_path = _write_csv(tmp_path / "richer", [_REAL_ROW])
        changed = stage.ingest(richer_path, binder)

        updated = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert updated.nocab_uuid == original.nocab_uuid
        assert updated.raw_content == _REAL_ROW
        assert changed == [original.nocab_uuid]


class TestRegisterPrintAliasGuard:
    def test_raises_if_card_id_never_seen_in_pass_one(self, tmp_path: Path) -> None:
        # Direct unit test of the private helper's own guard — the
        # public ingest() flow can never actually hit this branch
        # given the corpus invariant this module's docstring documents
        # (every card_id has an "en" row), so it's only reachable by
        # calling the helper directly with a violating case.
        binder = CardBinder()
        stage = CardVaultFabtcgCardIngestionStage()
        row = {**_REAL_ROW, "print_id": "DE_NEVER_SEEN", "print_language": "de"}

        with pytest.raises(ValueError):
            stage._register_print_alias(row, binder)
