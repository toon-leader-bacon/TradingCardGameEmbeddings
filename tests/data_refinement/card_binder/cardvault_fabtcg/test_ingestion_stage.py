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
    "product_name",
    "face_1_face_id",
    "face_1_name",
    "face_1_true_name",
    "face_1_thumbnail_large",
    "face_1_artist",
    "face_1_finish_type",
    "face_1_true_pitch",
    "face_1_true_color",
    "face_1_true_cost",
    "face_1_true_defense",
    "face_1_true_typebox",
    "face_1_typebox",
    "face_1_true_textbox",
    "face_1_rules_text",
    "face_1_flavor_text",
    "face_1_types",
    "face_1_classes",
    "face_2_true_name",
    "face_2_true_typebox",
    "face_2_true_textbox",
]

_REAL_ROW = {
    "print_id": "MST131",
    "card_id": "10000-year-reunion-1",
    "print_language": "en",
    "rarity": "majestic",
    "set_code": "MST",
    "product_name": "Part the Mistveil",
    "face_1_face_id": "MST131",
    "face_1_name": "10,000 Year Reunion",
    "face_1_true_name": "10,000 Year Reunion",
    "face_1_thumbnail_large": "https://cdn.example.com/large/MST131.webp",
    "face_1_artist": "Faizal Fikri",
    "face_1_finish_type": "regular",
    "face_1_true_pitch": "1",
    "face_1_true_color": "red",
    "face_1_true_cost": "8",
    "face_1_true_defense": "3",
    "face_1_true_typebox": "Illusionist Action - Aura",
    "face_1_typebox": "Illusionist Action - Aura",
    "face_1_true_textbox": "Remove three counters.{br}**Ward 10**",
    "face_1_rules_text": "Remove three counters.{br}**Ward 10** (display copy)",
    "face_1_flavor_text": "_The power burns._",
    "face_1_types": "Action",
    "face_1_classes": "Illusionist",
    "face_2_true_name": "",
    "face_2_true_typebox": "",
    "face_2_true_textbox": "",
}

_LEAN_CONTENT = {
    "name": "10,000 Year Reunion",
    "typebox": "Illusionist Action - Aura",
    "rarity": "majestic",
    "pitch": "1",
    "color": "red",
    "cost": "8",
    "defense": "3",
    "textbox": "Remove three counters.\n**Ward 10**",
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

    def test_raw_content_is_lean_card_level_content(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card.raw_content == _LEAN_CONTENT

    def test_print_level_and_display_columns_are_dropped(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        text = str(card.raw_content)
        for dropped in (
            "MST131",
            "Part the Mistveil",
            "cdn.example.com",
            "Faizal Fikri",
            "display copy",
            "power burns",
            "Illusionist'",
        ):
            assert dropped not in text

    def test_line_break_markup_becomes_a_newline(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert "{br}" not in card.raw_content["textbox"]
        assert "\n" in card.raw_content["textbox"]

    def test_second_face_becomes_a_nested_back_face(self, tmp_path: Path) -> None:
        row = {
            **_REAL_ROW,
            "face_2_true_name": "Inner Chi",
            "face_2_true_typebox": "Mystic Resource - Chi",
            "face_2_true_textbox": "Gain 1 resource.",
        }
        csv_path = _write_csv(tmp_path, [row])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card.raw_content["back_face"] == {
            "name": "Inner Chi",
            "typebox": "Mystic Resource - Chi",
            "textbox": "Gain 1 resource.",
        }
        assert list(card.raw_content)[-1] == "back_face"

    def test_single_faced_card_has_no_back_face(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert "back_face" not in card.raw_content

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
        # A German row with different content must NOT replace the English
        # row's content — pass 2 never builds/merges a candidate at all.
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
        assert card.raw_content == _LEAN_CONTENT

    def test_multiple_english_prints_of_a_card_id_make_one_card(
        self, tmp_path: Path
    ) -> None:
        other_print = {
            **_REAL_ROW,
            "print_id": "ENG131",
            "set_code": "ENG",
            "rarity": "",
        }
        csv_path = _write_csv(tmp_path, [other_print, _REAL_ROW])
        binder = CardBinder()

        changed = CardVaultFabtcgCardIngestionStage().ingest(csv_path, binder)

        assert len(list(binder.all_uuids(GameId.FLESH_AND_BLOOD))) == 1
        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        # Print-level columns are not content, so the second print is a no-op
        assert card.raw_content == _LEAN_CONTENT
        assert changed == [card.nocab_uuid]
        for print_id in ("ENG131", "MST131"):
            assert (
                binder.get_by_alias(
                    GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, print_id
                )
                == card
            )

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

    def test_changed_card_content_replaces_in_place_keeping_uuid(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        stage = CardVaultFabtcgCardIngestionStage()
        old_row = {**_REAL_ROW, "face_1_true_textbox": "A long stale rules text."}
        (tmp_path / "old").mkdir()
        stage.ingest(_write_csv(tmp_path / "old", [old_row]), binder)
        original = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )

        (tmp_path / "new").mkdir()
        new_row = {**_REAL_ROW, "face_1_true_textbox": "New."}
        changed = stage.ingest(_write_csv(tmp_path / "new", [new_row]), binder)

        updated = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert updated.nocab_uuid == original.nocab_uuid
        assert updated.raw_content["textbox"] == "New."
        assert changed == [original.nocab_uuid]

    def test_row_differing_only_in_print_columns_is_a_noop(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        stage = CardVaultFabtcgCardIngestionStage()
        (tmp_path / "first").mkdir()
        stage.ingest(_write_csv(tmp_path / "first", [_REAL_ROW]), binder)

        (tmp_path / "second").mkdir()
        reprint = {
            **_REAL_ROW,
            "print_id": "EVR131",
            "set_code": "EVR",
            "product_name": "Everfest",
        }
        changed = stage.ingest(_write_csv(tmp_path / "second", [reprint]), binder)

        assert changed == []


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


class TestLeastRestrictiveRarity:
    def _rarity_of(self, tmp_path: Path, rarities: list[str], **extra: str) -> str:
        rows = [
            {**_REAL_ROW, "print_id": f"P{i}", "rarity": rarity, **extra}
            for i, rarity in enumerate(rarities)
        ]
        binder = CardBinder()
        CardVaultFabtcgCardIngestionStage().ingest(_write_csv(tmp_path, rows), binder)
        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        return card.raw_content["rarity"]

    def test_common_wins_over_marvel_and_promo_treatments(self, tmp_path: Path) -> None:
        assert self._rarity_of(tmp_path, ["marvel", "promo", "common"]) == "common"

    def test_a_card_only_printed_as_promo_is_promo(self, tmp_path: Path) -> None:
        assert self._rarity_of(tmp_path, ["promo", "promo"]) == "promo"

    def test_promo_and_rare_is_rare(self, tmp_path: Path) -> None:
        assert self._rarity_of(tmp_path, ["promo", "rare"]) == "rare"

    def test_result_does_not_depend_on_row_order(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()

        forward = self._rarity_of(tmp_path / "a", ["common", "marvel", "rare"])
        backward = self._rarity_of(tmp_path / "b", ["rare", "marvel", "common"])

        assert forward == backward == "common"

    def test_unlisted_rarity_ranks_after_every_listed_one(self, tmp_path: Path) -> None:
        assert self._rarity_of(tmp_path, ["brand-new", "gold-marvel"]) == "gold-marvel"

    def test_non_english_rarity_is_ignored(self, tmp_path: Path) -> None:
        rows = [
            _REAL_ROW,
            {
                **_REAL_ROW,
                "print_id": "DE_1",
                "print_language": "de",
                "rarity": "common",
            },
        ]
        binder = CardBinder()

        CardVaultFabtcgCardIngestionStage().ingest(_write_csv(tmp_path, rows), binder)

        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert card.raw_content["rarity"] == "majestic"

    def test_empty_rarities_are_skipped_and_all_empty_omits_the_key(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "some").mkdir()
        (tmp_path / "none").mkdir()

        assert self._rarity_of(tmp_path / "some", ["", "rare"]) == "rare"
        binder = CardBinder()
        rows = [{**_REAL_ROW, "rarity": ""}]
        CardVaultFabtcgCardIngestionStage().ingest(
            _write_csv(tmp_path / "none", rows), binder
        )
        card = binder.get_by_alias(
            GameId.FLESH_AND_BLOOD, DataSource.CARDVAULT_FABTCG, "10000-year-reunion-1"
        )
        assert "rarity" not in card.raw_content
