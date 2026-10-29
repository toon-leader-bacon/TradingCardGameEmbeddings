"""Binder mods (src/training/binder_mods.py): the bake, its saved copy,
its stamp check, and CardShelf handing out the modded binder."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.mods.mod_specs import CardDropoutSpec, KeyCaseSpec, ShuffleKeysSpec
from src.dojos.mods.content_style_mods import KeyCase
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.training.binder_mods import (
    BinderModsMismatchError,
    ModdedCardBinder,
    modded_binder,
    require_writable_save_directory,
)
from src.training.binder_mods_spec import BinderModsSpec
from src.training.config_parsing.config_section import ConfigSection
from src.training.config_parsing.mod_parsing import parse_binder_mods
from src.training.dojo_recipes.card_shelf import CardShelf
from tests.dojos.mods.card_ids import content_uuid

_GAME = GameId.GWENT
_CAMEL = BinderModsSpec((KeyCaseSpec(styles=(KeyCase.CAMEL,)),))


def _card(raw_content: dict[str, Any]) -> GenericCard:
    """A Gwent card with a fixed id from its content."""
    return GenericCard(
        nocab_uuid=content_uuid(raw_content),
        source_game=_GAME,
        name=str(raw_content["card_name"]),
        raw_content=raw_content,
        provenance=Provenance(
            DataSource.GWENT_ONE, "test", datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


@pytest.fixture
def clean_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean Gwent binder of three cards plus the Unknown sentinel, at
    the path CardBinder.default_output_path answers."""
    binder = CardBinder()
    for index in range(3):
        binder.create(
            _card(
                {"card_name": f"Card {index}", "base_power": index, "flavor_text": "x"}
            )
        )
    binder.ensure_unknown_card(_GAME)
    path = tmp_path / "clean" / "gwent.jsonl"
    binder.save(path, _GAME)
    monkeypatch.setattr(CardBinder, "default_output_path", lambda game: path)
    return path


def _clean(path: Path) -> CardBinder:
    return CardBinder.load([path])


class TestBake:
    def test_every_card_but_the_sentinel_is_modded(self, clean_path: Path) -> None:
        binder = modded_binder(_GAME, _CAMEL)

        unknown = CardBinder.unknown_card_uuid(_GAME)
        clean = _clean(clean_path)
        for card in binder.all_cards(_GAME):
            original = clean.get_by_uuid(card.nocab_uuid)
            assert original is not None
            if card.nocab_uuid == unknown:
                assert card.raw_content == original.raw_content
            else:
                assert "cardName" in card.raw_content
                assert "card_name" not in card.raw_content

    def test_version_is_the_clean_binders(self, clean_path: Path) -> None:
        binder = modded_binder(_GAME, _CAMEL)

        assert binder.version_for(_GAME) == _clean(clean_path).version_for(_GAME)

    def test_another_games_version_is_the_empty_digest(self, clean_path: Path) -> None:
        binder = modded_binder(_GAME, _CAMEL)

        assert binder.version_for(GameId.MTG) == CardBinder().version_for(GameId.MTG)

    def test_clean_file_is_not_written(self, clean_path: Path) -> None:
        before = clean_path.read_bytes()
        modded_binder(_GAME, _CAMEL)

        assert clean_path.read_bytes() == before

    def test_unbounded_random_mod_is_reproducible_and_cacheable(
        self, clean_path: Path
    ) -> None:
        spec = BinderModsSpec((ShuffleKeysSpec(),), seed=3)
        first = modded_binder(_GAME, spec)
        second = modded_binder(_GAME, spec)

        for card in first.all_cards(_GAME):
            again = second.get_by_uuid(card.nocab_uuid)
            assert again is not None
            assert list(card.raw_content) == list(again.raw_content)
            assert card.cacheable

    def test_missing_clean_binder_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            CardBinder, "default_output_path", lambda game: tmp_path / "none.jsonl"
        )
        with pytest.raises(FileNotFoundError, match="run_card_binder_ingestion"):
            modded_binder(_GAME, _CAMEL)


class TestSavedBake:
    def _saved(self, directory: Path, **changes: Any) -> BinderModsSpec:
        return BinderModsSpec(
            changes.get("pipeline", _CAMEL.pipeline),
            seed=changes.get("seed", 0),
            save_directory=directory,
        )

    def test_save_writes_binder_ledger_and_stamp_only(
        self, clean_path: Path, tmp_path: Path
    ) -> None:
        directory = tmp_path / "bake"
        modded_binder(_GAME, self._saved(directory))

        assert sorted(path.name for path in directory.iterdir()) == [
            "gwent.alias_ledger.jsonl",
            "gwent.jsonl",
            "gwent.stamp.json",
        ]

    def test_saved_bake_is_loaded_without_the_clean_binder(
        self, clean_path: Path, tmp_path: Path
    ) -> None:
        spec = self._saved(tmp_path / "bake")
        baked = modded_binder(_GAME, spec)
        clean_version = _clean(clean_path).version_for(_GAME)
        clean_path.unlink()

        loaded = modded_binder(_GAME, spec)

        assert isinstance(loaded, ModdedCardBinder)
        assert loaded.version_for(_GAME) == clean_version
        for card in baked.all_cards(_GAME):
            again = loaded.get_by_uuid(card.nocab_uuid)
            assert again is not None
            assert again.raw_content == card.raw_content

    @pytest.mark.parametrize(
        "changes",
        [
            {"seed": 1},
            {"pipeline": (KeyCaseSpec(styles=(KeyCase.KEBAB,)),)},
        ],
    )
    def test_a_different_spec_aborts(
        self, clean_path: Path, tmp_path: Path, changes: dict[str, Any]
    ) -> None:
        directory = tmp_path / "bake"
        modded_binder(_GAME, self._saved(directory))

        with pytest.raises(BinderModsMismatchError):
            modded_binder(_GAME, self._saved(directory, **changes))

    def test_a_save_without_its_stamp_is_baked_again(
        self, clean_path: Path, tmp_path: Path
    ) -> None:
        directory = tmp_path / "bake"
        modded_binder(_GAME, self._saved(directory))
        (directory / "gwent.stamp.json").unlink()

        rebaked = modded_binder(_GAME, self._saved(directory))

        assert (directory / "gwent.stamp.json").exists()
        assert len(list(rebaked.all_cards(_GAME))) == 4

    def test_malformed_stamp_raises(self, clean_path: Path, tmp_path: Path) -> None:
        directory = tmp_path / "bake"
        modded_binder(_GAME, self._saved(directory))
        (directory / "gwent.stamp.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="malformed"):
            modded_binder(_GAME, self._saved(directory))


class TestWritableSaveDirectory:
    def test_none_is_a_no_op(self) -> None:
        require_writable_save_directory(_CAMEL)

    def test_creates_the_directory_and_leaves_it_empty(self, tmp_path: Path) -> None:
        directory = tmp_path / "a" / "b"
        require_writable_save_directory(
            BinderModsSpec(_CAMEL.pipeline, save_directory=directory)
        )

        assert directory.is_dir()
        assert list(directory.iterdir()) == []

    def test_a_file_in_the_way_raises(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("", encoding="utf-8")

        with pytest.raises(OSError):
            require_writable_save_directory(
                BinderModsSpec(_CAMEL.pipeline, save_directory=blocker / "bake")
            )


class TestSpec:
    def test_empty_pipeline_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            BinderModsSpec(())

    def test_deck_spec_raises(self) -> None:
        with pytest.raises(TypeError, match="card-field"):
            BinderModsSpec((CardDropoutSpec(0.1),))  # type: ignore[arg-type]


class TestParse:
    def _parse(self, raw: dict[str, Any]) -> BinderModsSpec:
        return parse_binder_mods(ConfigSection(raw, "config.binder_mods"))

    def test_full_section(self) -> None:
        spec = self._parse(
            {
                "pipeline": [{"kind": "shuffle_keys", "variants": 1}],
                "seed": 4,
                "save_directory": "data/cache/binder_mods/x",
            }
        )

        assert spec == BinderModsSpec(
            (ShuffleKeysSpec(variants=1),), 4, Path("data/cache/binder_mods/x")
        )

    def test_empty_save_directory_means_none(self) -> None:
        spec = self._parse(
            {"pipeline": [{"kind": "shuffle_keys"}], "save_directory": ""}
        )

        assert spec.save_directory is None
        assert spec.seed == 0

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({"pipeline": []}, "empty"),
            ({}, "config.binder_mods.pipeline is required"),
            (
                {"pipeline": [{"kind": "shuffle_keys"}, {"kind": "card_dropout"}]},
                "config.binder_mods.pipeline.1",
            ),
            (
                {"pipeline": [{"kind": "shuffle_keys"}], "extra": 1},
                "unknown config keys",
            ),
        ],
    )
    def test_malformed_section_raises(self, raw: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            self._parse(raw)


class TestCardShelf:
    def test_shelf_hands_out_the_modded_binder(self, clean_path: Path) -> None:
        binder = CardShelf(binder_mods=_CAMEL).card_binder(_GAME)

        assert isinstance(binder, ModdedCardBinder)
        assert binder.version_for(_GAME) == _clean(clean_path).version_for(_GAME)

    def test_shelf_without_mods_hands_out_the_clean_binder(
        self, clean_path: Path
    ) -> None:
        binder = CardShelf().card_binder(_GAME)

        assert not isinstance(binder, ModdedCardBinder)
