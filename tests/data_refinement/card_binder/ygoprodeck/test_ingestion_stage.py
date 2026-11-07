from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.ygoprodeck.ingestion_stage import (
    CardNameIndex,
    YgoprodeckPasscodeAliasStage,
)
from src.data_refinement.card_binder.ygoprodeck.passcode import passcode_id
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_YGO = GameId.YUGIOH


def _add_card(binder: CardBinder, name: str) -> UUID:
    card = GenericCard(
        nocab_uuid=uuid4(),
        source_game=_YGO,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            data_source=DataSource.TCGJSON,
            source_id="1",
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    binder.create(card)
    return card.nocab_uuid


def _write_mirror(
    raw_dir: Path, cards: list[tuple[str, str]], variants: list[tuple[str, str]] = []
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cards.csv").write_text(
        "id,name,type\n" + "".join(f'{i},"{n}",Effect Monster\n' for i, n in cards),
        encoding="utf-8",
    )
    (raw_dir / "cards_variants.csv").write_text(
        "original,variant\n" + "".join(f"{o},{v}\n" for o, v in variants),
        encoding="utf-8",
    )
    (raw_dir / "revision.json").write_text("{}", encoding="utf-8")
    return raw_dir


def _alias(binder: CardBinder, passcode: str) -> UUID | None:
    card = binder.get_by_alias(_YGO, DataSource.YGOPRODECK, passcode)
    return card.nocab_uuid if card else None


class TestIngest:
    def test_aliases_each_passcode_to_its_named_card(self, tmp_path: Path) -> None:
        binder = CardBinder()
        magician = _add_card(binder, "Dark Magician")
        raw = _write_mirror(tmp_path, [("46986414", "Dark Magician")])

        result = YgoprodeckPasscodeAliasStage().ingest(raw, binder)

        assert result == []
        assert _alias(binder, "46986414") == magician

    def test_a_casefold_match_counts_but_an_ambiguous_name_does_not(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        fusion = _add_card(binder, "Polymerization")
        _add_card(binder, "Twin Card")
        _add_card(binder, "twin card")
        raw = _write_mirror(
            tmp_path,
            [("24094653", "POLYMERIZATION"), ("1", "TWIN CARD"), ("2", "Not In TCG")],
        )

        YgoprodeckPasscodeAliasStage().ingest(raw, binder)

        assert _alias(binder, "24094653") == fusion
        assert _alias(binder, "1") is None
        assert _alias(binder, "2") is None

    def test_an_exact_name_wins_over_casefold_matches(self, tmp_path: Path) -> None:
        binder = CardBinder()
        exact = _add_card(binder, "Twin Card")
        _add_card(binder, "twin card")
        raw = _write_mirror(tmp_path, [("1", "Twin Card")])

        YgoprodeckPasscodeAliasStage().ingest(raw, binder)

        assert _alias(binder, "1") == exact

    def test_variants_alias_their_original_card(self, tmp_path: Path) -> None:
        binder = CardBinder()
        magician = _add_card(binder, "Dark Magician")
        raw = _write_mirror(
            tmp_path,
            [("46986414", "Dark Magician")],
            [("46986414", "36996508"), ("999", "1000")],
        )

        YgoprodeckPasscodeAliasStage().ingest(raw, binder)

        assert _alias(binder, "36996508") == magician
        assert _alias(binder, "1000") is None

    def test_leading_zeros_are_one_alias(self, tmp_path: Path) -> None:
        binder = CardBinder()
        card = _add_card(binder, "Mystical Space Typhoon")
        raw = _write_mirror(tmp_path, [("05318639", "Mystical Space Typhoon")])

        YgoprodeckPasscodeAliasStage().ingest(raw, binder)

        assert _alias(binder, passcode_id("5318639")) == card

    def test_a_rerun_repoints_a_passcode_at_the_current_card(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        old = _add_card(binder, "Dark Magician")
        raw = _write_mirror(tmp_path, [("46986414", "Dark Magician")])
        YgoprodeckPasscodeAliasStage().ingest(raw, binder)
        # tcgjson re-minted the card under a new uuid
        binder.delete(old)
        new = _add_card(binder, "Dark Magician")

        YgoprodeckPasscodeAliasStage().ingest(raw, binder)

        assert _alias(binder, "46986414") == new

    def test_the_unknown_sentinel_is_never_a_candidate(self, tmp_path: Path) -> None:
        binder = CardBinder()
        _add_card(binder, "Dark Magician")
        binder.ensure_unknown_card(_YGO)
        raw = _write_mirror(tmp_path, [("7", "Unknown")])

        YgoprodeckPasscodeAliasStage().ingest(raw, binder)

        assert _alias(binder, "7") is None

    def test_raises_on_a_binder_without_yugioh_cards(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.ensure_unknown_card(_YGO)
        raw = _write_mirror(tmp_path, [("46986414", "Dark Magician")])

        with pytest.raises(ValueError, match="tcgjson_yugioh"):
            YgoprodeckPasscodeAliasStage().ingest(raw, binder)

    def test_raises_on_an_incomplete_mirror(self, tmp_path: Path) -> None:
        binder = CardBinder()
        _add_card(binder, "Dark Magician")
        raw = _write_mirror(tmp_path, [("46986414", "Dark Magician")])
        (raw / "revision.json").unlink()

        with pytest.raises(FileNotFoundError):
            YgoprodeckPasscodeAliasStage().ingest(raw, binder)

    def test_raises_on_a_non_passcode_id(self, tmp_path: Path) -> None:
        binder = CardBinder()
        _add_card(binder, "Dark Magician")
        raw = _write_mirror(tmp_path, [("abc", "Dark Magician")])

        with pytest.raises(ValueError, match="passcode"):
            YgoprodeckPasscodeAliasStage().ingest(raw, binder)


class TestCardNameIndex:
    def test_symbols_tcgplayer_drops_still_match(self) -> None:
        binder = CardBinder()
        cat = _add_card(binder, "Maliss P Chessy Cat")
        cupsy = _add_card(binder, "CupsyYummy")
        doll = _add_card(binder, "Doll Monster Miss Madchen")
        index = CardNameIndex(binder)

        assert index.card_uuid("Maliss <P> Chessy Cat") == cat
        assert index.card_uuid("Cupsy☆Yummy") == cupsy
        assert index.card_uuid("Doll Monster Miss Mädchen") == doll

    def test_a_looser_key_never_breaks_a_stricter_tie(self) -> None:
        binder = CardBinder()
        _add_card(binder, "Hundred-Eyes Dragon")
        _add_card(binder, "Hundred Eyes Dragon")
        index = CardNameIndex(binder)

        assert index.card_uuid("Hundred Eyes Dragon") is not None
        assert index.card_uuid("Hundred=Eyes Dragon") is None

    def test_a_name_of_symbols_only_matches_nothing(self) -> None:
        binder = CardBinder()
        _add_card(binder, "Dark Magician")
        _add_card(binder, "!!!")
        index = CardNameIndex(binder)
        assert index.card_uuid("???") is None
        assert index.card_uuid("!!!") is not None

    def test_counts_cards_without_the_sentinel(self) -> None:
        binder = CardBinder()
        _add_card(binder, "Dark Magician")
        binder.ensure_unknown_card(_YGO)
        assert CardNameIndex(binder).card_count == 1


class TestPasscodeId:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("08512558", "8512558"), (" 46986414 ", "46986414"), ("0", "0")],
    )
    def test_spells_digits_without_leading_zeros(self, raw: str, expected: str) -> None:
        assert passcode_id(raw) == expected

    @pytest.mark.parametrize("raw", ["", "12a", "-5", "1.0", "١٢"])
    def test_rejects_non_passcodes(self, raw: str) -> None:
        with pytest.raises(ValueError):
            passcode_id(raw)
