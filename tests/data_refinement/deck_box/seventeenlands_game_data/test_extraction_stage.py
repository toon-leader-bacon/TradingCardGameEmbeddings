import csv
import io
import logging
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.seventeenlands_game_data.extraction_stage import (
    SeventeenLandsGameDataDeckExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _mtg_card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _mtg_card_binder(names: list[str]) -> CardBinder:
    binder = CardBinder()
    for name in names:
        binder.create(_mtg_card(name))
    return binder


def _row(
    draft_id: str,
    match_number: int,
    game_number: int,
    deck_counts: dict[str, int],
    expansion: str = "MSH",
    event_type: str = "PremierDraft",
) -> dict:
    return {
        "expansion": expansion,
        "event_type": event_type,
        "draft_id": draft_id,
        "match_number": match_number,
        "game_number": game_number,
        "deck_counts": deck_counts,
    }


def _game_data_csv_bytes(deck_card_names: list[str], rows: list[dict]) -> bytes:
    fieldnames = [
        "expansion",
        "event_type",
        "draft_id",
        "match_number",
        "game_number",
    ] + [f"deck_{name}" for name in deck_card_names]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(fieldnames)
    for row in rows:
        deck_counts = row["deck_counts"]
        writer.writerow(
            [
                row["expansion"],
                row["event_type"],
                row["draft_id"],
                row["match_number"],
                row["game_number"],
            ]
            + [deck_counts.get(name, 0) for name in deck_card_names]
        )
    return buffer.getvalue().encode("utf-8")


def _write_game_data_csv(
    path: Path, deck_card_names: list[str], rows: list[dict]
) -> None:
    path.write_bytes(_game_data_csv_bytes(deck_card_names, rows))


def _write_tar_wrapped_game_data_csv(
    path: Path, member_name: str, deck_card_names: list[str], rows: list[dict]
) -> None:
    # Reproduces the shape CONFIRMED on disk for 10 of 17Lands' 133
    # game_data CSVs (AFR/KHM/MID/STX/VOW) — a raw (already
    # decompressed) POSIX tar archive wrapping one CSV member, not a
    # plain CSV. No gzip layer here (unlike this session's
    # tests/data_retrieval/seventeenlands/test_downloader.py's
    # _tar_wrapped_gzip() helper) since that's exactly the on-disk
    # quirk this stage's _open_stream() reads through.
    content = _game_data_csv_bytes(deck_card_names, rows)
    with tarfile.open(path, mode="w") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))


class TestExtract:
    def test_all_resolvable_cards_creates_deck(self, tmp_path: Path) -> None:
        binder = _mtg_card_binder(["Plains", "Lightning Bolt"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Lightning Bolt"],
            [_row("d1", 1, 1, {"Plains": 17, "Lightning Bolt": 4})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        assert len(changed_uuids) == 1
        deck = box.get_by_uuid(changed_uuids[0])
        assert deck is not None
        assert deck.source_game == GameId.MTG
        plains_uuid = binder.get_by_name_single(GameId.MTG, "Plains").nocab_uuid
        bolt_uuid = binder.get_by_name_single(GameId.MTG, "Lightning Bolt").nocab_uuid
        assert Counter(deck.card_nocab_uuids) == Counter(
            {plains_uuid: 17, bolt_uuid: 4}
        )

    def test_copy_count_is_expanded_not_sampled_as_presence(
        self, tmp_path: Path
    ) -> None:
        # deck_<name> is a COUNT, not a presence flag — a row value of 3
        # must contribute exactly 3 copies of that card's uuid, never
        # just 1 (contrast metrics/seventeenlands/game_data's own
        # presence-sampling GameCardColumns.present_uuids(), which this
        # stage must not resemble).
        binder = _mtg_card_binder(["Mountain"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path, ["Mountain"], [_row("d1", 1, 1, {"Mountain": 3})]
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        deck = box.get_by_uuid(changed_uuids[0])
        mountain_uuid = binder.get_by_name_single(GameId.MTG, "Mountain").nocab_uuid
        assert deck.card_nocab_uuids.count(mountain_uuid) == 3
        assert len(deck.card_nocab_uuids) == 3

    def test_unresolvable_card_falls_back_to_unknown_sentinel(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _mtg_card_binder(["Plains"])
        unknown = binder.ensure_unknown_card(GameId.MTG)
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Missing Card"],
            [_row("d1", 1, 1, {"Plains": 16, "Missing Card": 1})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        with caplog.at_level(logging.ERROR):
            changed_uuids = stage.extract(raw_path, box, binder)

        deck = box.get_by_uuid(changed_uuids[0])
        assert deck.card_nocab_uuids.count(unknown.nocab_uuid) == 1
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_reextracting_same_raw_path_is_idempotent(self, tmp_path: Path) -> None:
        binder = _mtg_card_binder(["Plains"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(raw_path, ["Plains"], [_row("d1", 1, 1, {"Plains": 17})])
        stage = SeventeenLandsGameDataDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        second = stage.extract(raw_path, box, binder)

        assert len(first) == 1
        assert second == []
        assert len(list(box.all_decks(GameId.MTG))) == 1

    def test_reextracting_after_resolution_changes_updates_not_duplicates(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains"])
        unknown = binder.ensure_unknown_card(GameId.MTG)
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Lightning Bolt"],
            [_row("d1", 1, 1, {"Plains": 16, "Lightning Bolt": 1})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        assert len(first) == 1
        deck_uuid = first[0]
        deck_after_first = box.get_by_uuid(deck_uuid)
        assert unknown.nocab_uuid in deck_after_first.card_nocab_uuids

        # MTG's own CardBinder gains the previously-unresolvable card.
        bolt_card = _mtg_card("Lightning Bolt")
        binder.create(bolt_card)

        second = stage.extract(raw_path, box, binder)

        assert second == [deck_uuid]
        assert len(list(box.all_decks(GameId.MTG))) == 1
        deck_after_second = box.get_by_uuid(deck_uuid)
        assert bolt_card.nocab_uuid in deck_after_second.card_nocab_uuids
        assert unknown.nocab_uuid not in deck_after_second.card_nocab_uuids

    def test_raises_runtime_error_when_unknown_sentinel_not_seeded(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Missing Card"],
            [_row("d1", 1, 1, {"Plains": 16, "Missing Card": 1})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        with pytest.raises(RuntimeError):
            stage.extract(raw_path, box, binder)

    def test_directory_of_multiple_csv_files_all_processed(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        raw_dir = tmp_path / "game_data"
        raw_dir.mkdir()
        _write_game_data_csv(
            raw_dir / "MSH.PremierDraft.csv",
            ["Plains"],
            [_row("d1", 1, 1, {"Plains": 17}, expansion="MSH")],
        )
        _write_game_data_csv(
            raw_dir / "WOE.PremierDraft.csv",
            ["Mountain"],
            [_row("d2", 1, 1, {"Mountain": 17}, expansion="WOE")],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_dir, box, binder)

        assert len(changed_uuids) == 2
        assert len(set(changed_uuids)) == 2
        assert len(list(box.all_decks(GameId.MTG))) == 2

    def test_tar_wrapped_csv_produces_same_deck_as_plain_csv(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains", "Lightning Bolt"])
        deck_card_names = ["Plains", "Lightning Bolt"]
        rows = [_row("d1", 1, 1, {"Plains": 16, "Lightning Bolt": 4}, expansion="AFR")]

        plain_box = DeckBox()
        plain_path = tmp_path / "plain.csv"
        _write_game_data_csv(plain_path, deck_card_names, rows)
        plain_stage = SeventeenLandsGameDataDeckExtractionStage()
        plain_changed = plain_stage.extract(plain_path, plain_box, binder)
        plain_deck = plain_box.get_by_uuid(plain_changed[0])

        tar_box = DeckBox()
        tar_path = tmp_path / "AFR.PremierDraft.csv"
        _write_tar_wrapped_game_data_csv(
            tar_path, "game_data_public.AFR.PremierDraft.csv", deck_card_names, rows
        )
        tar_stage = SeventeenLandsGameDataDeckExtractionStage()
        tar_changed = tar_stage.extract(tar_path, tar_box, binder)
        tar_deck = tar_box.get_by_uuid(tar_changed[0])

        assert tar_changed == plain_changed
        assert Counter(tar_deck.card_nocab_uuids) == Counter(
            plain_deck.card_nocab_uuids
        )

    def test_split_mdfc_front_face_column_resolves_via_regex(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _mtg_card_binder(["Bala Ged Recovery // Bala Ged Sanctuary"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        # 17Lands' own header names a two-faced card by its front face
        # alone, while CardBinder stores it under the combined name
        # (matching Scryfall's DFC convention) — same front-face
        # fallback GameCardColumns._match_uuid() already established
        # for this raw source (see module docstring's CARD RESOLUTION
        # section), re-implemented here.
        _write_game_data_csv(
            raw_path,
            ["Bala Ged Recovery"],
            [_row("d1", 1, 1, {"Bala Ged Recovery": 2})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        with caplog.at_level(logging.ERROR):
            changed_uuids = stage.extract(raw_path, box, binder)

        deck = box.get_by_uuid(changed_uuids[0])
        front_face_uuid = binder.get_by_name_single(
            GameId.MTG, "Bala Ged Recovery // Bala Ged Sanctuary"
        ).nocab_uuid
        assert deck.card_nocab_uuids.count(front_face_uuid) == 2
        assert not any(record.levelno == logging.ERROR for record in caplog.records)

    def test_distinct_composite_keys_produce_distinct_decks(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains"],
            [
                _row("d1", 1, 1, {"Plains": 17}),
                _row("d1", 1, 2, {"Plains": 17}),
                _row("d1", 2, 1, {"Plains": 17}),
            ],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        assert len(changed_uuids) == 3
        assert len(set(changed_uuids)) == 3
        assert len(list(box.all_decks(GameId.MTG))) == 3

    def test_created_deck_carries_seventeenlands_game_data_provenance(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(raw_path, ["Plains"], [_row("d1", 1, 1, {"Plains": 17})])
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        deck = box.get_by_uuid(changed_uuids[0])
        assert deck.provenance is not None
        assert deck.provenance.data_source == DataSource.SEVENTEENLANDS_GAME_DATA
        assert deck.provenance.source_id == "d1:1:1"
