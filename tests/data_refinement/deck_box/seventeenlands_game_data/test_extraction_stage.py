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
from src.data_refinement.deck_box.seventeenlands_game_data import (
    extraction_stage as extraction_stage_module,
)
from src.data_refinement.deck_box.seventeenlands_game_data.extraction_stage import (
    SeventeenLandsGameDataDeckExtractionStage,
    _GameKey,
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
    build_index: int = 0,
) -> dict:
    return {
        "expansion": expansion,
        "event_type": event_type,
        "draft_id": draft_id,
        "build_index": build_index,
        "match_number": match_number,
        "game_number": game_number,
        "deck_counts": deck_counts,
    }


def _game_data_csv_bytes(
    deck_card_names: list[str], rows: list[dict], with_match_number: bool = True
) -> bytes:
    # with_match_number=False mirrors the 10 older 17Lands files, which
    # have no match_number column at all.
    key_columns = ["expansion", "event_type", "draft_id", "build_index"]
    if with_match_number:
        key_columns.append("match_number")
    key_columns.append("game_number")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(key_columns + [f"deck_{name}" for name in deck_card_names])
    for row in rows:
        deck_counts = row["deck_counts"]
        writer.writerow(
            [row[column] for column in key_columns]
            + [deck_counts.get(name, 0) for name in deck_card_names]
        )
    return buffer.getvalue().encode("utf-8")


def _write_game_data_csv(
    path: Path,
    deck_card_names: list[str],
    rows: list[dict],
    with_match_number: bool = True,
) -> None:
    path.write_bytes(_game_data_csv_bytes(deck_card_names, rows, with_match_number))


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
        # presence-sampling game_data ZoneCounts.present(), which this
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
        # The second run's create_if_absent() call recurs (no-op)
        # against the same content-addressed deck_uuid - still
        # reported, per extract()'s changed_uuids contract (see module
        # docstring).
        assert second == first
        assert len(list(box.all_decks(GameId.MTG))) == 1

    def test_reextracting_after_resolution_changes_creates_a_new_content_addressed_deck(
        self, tmp_path: Path
    ) -> None:
        # Content-addressed identity means resolving a previously-
        # unresolvable card changes the draft's card multiset, which
        # changes its deck_uuid - this is NOT an in-place update of the
        # old entry (there is no stable per-draft id to update anymore
        # - see module docstring's ONE DECK PER DRAFT, CONTENT-ADDRESSED
        # section). The box ends up holding BOTH the old, stale
        # (Unknown-sentinel) deck and the new, correctly-resolved one;
        # nothing here collapses or deletes the stale one, and that's
        # intentional under this identity scheme.
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
        first_deck_uuid = first[0]
        deck_after_first = box.get_by_uuid(first_deck_uuid)
        assert unknown.nocab_uuid in deck_after_first.card_nocab_uuids

        # MTG's own CardBinder gains the previously-unresolvable card.
        bolt_card = _mtg_card("Lightning Bolt")
        binder.create(bolt_card)

        second = stage.extract(raw_path, box, binder)

        assert second != first
        assert len(second) == 1
        second_deck_uuid = second[0]
        assert len(list(box.all_decks(GameId.MTG))) == 2
        deck_after_second = box.get_by_uuid(second_deck_uuid)
        assert bolt_card.nocab_uuid in deck_after_second.card_nocab_uuids
        assert unknown.nocab_uuid not in deck_after_second.card_nocab_uuids
        # The stale first deck is untouched.
        stale_deck = box.get_by_uuid(first_deck_uuid)
        assert unknown.nocab_uuid in stale_deck.card_nocab_uuids

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

    def test_every_game_of_one_draft_shares_one_deck(self, tmp_path: Path) -> None:
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

        assert len(changed_uuids) == 1
        assert len(list(box.all_decks(GameId.MTG))) == 1

    def test_distinct_drafts_with_distinct_decklists_produce_distinct_decks(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Mountain"],
            [_row("d1", 1, 1, {"Plains": 17}), _row("d2", 1, 1, {"Mountain": 17})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        assert len(set(changed_uuids)) == 2
        assert len(list(box.all_decks(GameId.MTG))) == 2

    def test_distinct_drafts_with_identical_decklists_collapse_to_one_deck(
        self, tmp_path: Path
    ) -> None:
        # The whole point of the content-addressed identity change
        # (see module docstring's ONE DECK PER DRAFT, CONTENT-ADDRESSED
        # section): two different drafts ("d1", "d2") that happen to
        # build the exact same 40-card deck must collapse into ONE
        # stored GenericDeck.
        binder = _mtg_card_binder(["Plains"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains"],
            [_row("d1", 1, 1, {"Plains": 17}), _row("d2", 1, 1, {"Plains": 17})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        assert len(set(changed_uuids)) == 1
        assert len(list(box.all_decks(GameId.MTG))) == 1

    @pytest.mark.parametrize(
        "game_keys",
        [[(1, 1), (1, 2), (2, 1)], [(2, 1), (1, 2), (1, 1)], [(1, 2), (2, 1), (1, 1)]],
    )
    def test_keeps_the_lowest_match_and_game_whatever_the_row_order(
        self, tmp_path: Path, game_keys: list[tuple[int, int]]
    ) -> None:
        # Each game carries a different Mountain count, so the stored
        # deck shows which game won.
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        rows = [
            _row("d1", match, game, {"Plains": 16, "Mountain": 10 * match + game})
            for match, game in game_keys
        ]
        _write_game_data_csv(raw_path, ["Plains", "Mountain"], rows)
        stage = SeventeenLandsGameDataDeckExtractionStage()

        stage.extract(raw_path, box, binder)

        (deck,) = list(box.all_decks(GameId.MTG))
        mountain_uuid = binder.get_by_name_single(GameId.MTG, "Mountain").nocab_uuid
        assert deck.card_nocab_uuids.count(mountain_uuid) == 11
        assert deck.provenance is not None
        assert deck.provenance.source_id == "d1:0:1:1"

    def test_reextracting_a_multi_game_draft_is_idempotent(
        self, tmp_path: Path
    ) -> None:
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Mountain"],
            [
                _row("d1", 1, 2, {"Plains": 17}),
                _row("d1", 1, 1, {"Plains": 16, "Mountain": 1}),
            ],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        second = stage.extract(raw_path, box, binder)

        assert second == first
        assert len(list(box.all_decks(GameId.MTG))) == 1

    def test_out_of_order_games_report_their_draft_once(self, tmp_path: Path) -> None:
        binder = _mtg_card_binder(["Plains"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains"],
            [_row("d1", 2, 1, {"Plains": 17}), _row("d1", 1, 1, {"Plains": 17})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        assert len(changed_uuids) == 1

    def test_same_draft_id_across_files_produces_independent_decks(
        self, tmp_path: Path
    ) -> None:
        # Accumulation is per-file (see module docstring's STREAMING
        # and TWO-PHASE EXTRACTION sections) - this stage never merges
        # one draft_id's rows across two files. 17Lands' own convention
        # keeps a draft_id's rows in one file, but if that assumption
        # is ever violated, each file's own canonical game for "d1"
        # is hashed and stored independently, producing two decks
        # rather than one being replaced by the other.
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        later_path = tmp_path / "later.csv"
        earlier_path = tmp_path / "earlier.csv"
        _write_game_data_csv(
            later_path, ["Plains", "Mountain"], [_row("d1", 2, 1, {"Plains": 17})]
        )
        _write_game_data_csv(
            earlier_path,
            ["Plains", "Mountain"],
            [_row("d1", 1, 1, {"Plains": 16, "Mountain": 1})],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        first = stage.extract(later_path, box, binder)
        second = stage.extract(earlier_path, box, binder)

        assert first != second
        assert len(list(box.all_decks(GameId.MTG))) == 2
        earlier_deck = box.get_by_uuid(second[0])
        assert earlier_deck is not None
        assert earlier_deck.provenance is not None
        assert earlier_deck.provenance.source_id == "d1:0:1:1"

    def test_a_file_without_match_number_keeps_the_first_build(
        self, tmp_path: Path
    ) -> None:
        # Older files number games per match and have no match_number
        # column, so a draft repeats game_number 1; build_index picks
        # the deck as first built.
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        raw_path = tmp_path / "STX.TradSealed.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Mountain"],
            [
                _row("d1", 0, 1, {"Plains": 15, "Mountain": 2}, build_index=1),
                _row("d1", 0, 1, {"Plains": 16, "Mountain": 1}, build_index=0),
                _row("d1", 0, 2, {"Plains": 14, "Mountain": 3}, build_index=2),
                _row("d1", 0, 1, {"Plains": 16, "Mountain": 1}, build_index=0),
            ],
            with_match_number=False,
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        second = stage.extract(raw_path, box, binder)

        (deck,) = list(box.all_decks(GameId.MTG))
        mountain_uuid = binder.get_by_name_single(GameId.MTG, "Mountain").nocab_uuid
        assert deck.card_nocab_uuids.count(mountain_uuid) == 1
        assert deck.provenance is not None
        assert deck.provenance.source_id == "d1:0:0:1"
        assert second == first

    def test_canonical_game_wins_even_when_split_across_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Forces each row into its own pandas chunk (see
        # extraction_stage.py's _CHUNK_SIZE), so the draft's three rows
        # are accumulated across three separate _accumulate_best_games()
        # calls spanning three chunk iterations - the lowest _GameKey
        # must still win regardless, exercising module docstring's
        # TWO-PHASE EXTRACTION accumulation across chunk boundaries, not
        # just across out-of-order rows within one chunk.
        monkeypatch.setattr(extraction_stage_module, "_CHUNK_SIZE", 1)
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Mountain"],
            [
                _row("d1", 2, 1, {"Plains": 16, "Mountain": 3}),
                _row("d1", 1, 2, {"Plains": 16, "Mountain": 2}),
                _row("d1", 1, 1, {"Plains": 16, "Mountain": 1}),
            ],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        stage.extract(raw_path, box, binder)

        (deck,) = list(box.all_decks(GameId.MTG))
        mountain_uuid = binder.get_by_name_single(GameId.MTG, "Mountain").nocab_uuid
        assert deck.card_nocab_uuids.count(mountain_uuid) == 1
        assert deck.provenance is not None
        assert deck.provenance.source_id == "d1:0:1:1"

    def test_reprocessing_the_same_file_twice_is_crash_retry_safe(
        self, tmp_path: Path
    ) -> None:
        # Simulates a crash-then-retry: re-running extract() against
        # the exact same file must not raise and must converge to the
        # exact same final box content (see module docstring's
        # TWO-PHASE EXTRACTION section's own crash-retry-safety claim).
        binder = _mtg_card_binder(["Plains", "Mountain"])
        box = DeckBox()
        raw_path = tmp_path / "MSH.PremierDraft.csv"
        _write_game_data_csv(
            raw_path,
            ["Plains", "Mountain"],
            [
                _row("d1", 1, 1, {"Plains": 17}),
                _row("d2", 1, 1, {"Plains": 16, "Mountain": 1}),
            ],
        )
        stage = SeventeenLandsGameDataDeckExtractionStage()

        stage.extract(raw_path, box, binder)
        decks_after_first = {
            deck.nocab_uuid: deck for deck in box.all_decks(GameId.MTG)
        }

        stage.extract(raw_path, box, binder)
        decks_after_second = {
            deck.nocab_uuid: deck for deck in box.all_decks(GameId.MTG)
        }

        assert decks_after_first.keys() == decks_after_second.keys()
        assert len(decks_after_second) == 2
        for nocab_uuid, deck in decks_after_first.items():
            assert Counter(deck.card_nocab_uuids) == Counter(
                decks_after_second[nocab_uuid].card_nocab_uuids
            )


class TestGameKey:
    def test_float_numbers_round_trip_as_integers(self) -> None:
        # pandas reads the columns as float when a chunk holds a NaN.
        game_key = _GameKey.from_row(
            {"build_index": 0.0, "match_number": 1.0, "game_number": 2.0}
        )
        provenance = Provenance(
            data_source=DataSource.SEVENTEENLANDS_GAME_DATA,
            source_id=game_key.source_id_for("d1"),
            fetched_at=datetime.now(timezone.utc),
        )

        assert provenance.source_id == "d1:0:1:2"
        assert _GameKey.from_provenance(provenance) == _GameKey(0, 1, 2)

    def test_a_row_without_match_number_counts_it_as_zero(self) -> None:
        assert _GameKey.from_row({"build_index": 2, "game_number": 1}) == _GameKey(
            2, 0, 1
        )

    def test_build_index_outranks_match_and_game(self) -> None:
        assert _GameKey(0, 3, 3) < _GameKey(1, 1, 1)

    @pytest.mark.parametrize(
        "source_id", ["deck-1", "d1:1:1", "d1:0:x:1", "d1:0.0:1.0:2.0"]
    )
    def test_a_source_id_this_stage_did_not_write_parses_as_none(
        self, source_id: str
    ) -> None:
        provenance = Provenance(
            data_source=DataSource.SEVENTEENLANDS_GAME_DATA,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        )

        assert _GameKey.from_provenance(provenance) is None

    def test_no_provenance_parses_as_none(self) -> None:
        assert _GameKey.from_provenance(None) is None

    def test_a_nan_game_number_raises(self) -> None:
        with pytest.raises(ValueError):
            _GameKey.from_row(
                {"build_index": 0, "match_number": 1, "game_number": float("nan")}
            )

    def test_a_missing_build_index_raises(self) -> None:
        with pytest.raises(KeyError):
            _GameKey.from_row({"match_number": 1, "game_number": 1})

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
        assert deck.provenance.source_id == "d1:0:1:1"
