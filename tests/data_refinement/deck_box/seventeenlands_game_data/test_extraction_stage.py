"""Tests for extraction_stage.py's SeventeenLandsGameDataDeckExtractionStage:
every distinct decklist played is stored once, with the same ids the
game_data metrics write."""

import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.seventeenlands_game_data.extraction_stage import (
    SeventeenLandsGameDataDeckExtractionStage,
)
from src.data_refinement.deck_ids import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (  # noqa: E501
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import scan_game_csv
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    parser_for,
    row,
    uuid_for,
    write_csv,
)

_UNMATCHED_COLUMN = "deck_Unmatched Card"


def _binder() -> CardBinder:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    binder.ensure_unknown_card(GameId.MTG)
    return binder


def _extract(path: Path, binder: CardBinder, box: DeckBox) -> list:
    return SeventeenLandsGameDataDeckExtractionStage().extract(path, box, binder)


def _deck(binder: CardBinder, owlbears: int, morningstars: int = 0):
    cards = [uuid_for(binder, OWLBEAR)] * owlbears
    cards += [uuid_for(binder, MORNINGSTAR)] * morningstars
    return deck_uuid_from_cards(cards)


def test_copy_counts_are_expanded_into_the_full_decklist(tmp_path: Path) -> None:
    binder, box = _binder(), DeckBox()
    path = write_csv(
        tmp_path / "MSH.PremierDraft.csv",
        [row(won=True, owlbear_deck=3, morningstar_deck=2)],
    )

    (deck_uuid,) = _extract(path, binder, box)

    deck = box.get_by_uuid(deck_uuid)
    assert deck is not None and deck.source_game == GameId.MTG
    assert Counter(deck.card_nocab_uuids) == Counter(
        {uuid_for(binder, OWLBEAR): 3, uuid_for(binder, MORNINGSTAR): 2}
    )
    assert deck_uuid == _deck(binder, 3, 2)


def test_an_unmatched_column_counts_as_the_unknown_card(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binder, box = _binder(), DeckBox()
    game = {**row(won=True, owlbear_deck=2), _UNMATCHED_COLUMN: 3}
    path = write_csv(tmp_path / "MSH.PremierDraft.csv", [game])

    with caplog.at_level(logging.ERROR):
        (deck_uuid,) = _extract(path, binder, box)

    deck = box.get_by_uuid(deck_uuid)
    assert deck is not None
    assert deck.card_nocab_uuids.count(CardBinder.unknown_card_uuid(GameId.MTG)) == 3
    assert len(deck.card_nocab_uuids) == 5
    assert any("Unmatched Card" in record.getMessage() for record in caplog.records)


def test_a_binder_without_the_unknown_card_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "MSH.PremierDraft.csv", [row(won=True)])

    with pytest.raises(RuntimeError, match="ensure_unknown_card"):
        _extract(path, binder_with_cards([OWLBEAR]), DeckBox())


def test_every_distinct_decklist_a_draft_plays_is_stored(tmp_path: Path) -> None:
    # Game 2 was played after sideboarding: a different, full decklist
    binder, box = _binder(), DeckBox()
    path = write_csv(
        tmp_path / "MSH.TradDraft.csv",
        [
            row(won=True, owlbear_deck=4, game_number=1),
            row(won=False, owlbear_deck=3, morningstar_deck=1, game_number=2),
            row(won=True, owlbear_deck=4, game_number=3),
        ],
    )

    deck_uuids = _extract(path, binder, box)

    assert deck_uuids == [_deck(binder, 4), _deck(binder, 3, 1)]
    assert len(list(box.all_decks(GameId.MTG))) == 2


def test_identical_decklists_from_two_drafts_are_stored_once(tmp_path: Path) -> None:
    binder, box = _binder(), DeckBox()
    path = write_csv(
        tmp_path / "MSH.PremierDraft.csv",
        [
            row(won=True, owlbear_deck=4, draft_id="d1"),
            row(won=False, owlbear_deck=4, draft_id="d2"),
        ],
    )

    assert _extract(path, binder, box) == [_deck(binder, 4)]


def test_a_deck_is_named_and_sourced_after_its_first_game(tmp_path: Path) -> None:
    binder, box = _binder(), DeckBox()
    path = write_csv(
        tmp_path / "MSH.PremierDraft.csv",
        [
            row(won=True, owlbear_deck=4, draft_id="d7", match_number=2, game_number=1),
            row(won=True, owlbear_deck=4, draft_id="d8", match_number=1, game_number=1),
        ],
    )

    (deck_uuid,) = _extract(path, binder, box)

    deck = box.get_by_uuid(deck_uuid)
    assert deck is not None and deck.provenance is not None
    assert deck.name == "17lands game_data MSH.PremierDraft d7:2:1 deck"
    assert deck.provenance.data_source == DataSource.SEVENTEENLANDS_GAME_DATA
    assert deck.provenance.source_id == "d7:2:1"


def test_reextracting_is_idempotent(tmp_path: Path) -> None:
    binder, box = _binder(), DeckBox()
    path = write_csv(tmp_path / "MSH.PremierDraft.csv", [row(won=True, owlbear_deck=4)])

    first = _extract(path, binder, box)
    second = _extract(path, binder, box)

    assert first == second
    assert len(list(box.all_decks(GameId.MTG))) == 1


def test_a_directory_reports_a_deck_shared_by_two_files_once(tmp_path: Path) -> None:
    binder, box = _binder(), DeckBox()
    write_csv(tmp_path / "MSH.PremierDraft.csv", [row(won=True, owlbear_deck=4)])
    write_csv(
        tmp_path / "MSH.TradDraft.csv",
        [row(won=True, owlbear_deck=4), row(won=True, owlbear_deck=2)],
    )

    deck_uuids = _extract(tmp_path, binder, box)

    assert deck_uuids == [_deck(binder, 4), _deck(binder, 2)]


def test_a_file_without_match_number_is_read(tmp_path: Path) -> None:
    binder, box = _binder(), DeckBox()
    game = row(won=True, owlbear_deck=4, draft_id="d1", game_number=2)
    del game["match_number"]
    header = [column for column in game]
    path = write_csv(tmp_path / "AFR.PremierDraft.csv", [game], header=header)

    (deck_uuid,) = _extract(path, binder, box)

    deck = box.get_by_uuid(deck_uuid)
    assert deck is not None and deck.provenance is not None
    assert deck.provenance.source_id == "d1:0:2"


def test_every_metric_deck_uuid_is_a_stored_deck(tmp_path: Path) -> None:
    # The round trip the deck dojos rely on: metric rows -> canonical box
    binder, box = _binder(), DeckBox()
    games = [
        row(won=True, owlbear_deck=4, draft_id="d1", game_number=1),
        row(
            won=False, owlbear_deck=3, morningstar_deck=1, draft_id="d1", game_number=2
        ),
        {**row(won=True, owlbear_deck=1, draft_id="d2"), _UNMATCHED_COLUMN: 2},
    ]
    path = write_csv(tmp_path / "MSH.TradDraft.csv", games)
    metric = DeckWinPredictionMetric(VERSION, tmp_path / "deck_win.parquet")
    scan_game_csv(path, [metric], parser_for(binder))

    stored = set(_extract(path, binder, box))
    metric_uuids = set(pd.read_parquet(metric.finalize())["deck_uuid"])

    assert {str(deck_uuid) for deck_uuid in stored} == metric_uuids
    assert all(box.get_by_uuid(deck_uuid) is not None for deck_uuid in stored)
