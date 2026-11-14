import logging
from collections import Counter
from pathlib import Path
from typing import ClassVar

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    DeckSkip,
    DeckSkipReason,
    MtgCardRef,
    MtgDeckEntry,
)
from src.data_refinement.deck_box.mtg_crawls.extraction_stage import (
    CrawledMtgDeckExtractionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.json_types import JsonObject
from tests.data_refinement.deck_box.mtg_crawls.fixtures import (
    card_uuid,
    make_mtg_binder,
    write_shards,
)

_BOLT = MtgCardRef("Lightning Bolt", "o-bolt", None)
_MISSING = MtgCardRef("Not A Card", None, None)


class _RowParser:
    """Reads test rows {"id", "name", "skip"?, "cards": [[count, name], ...]}."""

    def parse(self, row: JsonObject) -> CrawledDeck | DeckSkip:
        if row.get("skip"):
            return DeckSkip(DeckSkipReason.NOT_A_DECK)
        cards = {"Lightning Bolt": _BOLT, "Not A Card": _MISSING}
        entries = tuple(
            MtgDeckEntry(count, cards[name]) for count, name in row["cards"]  # type: ignore[misc]
        )
        return CrawledDeck(str(row["id"]), str(row["name"]), entries)


class _TestStage(CrawledMtgDeckExtractionStage):
    DATA_SOURCE: ClassVar[DataSource] = DataSource.MOXFIELD
    DEFAULT_RAW_PATH: ClassVar[Path] = Path("does/not/exist")

    def __init__(self) -> None:
        super().__init__(_RowParser())


def _row(deck_id: str, bolts: int, name: str = "Burn", **extra: object) -> dict:
    return {"id": deck_id, "name": name, "cards": [[bolts, "Lightning Bolt"]], **extra}


class TestExtract:
    def test_stores_full_distinct_decks_and_counts_the_rest(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = make_mtg_binder([("Lightning Bolt", "o-bolt", ())])
        shards = write_shards(
            tmp_path,
            [
                _row("1", 60),
                _row("2", 60, name="Same List"),  # identical multiset
                _row("3", 19),  # too small
                _row("4", 257),  # too large
                _row("5", 60, skip=True),
                {
                    "id": "6",
                    "name": "",
                    "cards": [[58, "Lightning Bolt"], [2, "Not A Card"]],
                },
            ],
        )
        box = DeckBox()

        with caplog.at_level(logging.INFO):
            created = _TestStage().extract(shards, box, binder)

        assert len(created) == 2
        first, second = (box.get_by_uuid(uuid) for uuid in created)
        assert first is not None and first.name == "Burn"
        assert first.provenance is not None
        assert first.provenance.data_source is DataSource.MOXFIELD
        assert first.provenance.source_id == "1"
        assert second is not None and second.name == "6"  # no name: its id
        assert Counter(second.card_nocab_uuids) == {
            card_uuid("o-bolt"): 58,
            CardBinder.unknown_card_uuid(GameId.MTG): 2,
        }
        assert "'created': 2" in caplog.text and "'wrong_size': 2" in caplog.text
        assert "'not_a_deck': 1" in caplog.text
        assert "('Not A Card', 1)" in caplog.text  # per entry, not per copy

    def test_a_rerun_creates_nothing(self, tmp_path: Path) -> None:
        binder = make_mtg_binder([("Lightning Bolt", "o-bolt", ())])
        shards = write_shards(tmp_path, [_row("1", 60)])
        box = DeckBox()
        _TestStage().extract(shards, box, binder)

        assert _TestStage().extract(shards, box, binder) == []

    def test_missing_shards_raise(self) -> None:
        with pytest.raises(FileNotFoundError):
            _TestStage().extract(None, DeckBox(), make_mtg_binder([]))

    def test_a_missing_unknown_sentinel_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Unknown sentinel"):
            _TestStage().extract(write_shards(tmp_path, []), DeckBox(), CardBinder())


class TestMtgDeckEntry:
    def test_an_entry_holds_at_least_one_copy(self) -> None:
        with pytest.raises(ValueError):
            MtgDeckEntry(0, _BOLT)
