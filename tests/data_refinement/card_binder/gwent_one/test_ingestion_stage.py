from pathlib import Path

import pytest

from src.data_refinement.card_binder.gwent_one.ingestion_stage import (
    GwentOneCardIngestionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# A normal card: category present, single-line ability text with one
# keyword span.
_MASK_OF_UROBOROS_HTML = """
<div class="card-wrap card-data" data-id="202512" data-power="0"
     data-armor="0" data-provision="0" data-faction="skellige"
     data-set="merchants of ofir" data-color="gold" data-type="stratagem"
     data-rarity="legendary">
    <div class="card-head">
        <div class="card-name"><a href="https://gwent.one/en/card/202512">Mask of Uroboros</a></div>
        <div class="card-category">Location</div>
    </div>
    <div class="card-body">
        <div class="card-body-ability"><span class="keyword order">Order</span>: Draw a card, then <span class="keyword discard">Discard</span> a card and <span class="keyword spawn">Spawn</span> 2 Crows on your <span class="keyword melee">Melee</span> row.<br /></div>
    </div>
</div>
"""

# A card with an empty category and a multi-clause ability (several
# keyword spans plus multiple <br> separators) — the werewolf example
# from this project's own design discussion.
_WEREWOLF_HTML = """
<div class="card-wrap card-data" data-id="201600" data-power="5"
     data-armor="0" data-provision="8" data-faction="monster"
     data-set="base" data-color="bronze" data-type="unit"
     data-rarity="common">
    <div class="card-head">
        <div class="card-name"><a href="https://gwent.one/en/card/201600">Werewolf</a></div>
        <div class="card-category">&nbsp;</div>
    </div>
    <div class="card-body">
        <div class="card-body-ability"><span class="keyword resilient">Resilience</span>.<br><span class="keyword deploy">Deploy</span>: Spawn Imperial Fleet to the left of self and Imperial Marine to the right of self.<br><span class="keyword order">Order</span>: Boost 2 allied units by 2 and swap their positions.<br></div>
    </div>
</div>
"""

# A card with no ability div at all (e.g. a vanilla unit).
_VANILLA_UNIT_HTML = """
<div class="card-wrap card-data" data-id="200055" data-power="3"
     data-armor="0" data-provision="4" data-faction="neutral"
     data-set="base" data-color="bronze" data-type="unit"
     data-rarity="common">
    <div class="card-head">
        <div class="card-name"><a href="https://gwent.one/en/card/200055">Nilfgaardian Knight</a></div>
        <div class="card-category">&nbsp;</div>
    </div>
    <div class="card-body"></div>
</div>
"""

# A card with a standalone single-clause, single-keyword ability and
# no body text after the keyword — the minimal "one line, one clause"
# case, distinct from _WEREWOLF_HTML where Resilience only ever
# appears as the first of several clauses.
_ARMORED_UNIT_HTML = """
<div class="card-wrap card-data" data-id="200600" data-power="4"
     data-armor="0" data-provision="6" data-faction="northern realms"
     data-set="base" data-color="bronze" data-type="unit"
     data-rarity="rare">
    <div class="card-head">
        <div class="card-name"><a href="https://gwent.one/en/card/200600">Blue Stripes Commando</a></div>
        <div class="card-category">&nbsp;</div>
    </div>
    <div class="card-body">
        <div class="card-body-ability"><span class="keyword resilient">Resilience</span>.<br /></div>
    </div>
</div>
"""


def _write_page(path: Path, *card_html: str) -> None:
    path.write_text('<div id="card-listing">' + "".join(card_html) + "</div>")


class TestIngest:
    def test_parses_one_candidate_per_card_in_one_page(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _MASK_OF_UROBOROS_HTML, _WEREWOLF_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert len(candidates) == 2
        assert {c.card.name for c in candidates} == {"Mask of Uroboros", "Werewolf"}

    def test_parses_across_multiple_pages(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _MASK_OF_UROBOROS_HTML)
        _write_page(tmp_path / "page_2.html", _WEREWOLF_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert {c.card.name for c in candidates} == {"Mask of Uroboros", "Werewolf"}

    def test_maps_data_id_to_provenance_source_id(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _MASK_OF_UROBOROS_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].card.provenance.source_id == "202512"
        assert candidates[0].card.provenance.data_source == DataSource.GWENT_ONE

    def test_raw_content_has_stripped_data_attrs_plus_name_category_ability(
        self, tmp_path: Path
    ) -> None:
        _write_page(tmp_path / "page_1.html", _MASK_OF_UROBOROS_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)
        raw_content = candidates[0].card.raw_content

        assert raw_content["id"] == "202512"
        assert raw_content["power"] == "0"
        assert raw_content["faction"] == "skellige"
        assert raw_content["set"] == "merchants of ofir"
        assert raw_content["rarity"] == "legendary"
        assert raw_content["name"] == "Mask of Uroboros"
        assert raw_content["category"] == "Location"
        assert raw_content["ability_text"] == (
            "Order: Draw a card, then Discard a card and Spawn 2 Crows on your "
            "Melee row."
        )
        assert "data-id" not in raw_content

    def test_flattens_multi_clause_ability_text_with_newlines(
        self, tmp_path: Path
    ) -> None:
        _write_page(tmp_path / "page_1.html", _WEREWOLF_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].card.raw_content["ability_text"] == (
            "Resilience.\n"
            "Deploy: Spawn Imperial Fleet to the left of self and Imperial "
            "Marine to the right of self.\n"
            "Order: Boost 2 allied units by 2 and swap their positions."
        )

    def test_standalone_single_clause_ability_has_no_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        _write_page(tmp_path / "page_1.html", _ARMORED_UNIT_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].card.raw_content["ability_text"] == "Resilience."

    def test_empty_category_becomes_empty_string(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _WEREWOLF_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].card.raw_content["category"] == ""

    def test_no_ability_div_becomes_empty_string(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _VANILLA_UNIT_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].card.raw_content["ability_text"] == ""

    def test_threads_source_game_through_unchanged(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _MASK_OF_UROBOROS_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].card.source_game == GameId.GWENT

    def test_each_card_gets_a_distinct_uuid(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _MASK_OF_UROBOROS_HTML, _WEREWOLF_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].card.nocab_uuid != candidates[1].card.nocab_uuid

    def test_aliases_are_always_empty(self, tmp_path: Path) -> None:
        _write_page(tmp_path / "page_1.html", _MASK_OF_UROBOROS_HTML)

        candidates = GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

        assert candidates[0].aliases == []

    def test_raises_if_raw_path_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            GwentOneCardIngestionStage().ingest(tmp_path / "missing", GameId.GWENT)

    def test_raises_if_raw_path_is_not_a_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir.html"
        file_path.write_text("<html></html>")

        with pytest.raises(ValueError):
            GwentOneCardIngestionStage().ingest(file_path, GameId.GWENT)

    def test_raises_if_no_page_html_files_found(self, tmp_path: Path) -> None:
        (tmp_path / "other.html").write_text("<html></html>")

        with pytest.raises(ValueError):
            GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

    def test_raises_if_card_block_missing_data_id(self, tmp_path: Path) -> None:
        malformed_html = (
            '<div class="card-wrap card-data" data-power="0">'
            '<div class="card-name"><a href="#">Mystery Card</a></div>'
            "</div>"
        )
        _write_page(tmp_path / "page_1.html", malformed_html)

        with pytest.raises(ValueError):
            GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)

    def test_raises_if_card_block_missing_name(self, tmp_path: Path) -> None:
        malformed_html = '<div class="card-wrap card-data" data-id="999999"></div>'
        _write_page(tmp_path / "page_1.html", malformed_html)

        with pytest.raises(ValueError):
            GwentOneCardIngestionStage().ingest(tmp_path, GameId.GWENT)
