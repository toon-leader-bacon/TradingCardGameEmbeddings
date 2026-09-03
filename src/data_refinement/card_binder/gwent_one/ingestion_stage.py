"""Translates gwent.one raw HTML page dumps into candidate cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and this container's
README for the raw source this reads (data/raw/gwent_one/page_*.html,
written by src/data_retrieval/gwent_one/downloader.py).

Like PokemonTcgCardIngestionStage (../pokemon_tcg/ingestion_stage.py)
and unlike ScryfallCardIngestionStage's single .jsonl file, this stage
reads a *directory* of page_*.html files — GwentOneDownloader.fetch()
can write more than one page, and there's no reason for this stage to
assume exactly one.

Confirmed by sampling a live gwent.one AJAX response directly this
session:
  - Each card is one `<div class="card-wrap card-data" data-id=...
    ...>` block. `data-id` is used as Provenance.source_id — the
    site's own per-card identity.
  - No secondary identifier system exists in this data comparable to
    Scryfall's arena_id/mtgo_id/multiverse_ids — IngestedCandidate.aliases
    is always [] for this source, same reasoning as
    PokemonTcgCardIngestionStage.
  - raw_content is a flat dict: every data-* attribute (prefix
    stripped, values kept as raw strings), plus "name" (card-name's
    text), "category" (card-category's text — empty string is valid,
    some cards have none), and "ability_text" (card-body-ability's
    text with keyword <span>s unwrapped to their inner text and <br>
    converted to "\\n" between ability clauses — plain text, matching
    Scryfall's oracle_text convention, not raw markup or a structured
    keyword breakdown).
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from bs4 import BeautifulSoup, Tag

from src.data_refinement.card_binder.ingestion import IngestedCandidate
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_CARD_BLOCK_SELECTOR = "div.card-wrap.card-data"


class GwentOneCardIngestionStage:
    """Translates a gwent.one raw HTML page directory into IngestedCandidates.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly (they go through
    CardIngestionStage/build_or_update_card_binder instead).
    """

    def ingest(self, raw_path: Path, source_game: GameId) -> list[IngestedCandidate]:
        """Parse every card-wrap block across raw_path's page_*.html
        files into IngestedCandidates.

        Composed of _find_card_blocks(raw_path) followed by one
        _parse_card_block() call per block found — no additional logic
        beyond calling the two and collecting the result. No
        filtering, no deduplication (including across pages — if the
        same card ever appears on two fetched pages, both become
        separate candidates and CardBinder.add()'s existing
        richness/collision handling is what collapses them, same as
        every other stage) — see
        src/data_refinement/card_binder/ingestion.py's module
        docstring.

        Inputs:
            raw_path: path to a directory of gwent.one page_*.html
                files (e.g. data/raw/gwent_one, written by
                src/data_retrieval/gwent_one/downloader.py). Not a
                single file — see this module's docstring.
            source_game: which game these cards belong to (expected to
                be GameId.GWENT for this implementation, but not
                enforced).
        Output: one IngestedCandidate per card-wrap block found across
            every page_*.html file under raw_path.
        Side effects: none — reads raw_path's page_*.html files, no
            other I/O.
        Exceptions: raises if raw_path doesn't exist, isn't a
            directory, or contains no page_*.html files (see
            _find_card_blocks); raises if a found card-wrap block is
            missing a data-id or a non-empty name (see
            _parse_card_block/_extract_raw_content) — DESIGN CHECKPOINT:
            a single malformed block is treated as a hard failure for
            the whole ingest() call here, matching
            ScryfallCardIngestionStage's strictness rather than
            HearthstoneJsonDownloader's per-item best-effort tolerance.
            Rationale: this reads a single well-formed, site-generated
            dump, not a flaky multi-request crawl, so a missing
            id/name signals this stage's own parsing assumptions are
            wrong, not that one item among many independently failed.
            Flagging for explicit human sign-off rather than assuming
            silently — revisit if that stops being true (e.g. once
            multi-page/filtered fetches are actually exercised and
            page-level partial failure becomes a real scenario).

        Example:
            >>> stage = GwentOneCardIngestionStage()
            >>> candidates = stage.ingest(
            ...     Path("data/raw/gwent_one"),
            ...     GameId.GWENT,
            ... )
        """
        card_blocks = self._find_card_blocks(raw_path)
        return [self._parse_card_block(block, source_game) for block in card_blocks]

    def _find_card_blocks(self, raw_path: Path) -> list[Tag]:
        """Find every `card-wrap card-data` div across raw_path's
        page_*.html files.

        Private helper — single consumer is ingest(). Parses each file
        with BeautifulSoup (html.parser backend) and selects every
        match for _CARD_BLOCK_SELECTOR.

        Inputs:
            raw_path: path to a directory of gwent.one page_*.html
                files.
        Output: every matching Tag found, in glob order over
            page_*.html filenames, then document order within each
            file.
        Side effects: reads every page_*.html file directly under
            raw_path.
        Exceptions: raises if raw_path doesn't exist, isn't a
            directory, or contains no page_*.html files — same
            ValueError-on-structural-mismatch convention as
            PokemonTcgCardIngestionStage.ingest().
        """
        if not raw_path.is_dir():
            raise ValueError(f"gwent_one ingest: {raw_path} is not a directory")

        page_paths = sorted(raw_path.glob("page_*.html"))
        if not page_paths:
            raise ValueError(
                f"gwent_one ingest: no page_*.html files found under {raw_path}"
            )

        card_blocks: list[Tag] = []
        for page_path in page_paths:
            html = page_path.read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "html.parser")
            card_blocks.extend(soup.select(_CARD_BLOCK_SELECTOR))
        return card_blocks

    def _parse_card_block(
        self, card_block: Tag, source_game: GameId
    ) -> IngestedCandidate:
        """Translate one `card-wrap card-data` block into an
        IngestedCandidate.

        Private helper — single consumer is ingest(). Composed of
        _extract_raw_content(card_block) to build the flat raw_content
        dict, then building a GenericCard (nocab_uuid=uuid4(),
        source_game, name=raw_content["name"], raw_content,
        provenance=Provenance(data_source=DataSource.GWENT_ONE,
        source_id=raw_content["id"], fetched_at=<now>)), with
        IngestedCandidate.aliases always [] — see this module's
        docstring for why no aliases are extracted for this source.

        Inputs:
            card_block: one parsed `<div class="card-wrap
                card-data">` Tag, e.g. one entry from
                _find_card_blocks()'s result.
            source_game: which game this block belongs to.
        Output: an IngestedCandidate wrapping the translated
            GenericCard, with an empty aliases list.
        Side effects: none.
        Exceptions: raises if _extract_raw_content(card_block) raises.
        """
        raw_content = self._extract_raw_content(card_block)
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=source_game,
            name=raw_content["name"],
            raw_content=raw_content,
            provenance=Provenance(
                data_source=DataSource.GWENT_ONE,
                source_id=raw_content["id"],
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        return IngestedCandidate(card=card, aliases=[])

    def _extract_raw_content(self, card_block: Tag) -> dict:
        """Build the flat raw_content dict for one card-wrap block.

        Inputs:
            card_block: one parsed `<div class="card-wrap
                card-data">` Tag.
        Output: a flat dict — every data-* attribute on card_block
            with the "data-" prefix stripped (values kept as raw
            strings, e.g. {"id": "203239", "power": "0", "armor": "0",
            "provision": "12", "faction": "nilfgaard", "set":
            "uroboros", "color": "gold", "type": "artifact", "rarity":
            "legendary"}), plus "name" (card_block's nested
            .card-name text, stripped), "category" (card_block's
            nested .card-category text, stripped — empty string is a
            valid, expected value, not an error), and "ability_text"
            (see _flatten_ability_text(card_block)).
        Side effects: none.
        Exceptions: raises ValueError if card_block has no "id"
            data-* attribute, or its .card-name text is empty after
            stripping — same "id or name missing is a hard failure"
            convention as ScryfallCardIngestionStage/
            PokemonTcgCardIngestionStage, adapted to bs4 lookups
            instead of dict key access.
        """
        raw_content = {
            key.removeprefix("data-"): value
            for key, value in card_block.attrs.items()
            if key.startswith("data-")
        }
        if "id" not in raw_content:
            raise ValueError(
                "gwent_one ingest: card-wrap block has no data-id attribute"
            )

        name_tag = card_block.select_one(".card-name")
        name = name_tag.get_text(strip=True) if name_tag is not None else ""
        if not name:
            raise ValueError(
                f"gwent_one ingest: card {raw_content['id']} has no .card-name text"
            )
        raw_content["name"] = name

        category_tag = card_block.select_one(".card-category")
        raw_content["category"] = (
            category_tag.get_text(strip=True) if category_tag is not None else ""
        )

        raw_content["ability_text"] = self._flatten_ability_text(card_block)

        return raw_content

    def _flatten_ability_text(self, card_block: Tag) -> str:
        """Extract card_block's ability text as plain text, one line
        per ability clause.

        Every keyword `<span class="keyword ...">` is unwrapped to
        just its inner text (the span/class is discarded, not kept as
        markup or structured data — see this module's docstring), and
        every `<br>` inside .card-body-ability becomes a "\\n"
        separating ability clauses, matching Scryfall's oracle_text
        convention (plain text, "\\n" between ability lines) rather
        than raw HTML or a keyword-tagged structure.

        Inputs:
            card_block: one parsed `<div class="card-wrap
                card-data">` Tag.
        Output: plain-text ability text; "" if card_block has no
            .card-body-ability div (some card types have no ability
            text at all — treated as valid/expected, not an error,
            same spirit as "category" tolerating "").
        Side effects: none.
        Exceptions: none.
        """
        ability_div = card_block.select_one(".card-body-ability")
        if ability_div is None:
            return ""

        for keyword_span in ability_div.select("span.keyword"):
            keyword_span.unwrap()
        for line_break in ability_div.find_all("br"):
            line_break.replace_with("\n")

        lines = [line.strip() for line in ability_div.get_text().split("\n")]
        return "\n".join(line for line in lines if line)
