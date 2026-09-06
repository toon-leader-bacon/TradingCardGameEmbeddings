"""Translates gwent.one raw HTML page dumps into stored cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and plans/card_binder_v2.md
for the design this implements. This class is handed a live CardBinder
and owns its own duplicate-detection and collision-resolution directly
against it — see src/data_refinement/card_binder/scryfall/ingestion_stage.py
for the reference shape this follows.

This stage reads a *directory* of page_*.html files — GwentOneDownloader.fetch()
can write more than one page, and there's no reason for this stage to
assume exactly one.

Confirmed by sampling a live gwent.one AJAX response directly:
  - Each card is one `<div class="card-wrap card-data" data-id=...
    ...>` block. `data-id` is used as Provenance.source_id — the
    site's own per-card identity, and (like Scryfall's oracle_id and
    spire-codex's id) a perfect natural key, so identity is a plain
    get_by_alias() lookup with no heuristic fallback needed.
  - No secondary identifier system exists in this data comparable to
    Scryfall's arena_id/mtgo_id/multiverse_ids — no extra aliases are
    ever registered for this source, only each row's own data-id.
  - raw_content is a flat dict: every data-* attribute (prefix
    stripped, values kept as raw strings), plus "name" (card-name's
    text), "category" (card-category's text — empty string is valid,
    some cards have none), and "ability_text" (card-body-ability's
    text with keyword <span>s unwrapped to their inner text and <br>
    converted to "\\n" between ability clauses — plain text, matching
    Scryfall's oracle_text convention, not raw markup or a structured
    keyword breakdown).

Collision policy is merge_strategies.keep_longer_content, same as
ScryfallCardIngestionStage.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

from bs4 import BeautifulSoup, Tag

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_CARD_BLOCK_SELECTOR = "div.card-wrap.card-data"


class GwentOneCardIngestionStage:
    """Translates a gwent.one raw HTML page directory directly into binder.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse every card-wrap block across raw_path's page_*.html
        files, creating/updating cards directly on binder as a side
        effect.

        Composed of _find_card_blocks(raw_path) followed by one
        _ingest_card_block() call per block found — no additional
        logic beyond calling the two and collecting the non-None
        results. No filtering — every block becomes exactly one
        create() or one considered-for-update() call.

        Inputs:
            raw_path: path to a directory of gwent.one page_*.html
                files (e.g. data/raw/gwent_one, written by
                src/data_retrieval/gwent_one/downloader.py). Not a
                single file — see this module's docstring.
            binder: the CardBinder to create/update cards on and
                register aliases against, as a side effect. Every card
                this call touches is stored under self.SOURCE_GAME
                (GameId.GWENT).
        Output: nocab_uuid of every block that caused a create() or an
            actual content-changing replace() this call. A block that
            matched an existing card but changed nothing (an
            exact-duplicate re-fetch — including a card appearing on
            two fetched pages) is NOT included.
        Side effects: reads raw_path's page_*.html files; creates/
            updates cards and registers aliases directly on binder.
        Exceptions: raises if raw_path doesn't exist, isn't a
            directory, or contains no page_*.html files (see
            _find_card_blocks); raises if a found card-wrap block is
            missing a data-id or a non-empty name (see
            _extract_raw_content) — DESIGN CHECKPOINT: a single
            malformed block is treated as a hard failure for the whole
            ingest() call here, matching ScryfallCardIngestionStage's
            strictness rather than HearthstoneJsonDownloader's
            per-item best-effort tolerance. Rationale: this reads a
            single well-formed, site-generated dump, not a flaky
            multi-request crawl, so a missing id/name signals this
            stage's own parsing assumptions are wrong, not that one
            item among many independently failed.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/gwent.jsonl")])
            >>> stage = GwentOneCardIngestionStage()
            >>> changed_uuids = stage.ingest(Path("data/raw/gwent_one"), binder)
        """
        card_blocks = self._find_card_blocks(raw_path)
        changed_uuids = []
        for card_block in card_blocks:
            result = self._ingest_card_block(card_block, binder)
            if result is not None:
                changed_uuids.append(result)
        return changed_uuids

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
            directory, or contains no page_*.html files.
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

    def _ingest_card_block(self, card_block: Tag, binder: CardBinder) -> UUID | None:
        """Create-or-merge one card-wrap block directly against binder.

        Private helper — single consumer is ingest(). THIS METHOD IS
        WHERE DEDUPLICATION HAPPENS: the identity check —
        binder.get_by_alias(self.SOURCE_GAME, DataSource.GWENT_ONE,
        raw_content["id"]) — IS the duplicate check. A hit means this
        block is a duplicate of the returned card; a miss means it's
        new.

        If no existing card resolves: builds a fresh GenericCard from
        _extract_raw_content(card_block) and calls binder.create().

        If an existing card resolves: builds the same kind of
        GenericCard as a throwaway candidate, then calls
        merge_strategies.keep_longer_content(existing, candidate).
        Compares the result to existing BY VALUE (dataclass equality):
        if different, calls binder.replace(existing.nocab_uuid,
        merged).

        Regardless of branch: registers this block's own data-id alias
        against whichever uuid ended up stored — on every branch, not
        only the content-changing ones.

        Inputs:
            card_block: one parsed `<div class="card-wrap
                card-data">` Tag.
            binder: the CardBinder to read from and write to.
        Output: the stored/canonical nocab_uuid for this block IF this
            call caused a create() or an actual content-changing
            replace(); None if this block matched an existing card but
            changed nothing.
        Side effects: creates or updates exactly one card on binder;
            registers one alias on binder.
        Exceptions: raises if _extract_raw_content(card_block) raises.
        """
        raw_content = self._extract_raw_content(card_block)
        card_id = raw_content["id"]
        existing = binder.get_by_alias(self.SOURCE_GAME, DataSource.GWENT_ONE, card_id)

        if existing is None:
            card = self._build_card(raw_content)
            binder.create(card)
            stored_uuid = card.nocab_uuid
            changed = True
        else:
            candidate = self._build_card(raw_content)
            merged = merge_strategies.keep_longer_content(existing, candidate)
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(
            self.SOURCE_GAME, DataSource.GWENT_ONE, card_id, stored_uuid
        )

        return stored_uuid if changed else None

    def _build_card(self, raw_content: dict) -> GenericCard:
        """Build a fresh GenericCard from one card-wrap block's raw_content.

        Private helper — single consumer is _ingest_card_block(), from
        both its create-path and its throwaway-candidate-for-merge
        path.

        Inputs:
            raw_content: the flat dict from _extract_raw_content().
        Output: a new GenericCard with a freshly minted nocab_uuid.
        Side effects: none.
        Exceptions: none — raw_content is already validated by
            _extract_raw_content() to carry "id" and "name".
        """
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=raw_content["name"],
            raw_content=raw_content,
            provenance=Provenance(
                data_source=DataSource.GWENT_ONE,
                source_id=raw_content["id"],
                fetched_at=datetime.now(timezone.utc),
            ),
        )

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
            stripping.
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
        markup or structured data), and every `<br>` inside
        .card-body-ability becomes a "\\n" separating ability clauses,
        matching Scryfall's oracle_text convention (plain text, "\\n"
        between ability lines) rather than raw HTML or a
        keyword-tagged structure.

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
