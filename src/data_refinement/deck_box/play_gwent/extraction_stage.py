"""Translates playgwent.com deck guides into decks, directly into a DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface this implements, and
src/data_refinement/deck_box/sts_gg/extraction_stage.py for the sibling
this is modeled on.

RAW SHAPE: src/data_retrieval/play_gwent/downloader.py's guides.jsonl —
one playgwent.com "guide" JSON object per line (that module's
PlayGwentDownloader.phase_2()'s _extract_guide_payload() already
unwraps the page's data-state payload down to this exact dict).
Confirmed against a live sample (nocab/blab.json): each row carries the
guide's own "id" (int, e.g. 407697 — this row's own identity, distinct
from the nested "deck"."id"), an optional "name" (the guide's human
title, e.g. "Unorganised crime" — some guides may have none/empty; used
verbatim as the stored GenericDeck.name when present — see
_extract_guide()'s docstring for the empty/missing fallback), and
"deck"."srcCardTemplates": a flat list[int] of every card template id
in the deck, INCLUDING the leader and stratagem, with duplicates
already representing copy counts (e.g. a card appearing twice in the
list means 2 copies). This is already exactly deck_box's own
card_nocab_uuids multiset shape (see extraction.py's module docstring)
— no separate walk of "deck"."cards" (which lists each *unique* card's
own metadata, not copy counts) is needed.

CARD RESOLUTION: each srcCardTemplates entry is gwent.one's own numeric
card id — the same id ingested as DataSource.GWENT_ONE's alias key by
src/data_refinement/card_binder/gwent_one/ingestion_stage.py (that
stage's data-id attribute, kept as a string there). Resolution is a
plain card_lookup.get_by_alias(GameId.GWENT, DataSource.GWENT_ONE,
str(raw_card_id)) call — no fuzzy/name-based match needed. This is a
cross-source resolution (playgwent.com deck data against gwent.one's
already-ingested cards) rather than a same-source one, but the
resolution mechanics are otherwise identical to every other
DeckExtractionStage in this container.

UNRESOLVED CARDS: same policy as StsGgDeckExtractionStage — a miss
substitutes the Unknown sentinel card's nocab_uuid (logged loudly)
rather than dropping the slot. PRECONDITION: card_lookup must already
have GameId.GWENT's Unknown sentinel card seeded
(CardBinder.ensure_unknown_card(GameId.GWENT), called once against a
real CardBinder before any extract() call) — this stage stays
read-only (CardLookup, never a full CardBinder) and does NOT create
that card lazily; a missing sentinel raises RuntimeError.

IDEMPOTENT RE-RUNS: deck identity is uuid5(_DECK_NAMESPACE,
str(guide_id)) — the guide's own row id, never a hash of resolved card
content (same explicit constraint as every other DeckExtractionStage
in this container) — so re-running extraction over the same
guides.jsonl updates rather than duplicates a deck.

STREAMING: guides.jsonl is large (observed ~5GB, ~60k lines) —
extract() reads it one line at a time (same convention as
StsGgDeckExtractionStage), never loading the whole file into memory.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid5

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_retrieval.play_gwent.downloader import PlayGwentDownloader
from src.schema.card import GenericDeck
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

# Fixed, arbitrary — never regenerate. Namespace for this stage's
# deterministic per-guide deck uuids (see module docstring's IDEMPOTENT
# RE-RUNS section).
_DECK_NAMESPACE = UUID("8f1c2b3a-4d5e-4f60-9a7b-1c2d3e4f5061")


class PlayGwentDeckExtractionStage:
    """Translates playgwent.com's guides.jsonl directly into a DeckBox.

    Single-consumer to src/data_refinement/deck_box/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        PlayGwentDownloader.DEFAULT_RAW_DATA_DIR / "guides.jsonl"
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Parse playgwent.com's guides.jsonl, creating/updating decks on box.

        One _extract_guide() call per line of raw_path — no additional
        logic beyond calling that and collecting the non-None results.

        Inputs:
            raw_path: path to a guides.jsonl file (one JSON guide
                object per line — e.g.
                data/raw/play_gwent/guides.jsonl). Defaults to
                DEFAULT_RAW_PATH when None.
            box: the DeckBox to create/update decks on, as a side
                effect. Every deck this call touches is stored under
                self.SOURCE_GAME.
            card_lookup: must already have self.SOURCE_GAME's Unknown
                sentinel card seeded (see module docstring's
                PRECONDITION) in addition to gwent.one's cards.
        Output: nocab_uuid of every guide whose resolved card list this
            call created or changed. A re-seen guide whose resolved
            list is identical to what's already stored is NOT included.
        Side effects: reads raw_path, one line at a time (see module
            docstring's STREAMING section); creates/updates decks
            directly on box; emits one logging.error() per card that
            falls back to the Unknown sentinel.
        Exceptions: raises if raw_path doesn't exist, isn't valid
            JSONL, or a line is missing "id" or
            "deck"."srcCardTemplates". Raises RuntimeError if
            self.SOURCE_GAME's Unknown sentinel card isn't found on
            card_lookup.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/gwent.jsonl")])
            >>> box = DeckBox.load([Path("data/final/decks/gwent.jsonl")])
            >>> stage = PlayGwentDeckExtractionStage()
            >>> changed_uuids = stage.extract(None, box, binder)
        """
        path = raw_path or self.DEFAULT_RAW_PATH

        changed_uuids = []
        with open(path, "r", encoding="utf-8") as raw_file:
            for line in raw_file:
                if not line.strip():
                    continue
                guide = json.loads(line)
                result = self._extract_guide(guide, box, card_lookup)
                if result is not None:
                    changed_uuids.append(result)
        return changed_uuids

    def _extract_guide(
        self, guide: dict, box: DeckBox, card_lookup: CardLookup
    ) -> UUID | None:
        """Create-or-update one playgwent.com guide directly against box.

        Private helper — single consumer is extract(). THIS METHOD IS
        WHERE IDEMPOTENCY HAPPENS: deck_uuid is a pure function of
        guide["id"] (see module docstring's IDEMPOTENT RE-RUNS
        section), so re-processing the same guide always targets the
        same stored deck.

        Inputs:
            guide: one parsed JSON object from raw_path (one guide),
                carrying at least "id" (int) and
                "deck"."srcCardTemplates" (list[int]). May carry "name"
                (str, possibly absent/empty).
            box: the DeckBox to read from and write to.
            card_lookup: used to resolve each card template id (see
                _card_uuid()).
        Output: deck_uuid if this call caused a create() or an actual
            content-changing update(); None if this guide matched an
            already-stored deck with an identical resolved card list.
            The stored GenericDeck.name is guide["name"] verbatim when
            present and non-empty (e.g. "Unorganised crime" — the
            guide author's own title, worth keeping since
            playgwent.com guides are human-curated and often named
            meaningfully, unlike sts_gg's anonymous runs), else
            f"playgwent.com guide {guide_id}".
        Side effects: creates or updates exactly one deck on box.
        Exceptions: raises if guide is missing "id" or
            "deck"."srcCardTemplates". Whatever _card_uuid() raises
            propagates.
        """
        guide_id = guide["id"]
        deck_uuid = self._deck_uuid(guide_id)

        # Resolve every deck entry's card template id — never None,
        # falls back to the Unknown sentinel on a miss (see
        # _card_uuid()). srcCardTemplates is already the flat multiset
        # (see module docstring's RAW SHAPE section) — no grouping/
        # dedup of our own.
        card_nocab_uuids = [
            self._card_uuid(raw_card_id, card_lookup)
            for raw_card_id in guide["deck"]["srcCardTemplates"]
        ]

        deck_name = guide.get("name") or f"playgwent.com guide {guide_id}"

        existing = box.get_by_uuid(deck_uuid)
        if existing is None:
            # Brand-new guide.
            box.create(
                GenericDeck(
                    nocab_uuid=deck_uuid,
                    source_game=self.SOURCE_GAME,
                    name=deck_name,
                    card_nocab_uuids=card_nocab_uuids,
                )
            )
            return deck_uuid

        # card_nocab_uuids is a multiset (see src/schema/card.py) —
        # order is not guaranteed, so compare copy counts per uuid via
        # Counter, never by list equality (order-sensitive) or set
        # equality (loses duplicate/copy-count information).
        if Counter(existing.card_nocab_uuids) != Counter(card_nocab_uuids):
            # Re-seen guide whose resolved list changed — update in
            # place. Deliberately does NOT also refresh existing.name
            # on a content-only update — matches box.update()'s own
            # "None leaves that field untouched" contract, and a
            # guide's title/id pairing isn't expected to drift once
            # first seen.
            box.update(deck_uuid, card_nocab_uuids=card_nocab_uuids)
            return deck_uuid

        # Re-seen guide, identical resolved multiset — no-op.
        return None

    def _card_uuid(self, raw_card_id: int, card_lookup: CardLookup) -> UUID:
        """Look up the nocab_uuid for one gwent.one card template id.

        Private helper — single consumer is _extract_guide(). Looks
        raw_card_id up directly against gwent.one's own registered
        aliases (see module docstring's CARD RESOLUTION section). On a
        miss, logs loudly and falls back to the Unknown sentinel's uuid
        instead of returning None (see module docstring's UNRESOLVED
        CARDS section) — this method never returns "no card at all"
        for a deck slot.

        Inputs:
            raw_card_id: one deck entry's card template id, e.g. 202338.
        Output: the matching nocab_uuid, or (on a miss) the Unknown
            sentinel's nocab_uuid.
        Side effects: emits one logging.error() call on a miss.
        Exceptions: raises RuntimeError if even the Unknown sentinel
            isn't found on card_lookup (see module docstring's
            PRECONDITION).
        """
        card = card_lookup.get_by_alias(
            self.SOURCE_GAME, DataSource.GWENT_ONE, str(raw_card_id)
        )
        if card is not None:
            return card.nocab_uuid

        _logger.error(
            "PlayGwentDeckExtractionStage: unresolved card template id %r — "
            "substituting the Unknown sentinel card",
            raw_card_id,
        )
        unknown_card = card_lookup.get_by_name_single(
            self.SOURCE_GAME, CardBinder.UNKNOWN_CARD_NAME, strict=False
        )
        if unknown_card is None:
            raise RuntimeError(
                f"PlayGwentDeckExtractionStage: {self.SOURCE_GAME!r}'s Unknown "
                "sentinel card is not seeded — call "
                "CardBinder.ensure_unknown_card() before extract()"
            )
        return unknown_card.nocab_uuid

    @staticmethod
    def _deck_uuid(guide_id: int) -> UUID:
        """Compute the deterministic nocab_uuid for one playgwent.com guide.

        Private helper — single consumer is _extract_guide(). uuid5
        (not uuid4): the same guide_id must always produce the same
        uuid, across every extract() call, so a re-seen guide updates
        rather than duplicates (see module docstring's IDEMPOTENT
        RE-RUNS section).

        Inputs:
            guide_id: one guide's own "id" field.
        Output: a uuid unique to (this fixed namespace, str(guide_id))
            — stable forever for a given guide_id.
        Side effects: none.
        Exceptions: none.
        """
        return uuid5(_DECK_NAMESPACE, str(guide_id))
