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
extract_one()'s docstring for the empty/missing fallback), and
"deck"."srcCardTemplates": every card template id in the deck,
INCLUDING the leader and stratagem, with duplicates already
representing copy counts (e.g. a card appearing twice means 2 copies).
This is already exactly deck_box's own card_nocab_uuids multiset shape
(see extraction.py's module docstring) — no separate walk of
"deck"."cards" (which lists each *unique* card's own metadata, not
copy counts) is needed.

TWO OBSERVED SHAPES FOR srcCardTemplates: CONFIRMED LIVE against
data/raw/play_gwent/guides.jsonl (5.2GB) at multiple file offsets —
roughly the first half of the file (older guides) encodes
srcCardTemplates as a JSON OBJECT, {deck_slot_index_str: card_id, ...}
(e.g. {"0": 202186, "2": 203242, ..., "24": 162315, ...} — note the
deck-slot keys are NOT contiguous, some slots are skipped), while
roughly the second half (newer guides) encodes it as a plain JSON
ARRAY of card ids directly (e.g. [202185, 202493, ...]). Both shapes
carry the same information (a multiset of card ids); only the JSON
container differs, apparently a playgwent.com API/schema change
somewhere in this guide corpus' history. _card_template_ids() is where
this is normalized to "just the ids" regardless of which shape a given
row uses — iterating a dict directly (as an earlier version of this
method did) yields its STRING KEYS (the deck-slot indices, e.g. "24"),
not the card ids they map to, which silently misresolves every card in
every dict-shaped guide against the Unknown sentinel.

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

PROVENANCE: each created/updated deck carries a Provenance
(DataSource.PLAY_GWENT, source_id=str(guide_id), fetched_at=now) — the
guide's own row id (this stage's own deck identity, see IDEMPOTENT
RE-RUNS above), not GWENT_ONE (that's each card's own data source, used
only for card resolution). Refreshed on every content-changing
update(), not just on first create().
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid5

from tqdm import tqdm

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_retrieval.play_gwent.downloader import PlayGwentDownloader
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

# Fixed, arbitrary — never regenerate. Namespace for this stage's
# deterministic per-guide deck uuids (see module docstring's IDEMPOTENT
# RE-RUNS section).
_DECK_NAMESPACE = UUID("8f1c2b3a-4d5e-4f60-9a7b-1c2d3e4f5061")


class PlayGwentDeckExtractionStage:
    """Translates playgwent.com's guides.jsonl directly into a DeckBox.

    extract() (whole-file) is single-consumer to
    src/data_refinement/deck_box/, but extract_one() and
    deck_uuid_for_guide() are also called directly by
    src/data_refinement/metrics/play_gwent/'s deck-masking metrics, so
    every raw guide row becomes a deck the same way regardless of
    caller — see those two methods' own docstrings and
    plans/deck_card_masking.md.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        PlayGwentDownloader.DEFAULT_RAW_DATA_DIR / "guides.jsonl"
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Parse playgwent.com's guides.jsonl, creating/updating decks on box.

        One extract_one() call per line of raw_path — no additional
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
            falls back to the Unknown sentinel. Prints a tqdm progress
            bar to stderr, sized against raw_path's byte size
            (guides.jsonl runs to several GB, so a line-count total
            isn't worth a separate full read to compute).
        Exceptions: raises if raw_path doesn't exist, isn't valid
            JSONL, or a line is missing "id" or
            "deck"."srcCardTemplates". Raises RuntimeError if
            self.SOURCE_GAME's Unknown sentinel card isn't found on
            card_lookup.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/gwent.jsonl")])
            >>> box = DeckBox.load([Path("data/final/decks/gwent.db")])
            >>> stage = PlayGwentDeckExtractionStage()
            >>> changed_uuids = stage.extract(None, box, binder)
        """
        path = raw_path or self.DEFAULT_RAW_PATH

        changed_uuids = []
        total_bytes = path.stat().st_size
        with open(path, "r", encoding="utf-8") as raw_file, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=f"play_gwent extract: {path.name}",
        ) as progress:
            for line in raw_file:
                progress.update(len(line.encode("utf-8")))
                if not line.strip():
                    continue
                guide = json.loads(line)
                result = self.extract_one(guide, box, card_lookup)
                if result is not None:
                    changed_uuids.append(result)
        return changed_uuids

    def extract_one(
        self, guide: dict, box: DeckBox, card_lookup: CardLookup
    ) -> UUID | None:
        """Create-or-update one playgwent.com guide directly against box.

        Public per-row entry point — extract() itself is just this
        method called in a loop over every line of raw_path. Also
        called directly by src/data_refinement/metrics/play_gwent/'s
        deck-masking metrics, which need this exact same "one raw
        guide -> its deck exists in box" side effect on every raw row
        they see (see plans/deck_card_masking.md) without duplicating
        this class's card-resolution/uuid-minting logic. THIS METHOD
        IS WHERE IDEMPOTENCY HAPPENS: deck_uuid is a pure function of
        guide["id"] (see module docstring's IDEMPOTENT RE-RUNS
        section, and deck_uuid_for_guide()), so re-processing the same
        guide always targets the same stored deck.

        Inputs:
            guide: one parsed JSON object from raw_path (one guide),
                carrying at least "id" (int) and
                "deck"."srcCardTemplates" (dict or list — see module
                docstring's TWO OBSERVED SHAPES section). May carry
                "name" (str, possibly absent/empty).
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
        Side effects: creates or updates exactly one deck on box, with
            a freshly-computed Provenance (see module docstring's
            PROVENANCE section) — set on create(), and refreshed on a
            content-changing update() too.
        Exceptions: raises if guide is missing "id" or
            "deck"."srcCardTemplates". Whatever _card_uuid() raises
            propagates.
        """
        guide_id = guide["id"]
        deck_uuid = self.deck_uuid_for_guide(guide_id)

        # Resolve every deck entry's card template id — never None,
        # falls back to the Unknown sentinel on a miss (see
        # _card_uuid()). _card_template_ids() already yields the flat
        # multiset regardless of srcCardTemplates' raw shape (see
        # module docstring's TWO OBSERVED SHAPES section) — no
        # grouping/dedup of our own.
        card_nocab_uuids = [
            self._card_uuid(raw_card_id, card_lookup)
            for raw_card_id in self._card_template_ids(
                guide["deck"]["srcCardTemplates"]
            )
        ]

        deck_name = guide.get("name") or f"playgwent.com guide {guide_id}"
        provenance = Provenance(
            data_source=DataSource.PLAY_GWENT,
            source_id=str(guide_id),
            fetched_at=datetime.now(timezone.utc),
        )

        existing = box.get_by_uuid(deck_uuid)
        if existing is None:
            # Brand-new guide.
            box.create(
                GenericDeck(
                    nocab_uuid=deck_uuid,
                    source_game=self.SOURCE_GAME,
                    name=deck_name,
                    card_nocab_uuids=card_nocab_uuids,
                    provenance=provenance,
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
            # first seen. provenance IS refreshed, since the content
            # itself did change.
            box.update(
                deck_uuid, card_nocab_uuids=card_nocab_uuids, provenance=provenance
            )
            return deck_uuid

        # Re-seen guide, identical resolved multiset — no-op.
        return None

    @staticmethod
    def _card_template_ids(src_card_templates: dict | list) -> list:
        """Flatten one guide's raw srcCardTemplates into a plain list
        of card template ids, regardless of which raw shape it uses.

        Private helper — single consumer is extract_one(). See module
        docstring's TWO OBSERVED SHAPES section: a dict's VALUES are
        the card ids (its keys are deck-slot indices, not ids — the
        bug this guards against), while a list already IS the card
        ids.

        Inputs:
            src_card_templates: one guide's "deck"."srcCardTemplates"
                value, either {deck_slot_index_str: card_id} or
                [card_id, ...].
        Output: the card ids alone, in src_card_templates' own
            iteration order, duplicates preserved.
        Side effects: none.
        Exceptions: none.
        """
        if isinstance(src_card_templates, dict):
            return list(src_card_templates.values())
        return list(src_card_templates)

    def _card_uuid(self, raw_card_id: int, card_lookup: CardLookup) -> UUID:
        """Look up the nocab_uuid for one gwent.one card template id.

        Private helper — single consumer is extract_one(). Looks
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
    def deck_uuid_for_guide(guide_id: int) -> UUID:
        """Compute the deterministic nocab_uuid for one playgwent.com guide.

        Public — called from extract_one() above, and directly by
        src/data_refinement/metrics/play_gwent/'s deck-masking metrics,
        which need this exact id (independent of whether extract_one()
        found this guide's deck already up to date and returned None)
        for every raw row they process — see
        plans/deck_card_masking.md. uuid5 (not uuid4): the same
        guide_id must always produce the same uuid, across every call,
        so a re-seen guide updates rather than duplicates (see module
        docstring's IDEMPOTENT RE-RUNS section).

        Inputs:
            guide_id: one guide's own "id" field.
        Output: a uuid unique to (this fixed namespace, str(guide_id))
            — stable forever for a given guide_id.
        Side effects: none.
        Exceptions: none.
        """
        return uuid5(_DECK_NAMESPACE, str(guide_id))
