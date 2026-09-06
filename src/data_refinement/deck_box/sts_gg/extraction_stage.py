"""Translates sts_gg run data into decks, directly into a DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface this implements, and
src/data_refinement/card_binder/spire_codex/ingestion_stage.py for the
sibling this was deliberately modeled on.

CARD RESOLUTION: identical to the existing
src/data_refinement/sts_gg/deck_outcome_metric.py's DeckOutcomeMetric —
sts_gg's raw deck entries carry ids like "CARD.GRAND_FINALE";
stripping the literal "CARD." prefix reproduces spire_codex's own "id"
exactly, so resolution is a plain
card_lookup.get_by_alias(GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX,
...) call, not a fuzzy/name-based match. See that module's docstring
for the full reasoning; not repeated here.

UNRESOLVED CARDS: unlike DeckOutcomeMetric (which drops an unresolved
card from that run's list and logs it), this stage substitutes the
Unknown sentinel card's nocab_uuid instead (see
CardBinder.ensure_unknown_card()) — the deck's card count stays
correct (a real copy occupied a real deck slot), and the substitution
is still logged loudly via the stdlib `logging` module.
PRECONDITION: card_lookup must already have this game's Unknown
sentinel card seeded (CardBinder.ensure_unknown_card(GameId.SLAY_THE_SPIRE_2),
called once against a real CardBinder before any extract() call) — this
stage stays read-only (CardLookup, never a full CardBinder) and does
NOT create that card lazily; a missing sentinel is a setup bug, so it
raises RuntimeError rather than silently dropping the card.

IDEMPOTENT RE-RUNS: unlike a typical DeckExtractionStage (see
extraction.py's module docstring), this one calls box.update() on a
re-seen run rather than duplicating it — made possible by deriving a
deck's nocab_uuid deterministically from sts_gg's own run id
(uuid5(_DECK_NAMESPACE, run_id)), rather than minting a fresh uuid4 per
extract() call.

DELIBERATELY DROPPED: sts_gg's "win" field (DeckOutcomeMetric's whole
reason for existing) is not carried into GenericDeck at all —
GenericDeck has no metadata field (see
src/data_refinement/deck_box/README.md's multiset/no-metadata section).
Carrying win/outcome forward as a separate metric keyed by this stage's
deck nocab_uuid is future work, not this file's concern.
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
from src.data_retrieval.sts_gg.run_downloader import STSGGRunDownloader
from src.schema.card import GenericDeck
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_STS_GG_CARD_ID_PREFIX = "CARD."

# Fixed, arbitrary — never regenerate. Namespace for this stage's
# deterministic per-run deck uuids (see module docstring's IDEMPOTENT
# RE-RUNS section).
_DECK_NAMESPACE = UUID("2f6f6a5a-9b0a-4a8b-8b0a-7a1e6f2e5c3d")


class StsGgDeckExtractionStage:
    """Translates sts_gg's runs.jsonl directly into a DeckBox.

    Single-consumer to src/data_refinement/deck_box/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        STSGGRunDownloader.DEFAULT_RAW_DATA_DIR / "runs.jsonl"
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Parse sts_gg's runs.jsonl, creating/updating decks on box.

        One _extract_row() call per line of raw_path — no additional
        logic beyond calling that and collecting the non-None results.

        Inputs:
            raw_path: path to a runs.jsonl file (one JSON object per
                line — e.g. data/raw/sts_gg/runs.jsonl). Defaults to
                DEFAULT_RAW_PATH when None.
            box: the DeckBox to create/update decks on, as a side
                effect. Every deck this call touches is stored under
                self.SOURCE_GAME.
            card_lookup: must already have self.SOURCE_GAME's Unknown
                sentinel card seeded (see module docstring's
                PRECONDITION) in addition to spire_codex's cards.
        Output: nocab_uuid of every run whose resolved card list this
            call created or changed. A re-seen run whose resolved list
            is identical to what's already stored is NOT included.
        Side effects: reads raw_path; creates/updates decks directly
            on box; emits one logging.error() per card that falls back
            to the Unknown sentinel.
        Exceptions: raises if raw_path doesn't exist, isn't valid
            JSONL, or a line is missing "id" or "deck". Raises
            RuntimeError if self.SOURCE_GAME's Unknown sentinel card
            isn't found on card_lookup (see module docstring's
            PRECONDITION).

        Example:
            >>> binder = CardBinder.load(
            ...     [Path("data/final/cards/slay_the_spire_2.jsonl")]
            ... )
            >>> box = DeckBox.load([Path("data/final/decks/slay_the_spire_2.jsonl")])
            >>> stage = StsGgDeckExtractionStage()
            >>> changed_uuids = stage.extract(None, box, binder)
        """
        path = raw_path or self.DEFAULT_RAW_PATH

        changed_uuids = []
        with open(path, "r", encoding="utf-8") as raw_file:
            for line in raw_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                result = self._extract_row(row, box, card_lookup)
                if result is not None:
                    changed_uuids.append(result)
        return changed_uuids

    def _extract_row(
        self, row: dict, box: DeckBox, card_lookup: CardLookup
    ) -> UUID | None:
        """Create-or-update one sts_gg run directly against box.

        Private helper — single consumer is extract(). THIS METHOD IS
        WHERE IDEMPOTENCY HAPPENS: deck_uuid is a pure function of
        row["id"] (see module docstring's IDEMPOTENT RE-RUNS section),
        so re-processing the same run always targets the same stored
        deck.

        Inputs:
            row: one parsed JSON object from raw_path (one run),
                carrying at least "id" (str) and "deck" (list of
                {"id": str, ...} card entries).
            box: the DeckBox to read from and write to.
            card_lookup: used to resolve each card entry's id (see
                _card_uuid()).
        Output: deck_uuid if this call caused a create() or an actual
            content-changing update(); None if this run matched an
            already-stored deck with an identical resolved card list.
        Side effects: creates or updates exactly one deck on box.
        Exceptions: raises if row is missing "id" or "deck". Whatever
            _card_uuid() raises (see its own Exceptions) propagates.
        """
        run_id = row["id"]
        deck_uuid = self._deck_uuid(run_id)

        # Resolve every deck entry's card id — never None, falls back
        # to the Unknown sentinel on a miss (see _card_uuid()).
        card_nocab_uuids = [
            self._card_uuid(card_entry["id"], card_lookup) for card_entry in row["deck"]
        ]

        existing = box.get_by_uuid(deck_uuid)
        if existing is None:
            # Brand-new run.
            box.create(
                GenericDeck(
                    nocab_uuid=deck_uuid,
                    source_game=self.SOURCE_GAME,
                    name=f"sts_gg run {run_id}",
                    card_nocab_uuids=card_nocab_uuids,
                )
            )
            return deck_uuid

        # card_nocab_uuids is a multiset (see src/schema/card.py) —
        # order is not guaranteed, so compare copy counts per uuid via
        # Counter, never by list equality (order-sensitive) or set
        # equality (loses duplicate/copy-count information).
        if Counter(existing.card_nocab_uuids) != Counter(card_nocab_uuids):
            # Re-seen run whose resolved list changed — update in place.
            box.update(deck_uuid, card_nocab_uuids=card_nocab_uuids)
            return deck_uuid

        # Re-seen run, identical resolved multiset — no-op.
        return None

    def _card_uuid(self, raw_card_id: str, card_lookup: CardLookup) -> UUID:
        """Look up the nocab_uuid for one sts_gg deck entry's card id.

        Private helper — single consumer is _extract_row(). Strips
        sts_gg's "CARD." prefix and looks the remainder up directly
        against spire_codex's own registered aliases (see module
        docstring's CARD RESOLUTION section). On a miss, logs loudly
        and falls back to the Unknown sentinel's uuid instead of
        returning None (see module docstring's UNRESOLVED CARDS
        section) — unlike DeckOutcomeMetric._card_uuid(), this method
        never returns "no card at all" for a deck slot.

        Inputs:
            raw_card_id: one deck entry's "id" field, e.g.
                "CARD.GRAND_FINALE".
        Output: the matching nocab_uuid, or (on a miss) the Unknown
            sentinel's nocab_uuid.
        Side effects: emits one logging.error() call on a miss.
        Exceptions: raises RuntimeError if even the Unknown sentinel
            isn't found on card_lookup (see module docstring's
            PRECONDITION).
        """
        stripped_id = raw_card_id.removeprefix(_STS_GG_CARD_ID_PREFIX)
        card = card_lookup.get_by_alias(
            self.SOURCE_GAME, DataSource.SPIRE_CODEX, stripped_id
        )
        if card is not None:
            return card.nocab_uuid

        _logger.error(
            "StsGgDeckExtractionStage: unresolved card id %r — "
            "substituting the Unknown sentinel card",
            raw_card_id,
        )
        unknown_card = card_lookup.get_by_name_single(
            self.SOURCE_GAME, CardBinder.UNKNOWN_CARD_NAME, strict=False
        )
        if unknown_card is None:
            raise RuntimeError(
                f"StsGgDeckExtractionStage: {self.SOURCE_GAME!r}'s Unknown "
                "sentinel card is not seeded — call "
                "CardBinder.ensure_unknown_card() before extract()"
            )
        return unknown_card.nocab_uuid

    @staticmethod
    def _deck_uuid(run_id: str) -> UUID:
        """Compute the deterministic nocab_uuid for one sts_gg run.

        Private helper — single consumer is _extract_row(). uuid5 (not
        uuid4): the same run_id must always produce the same uuid,
        across every extract() call, so a re-seen run updates rather
        than duplicates (see module docstring's IDEMPOTENT RE-RUNS
        section).

        Inputs:
            run_id: one run's own "id" field.
        Output: a uuid unique to (this fixed namespace, run_id) —
            stable forever for a given run_id.
        Side effects: none.
        Exceptions: none.
        """
        return uuid5(_DECK_NAMESPACE, run_id)
