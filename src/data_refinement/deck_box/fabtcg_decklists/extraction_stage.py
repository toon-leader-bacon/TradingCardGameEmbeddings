"""Translates fabtcg.com decklist HTML fragments into decks, directly
into a DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface this implements, and
src/data_refinement/deck_box/sts_gg/extraction_stage.py for the sibling
this was deliberately modeled on.

FRAGMENT SHAPE AND PARSING: the actual HTML-walking mechanics
(selectors, quantity/name splitting) live in
src/data_refinement/fabtcg_decklists/fragment_parsing.py, shared with
the pre-existing src/data_refinement/fabtcg_decklists/
decklist_cards_metric.py — see that module's docstring for the full
reasoning. This stage only supplies its own per-card resolution/failure
policy (substitute the Unknown sentinel, never drop a slot — see
UNRESOLVED CARDS below).

CARD RESOLUTION: cards are looked up via
card_lookup.get_by_name_single(GameId.FLESH_AND_BLOOD, name,
strict=False) — same non-strict lookup and reasoning as
DecklistCardsMetric (a global name collision across the whole Flesh
and Blood card pool can't be ruled out; see that module's CARD
RESOLUTION AND FAILURE POLICY section).

UNRESOLVED CARDS: unlike DecklistCardsMetric (which drops an
unresolved card name from that deck's list and logs it), this stage
substitutes the Unknown sentinel card's nocab_uuid instead (see
CardBinder.ensure_unknown_card()) — the deck's card count stays
correct (a real copy occupied a real deck slot), and the substitution
is still logged loudly via the stdlib `logging` module.
PRECONDITION: card_lookup must already have this game's Unknown
sentinel card seeded (CardBinder.ensure_unknown_card(GameId.FLESH_AND_BLOOD),
called once against a real CardBinder before any extract() call) — this
stage stays read-only (CardLookup, never a full CardBinder) and does
NOT create that card lazily; a missing sentinel is a setup bug, so it
raises RuntimeError rather than silently dropping the card.

A fragment resolving to literally zero card-item elements (as opposed
to zero RESOLVED cards, which is now impossible — see UNRESOLVED CARDS
above) still raises ValueError — that's not an unresolved-card
situation, it means fabtcg.com's markup changed underneath this parser
and nothing was found to even attempt resolving.

IDEMPOTENT RE-RUNS: unlike a typical DeckExtractionStage (see
extraction.py's module docstring), this one calls box.update() on a
re-seen deck rather than duplicating it — made possible by deriving a
deck's nocab_uuid deterministically from fabtcg_decklists' own deck
slug (uuid5(_DECK_NAMESPACE, deck_slug)), rather than minting a fresh
uuid4 per extract() call.

DELIBERATELY DROPPED: which group (Hero/Weapon/Equipment vs. each
Pitch N) each card came from is discarded here, same as
DecklistCardsMetric's FIRST ITERATION FLATTENS GROUPS — GenericDeck has
no per-slot structure field (see
src/data_refinement/deck_box/README.md's multiset/no-metadata
section).
"""

import logging
from collections import Counter
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid5

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.fabtcg_decklists.fragment_parsing import (
    iter_card_quantities_and_names,
)
from src.data_retrieval.fabtcg_decklists.downloader import FabtcgDecklistDownloader
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

# Fixed, arbitrary — never regenerate. Namespace for this stage's
# deterministic per-deck deck uuids (see module docstring's IDEMPOTENT
# RE-RUNS section).
_DECK_NAMESPACE = UUID("6c6a1c1e-3e3b-4a3a-9f0b-2b7f8a2f1e4d")


class FabtcgDecklistsExtractionStage:
    """Translates fabtcg_decklists' saved HTML fragments directly into
    a DeckBox.

    Single-consumer to src/data_refinement/deck_box/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        FabtcgDecklistDownloader.DEFAULT_RAW_DATA_DIR / "decklists"
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Parse every fabtcg_decklists fragment under raw_path,
        creating/updating decks on box.

        One _extract_deck() call per `*.html` fragment file, in a
        stable (sorted) order — no additional logic beyond calling that
        and collecting the non-None results.

        Inputs:
            raw_path: a directory of `<slug>.html` decklist fragment
                files (e.g. data/raw/fabtcg_decklists/decklists/), each
                file's stem used as that deck's deck_slug. Defaults to
                DEFAULT_RAW_PATH when None.
            box: the DeckBox to create/update decks on, as a side
                effect. Every deck this call touches is stored under
                self.SOURCE_GAME.
            card_lookup: must already have self.SOURCE_GAME's Unknown
                sentinel card seeded (see module docstring's
                PRECONDITION) in addition to cardvault_fabtcg's cards.
        Output: nocab_uuid of every deck slug whose resolved card list
            this call created or changed. A re-seen slug whose resolved
            list is identical to what's already stored is NOT included.
        Side effects: reads every `*.html` file directly under
            raw_path; creates/updates decks directly on box; emits one
            logging.error() per card name that falls back to the
            Unknown sentinel.
        Exceptions: raises if raw_path doesn't exist. Whatever
            _extract_deck() raises (see its own Exceptions) propagates.

        Example:
            >>> binder = CardBinder.load(
            ...     [Path("data/final/cards/flesh_and_blood.jsonl")]
            ... )
            >>> box = DeckBox.load([Path("data/final/decks/flesh_and_blood.jsonl")])
            >>> stage = FabtcgDecklistsExtractionStage()
            >>> changed_uuids = stage.extract(None, box, binder)
        """
        path = raw_path or self.DEFAULT_RAW_PATH

        changed_uuids = []
        for fragment_path in sorted(path.glob("*.html")):
            deck_slug = fragment_path.stem
            fragment_html = fragment_path.read_text(encoding="utf-8")
            result = self._extract_deck(deck_slug, fragment_html, box, card_lookup)
            if result is not None:
                changed_uuids.append(result)
        return changed_uuids

    def _extract_deck(
        self, deck_slug: str, fragment_html: str, box: DeckBox, card_lookup: CardLookup
    ) -> UUID | None:
        """Create-or-update one fabtcg decklist fragment directly
        against box.

        Private helper — single consumer is extract(). THIS METHOD IS
        WHERE IDEMPOTENCY HAPPENS: deck_uuid is a pure function of
        deck_slug (see module docstring's IDEMPOTENT RE-RUNS section),
        so re-processing the same fragment always targets the same
        stored deck.

        Inputs:
            deck_slug: the deck's slug (the fragment file's stem).
            fragment_html: one deck's saved decklist fragment, exactly
                as src/data_retrieval/fabtcg_decklists/ wrote it to
                disk.
            box: the DeckBox to read from and write to.
            card_lookup: used to resolve each card name (see
                _card_uuids_in_fragment()).
        Output: deck_uuid if this call caused a create() or an actual
            content-changing update(); None if this slug matched an
            already-stored deck with an identical resolved card list.
        Side effects: creates or updates exactly one deck on box.
        Exceptions: whatever _card_uuids_in_fragment() raises
            propagates.
        """
        deck_uuid = self._deck_uuid(deck_slug)
        card_nocab_uuids = self._card_uuids_in_fragment(fragment_html, card_lookup)

        existing = box.get_by_uuid(deck_uuid)
        if existing is None:
            # Brand-new deck slug.
            box.create(
                GenericDeck(
                    nocab_uuid=deck_uuid,
                    source_game=self.SOURCE_GAME,
                    name=f"fabtcg decklist {deck_slug}",
                    card_nocab_uuids=card_nocab_uuids,
                )
            )
            return deck_uuid

        # card_nocab_uuids is a multiset (see src/schema/card.py) —
        # order is not guaranteed, so compare copy counts per uuid via
        # Counter, never by list equality (order-sensitive) or set
        # equality (loses duplicate/copy-count information).
        if Counter(existing.card_nocab_uuids) != Counter(card_nocab_uuids):
            # Re-seen slug whose resolved list changed — update in place.
            box.update(deck_uuid, card_nocab_uuids=card_nocab_uuids)
            return deck_uuid

        # Re-seen slug, identical resolved multiset — no-op.
        return None

    def _card_uuids_in_fragment(
        self, fragment_html: str, card_lookup: CardLookup
    ) -> list[UUID]:
        """Walk every card group in one deck fragment, resolving each
        card name to a nocab_uuid, duplicated per copy.

        Private helper — single consumer is _extract_deck(). See module
        docstring's FRAGMENT SHAPE AND PARSING section for exactly what
        selectors are walked and why. Unlike
        DecklistCardsMetric._extract_card_uuids() (which this mirrors),
        an unresolved card name is substituted with the Unknown
        sentinel's uuid rather than dropped (see module docstring's
        UNRESOLVED CARDS section) — this method never shrinks the
        returned list relative to the number of card-item elements
        found.

        Inputs:
            fragment_html: one deck's saved decklist fragment.
            card_lookup: used to resolve each card name, and (on a
                miss) to fetch the pre-seeded Unknown sentinel.
        Output: one nocab_uuid per copy of every card-item found in
            fragment_html, in document order.
        Side effects: emits one logging.error() per card name that
            falls back to the Unknown sentinel.
        Exceptions: raises ValueError if fragment_html contains zero
            card-item elements, or whatever
            iter_card_quantities_and_names() raises for a malformed
            fragment. Raises RuntimeError if a card name is unresolved
            and self.SOURCE_GAME's Unknown sentinel isn't found on
            card_lookup (see module docstring's PRECONDITION).
        """
        card_uuids: list[UUID] = [
            self._card_uuid(name, card_lookup)
            for quantity, name in iter_card_quantities_and_names(fragment_html)
            for _ in range(quantity)
        ]

        if not card_uuids:
            raise ValueError(
                "_card_uuids_in_fragment: fragment resolved zero card-item "
                "elements — likely a fabtcg.com markup change"
            )

        return card_uuids

    def _card_uuid(self, name: str, card_lookup: CardLookup) -> UUID:
        """Look up the nocab_uuid for one fabtcg decklist card name.

        Private helper — single consumer is _card_uuids_in_fragment().
        On a miss, logs loudly and falls back to the Unknown sentinel's
        uuid instead of dropping the card (see module docstring's
        UNRESOLVED CARDS section) — unlike
        DecklistCardsMetric._extract_card_uuids(), this method never
        excludes a card-item from the returned list.

        Inputs:
            name: one card-item's parsed card name, e.g. "Dorinthea
                Ironsong".
        Output: the matching nocab_uuid, or (on a miss) the Unknown
            sentinel's nocab_uuid.
        Side effects: emits one logging.error() call on a miss.
        Exceptions: raises RuntimeError if even the Unknown sentinel
            isn't found on card_lookup (see module docstring's
            PRECONDITION).
        """
        card = card_lookup.get_by_name_single(self.SOURCE_GAME, name, strict=False)
        if card is not None:
            return card.nocab_uuid

        _logger.error(
            "FabtcgDecklistsExtractionStage: unresolved card name %r — "
            "substituting the Unknown sentinel card",
            name,
        )
        unknown_card = card_lookup.get_by_name_single(
            self.SOURCE_GAME, CardBinder.UNKNOWN_CARD_NAME, strict=False
        )
        if unknown_card is None:
            raise RuntimeError(
                f"FabtcgDecklistsExtractionStage: {self.SOURCE_GAME!r}'s Unknown "
                "sentinel card is not seeded — call "
                "CardBinder.ensure_unknown_card() before extract()"
            )
        return unknown_card.nocab_uuid

    @staticmethod
    def _deck_uuid(deck_slug: str) -> UUID:
        """Compute the deterministic nocab_uuid for one fabtcg decklist
        slug.

        Private helper — single consumer is _extract_deck(). uuid5 (not
        uuid4): the same deck_slug must always produce the same uuid,
        across every extract() call, so a re-seen slug updates rather
        than duplicates (see module docstring's IDEMPOTENT RE-RUNS
        section).

        Inputs:
            deck_slug: one deck fragment file's stem.
        Output: a uuid unique to (this fixed namespace, deck_slug) —
            stable forever for a given deck_slug.
        Side effects: none.
        Exceptions: none.
        """
        return uuid5(_DECK_NAMESPACE, deck_slug)
