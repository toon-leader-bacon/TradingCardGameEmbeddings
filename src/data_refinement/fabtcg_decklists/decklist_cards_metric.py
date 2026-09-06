"""Converts a fabtcg.com decklist HTML fragment into a deck-level
card-uuid metric.

Structurally a sibling to src/data_refinement/sts_gg/
deck_outcome_metric.py (DeckOutcomeMetric + scan_runs_jsonl) — NOT a
card_binder/<source>/ingestion_stage.py. This stage never creates or
replaces a GenericCard; it only resolves already-ingested cards (from
cardvault_fabtcg's ingestion stage) to nocab_uuids. See
plans/fabtcg_decklists_refinement.md for the full design this
implements.

FRAGMENT SHAPE (confirmed live against data/tmp/list_view.html):
src/data_retrieval/fabtcg_decklists/ saves one prettified
`<section class="decklist-list-view block hidden">` fragment per deck.
Every group inside it — the combined "Hero / Weapon / Equipment" group,
and each of "Pitch 1"/"Pitch 2"/"Pitch 3" — is a uniformly-shaped
`<div class="list-view-container">`, so a single
`soup.select("div.list-view-container")` finds all of them regardless
of the pitch groups' extra `<section class="pitch-section">` wrapper —
no special-casing needed. Each `<li class="card-item group">` holds
`<div class="card-name"><span>1x</span> Card Name</div>` — quantity and
name are not separate elements, so extraction reads
`" ".join(name_div.get_text().split())` ("1x Card Name") and splits it
with _QUANTITY_AND_NAME_PATTERN. The split()/join() (rather than
get_text(" ", strip=True)) is deliberate: some real card-name divs wrap
their text across an internal newline as one text node — get_text's
separator only joins BETWEEN separate text nodes, so it would never
collapse an internal one, and a long name plus a trailing pitch-color
marker on its own line would otherwise fail to match
_QUANTITY_AND_NAME_PATTERN.

FIRST ITERATION FLATTENS GROUPS: which group each card came from is
discarded here — see TODO.md for the deferred structured/grouped
follow-up and why that's a deliberate simplification, not an oversight.

CARD RESOLUTION AND FAILURE POLICY: cards are looked up via
CardBinder.get_by_name_single(..., strict=False) rather than the
strict=True default — cardvault_fabtcg/ingestion_stage.py documents
that a card's face_1_true_name is unique per its own card_id, but says
nothing about uniqueness ACROSS different card_ids, so a global name
collision across the whole Flesh and Blood card pool can't be ruled
out; strict=False accepts that rare ambiguity rather than letting one
collision raise and lose an entire deck. A single card name that
doesn't resolve at all via CardBinder.get_by_name_single is dropped and
logged
(matching DeckOutcomeMetric's per-card best-effort policy) — one bad
name shouldn't lose an otherwise-good deck. A deck resolving to ZERO
cards is treated differently: that almost certainly means the binder
wasn't populated with cardvault_fabtcg's cards, or fabtcg.com's markup
changed underneath this parser, so accumulate() raises instead of
silently emitting an empty-list row a downstream reader could mistake
for "this deck genuinely has no cards."
"""

import logging
import re
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_LIST_VIEW_CONTAINER_SELECTOR = "div.list-view-container"
_CARD_ITEM_SELECTOR = "li.card-item"
_CARD_NAME_SELECTOR = "div.card-name"
_QUANTITY_AND_NAME_PATTERN = re.compile(r"^(\d+)x\s+(.+)$")

# Fixed and known upfront, same convention as DeckOutcomeMetric's
# _OUTPUT_SCHEMA.
_OUTPUT_SCHEMA = pa.schema(
    [
        ("deck_slug", pa.string()),
        ("card_nocab_uuids", pa.list_(pa.string())),
    ]
)


class DecklistCardsMetric:
    """Streaming per-deck metric: one fabtcg.com decklist fragment in,
    one (deck_slug, card_nocab_uuids) row out, written immediately.

    Single-consumer to src/data_refinement/fabtcg_decklists/ — no other
    container depends on this class.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("data/final/decks/fab/decklists.parquet")

    def __init__(
        self, card_binder: CardBinder, output_path: Path | None = None
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve card names against — must
                already have cardvault_fabtcg's cards ingested (this
                class never writes to it).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file at that path) via a pyarrow.parquet.
            ParquetWriter held open for the lifetime of this instance —
            callers MUST call finalize() when done, or the file is left
            incomplete/unreadable. See scan_decklists_dir for this
            project's one call site, which guarantees finalize() runs
            (via try/finally) even if a deck fails partway through a
            scan.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
        """
        self._card_binder = card_binder
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, _OUTPUT_SCHEMA)
        self._closed = False

    def accumulate(self, deck_slug: str, fragment_html: str) -> None:
        """Convert one deck's HTML fragment into a single output row
        and write it immediately.

        Streaming, not accumulating (see module docstring): this row's
        output depends on nothing else this instance has seen or will
        see, so there's no reason to hold it in memory past this call.

        Inputs:
            deck_slug: the deck's slug, e.g. the stem of its saved
                fragment filename ("tom-penny-go-wide-warrior-deck") —
                the only deck-name signal available from a saved
                fragment (the fragment itself carries no deck title).
            fragment_html: one deck's saved
                `<section class="decklist-list-view block hidden">`
                fragment, exactly as src/data_retrieval/
                fabtcg_decklists/ wrote it to disk.
        Output: none.
        Side effects: writes exactly one row to the open
            ParquetWriter — visible in output_path once finalize() has
            flushed/closed the writer, not necessarily before. Emits
            one logging.warning() per card name in this deck that
            fails to resolve (see module docstring's CARD RESOLUTION
            AND FAILURE POLICY).
        Exceptions: raises ValueError if fragment_html resolves to zero
            cards total (see module docstring) or if a card-name span's
            text doesn't match _QUANTITY_AND_NAME_PATTERN (a genuine
            site-format change should be loud, not silently produce a
            wrong count).

        Example:
            >>> metric = DecklistCardsMetric(card_binder)
            >>> metric.accumulate("tom-penny-go-wide-warrior-deck", fragment_html)
            >>> metric.finalize()
        """
        # Parse every card group in this fragment into a flat list of
        # resolved nocab_uuids (duplicated per copy) — see
        # _extract_card_uuids for the per-group/per-card walk.
        card_nocab_uuids = self._extract_card_uuids(fragment_html)

        if not card_nocab_uuids:
            raise ValueError(
                f"accumulate: deck {deck_slug!r} resolved zero cards — likely "
                "an unpopulated CardBinder or a fabtcg.com markup change"
            )

        # Build this deck's one-row Table against _OUTPUT_SCHEMA and
        # write it immediately.
        output_row = pa.Table.from_pydict(
            {
                "deck_slug": [deck_slug],
                "card_nocab_uuids": [[str(uuid) for uuid in card_nocab_uuids]],
            },
            schema=_OUTPUT_SCHEMA,
        )
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        A true no-op relative to data — every row this instance will
        ever write was already written by accumulate() (see module
        docstring). This only flushes/closes the file handle.

        Idempotent: a second call is a no-op rather than an error — see
        scan_decklists_dir's try/finally, which calls this
        unconditionally even after an accumulate() failure, so a caller
        that also calls finalize() itself (per this method's own
        contract) shouldn't be punished for a writer that's already
        closed.

        Inputs: none.
        Output: output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/final/decks/fab/decklists.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _extract_card_uuids(self, fragment_html: str) -> list[UUID]:
        """Walk every card group in one deck fragment, resolving each
        card name to a nocab_uuid, duplicated per copy.

        Private helper — single consumer is accumulate(). See module
        docstring's FRAGMENT SHAPE and FIRST ITERATION FLATTENS GROUPS
        sections for exactly what's walked and what's deliberately
        discarded (group membership).

        Inputs:
            fragment_html: one deck's saved decklist fragment.
        Output: one nocab_uuid per copy of every resolvable card in
            fragment_html, in document order. A card name that doesn't
            resolve is excluded (logged, not raised) — see module
            docstring. Can be empty if nothing in the fragment
            resolves.
        Side effects: emits one logging.warning() per unresolved card
            name.
        Exceptions: raises ValueError if a card-item's name text
            doesn't match _QUANTITY_AND_NAME_PATTERN.
        """
        soup = BeautifulSoup(fragment_html, "html.parser")
        card_uuids: list[UUID] = []

        # Walk every group container (Hero/Weapon/Equipment, and each
        # Pitch N) uniformly — see module docstring's FRAGMENT SHAPE.
        for container in soup.select(_LIST_VIEW_CONTAINER_SELECTOR):
            for card_item in container.select(_CARD_ITEM_SELECTOR):
                name_div = card_item.select_one(_CARD_NAME_SELECTOR)
                if name_div is None:
                    raise ValueError(
                        f"_extract_card_uuids: a {_CARD_ITEM_SELECTOR!r} element "
                        f"has no {_CARD_NAME_SELECTOR!r} child — fabtcg.com's "
                        "markup may have changed"
                    )

                # " ".join(...split()) rather than get_text(" ",
                # strip=True): some real card-name divs wrap their text
                # across an internal newline (e.g. a long name plus a
                # trailing "(red)" pitch marker on its own line) as ONE
                # text node — get_text's separator only joins BETWEEN
                # separate text nodes, so it never collapses that
                # internal newline, and _QUANTITY_AND_NAME_PATTERN
                # (whose `.` doesn't match newlines) would otherwise
                # raise on a perfectly valid card. split()/join()
                # normalizes every run of whitespace uniformly.
                quantity, name = self._parse_quantity_and_name(
                    " ".join(name_div.get_text().split())
                )

                # Look up this card name against the binder
                card = self._card_binder.get_by_name_single(
                    GameId.FLESH_AND_BLOOD, name, strict=False
                )
                if card is None:
                    _logger.warning(
                        "DecklistCardsMetric: card name %r did not resolve "
                        "against the CardBinder — excluding it from this "
                        "deck's card_nocab_uuids",
                        name,
                    )
                    continue

                card_uuids.extend([card.nocab_uuid] * quantity)

        return card_uuids

    def _parse_quantity_and_name(self, name_div_text: str) -> tuple[int, str]:
        """Split one card-name div's flattened text into its quantity
        and card name.

        Private helper — single consumer is _extract_card_uuids().
        E.g. "1x Dorinthea Ironsong" -> (1, "Dorinthea Ironsong").

        Inputs:
            name_div_text: a `div.card-name` element's text, with every
                run of whitespace (including any internal newline)
                already collapsed to a single space (see
                _extract_card_uuids' call site).
        Output: (quantity, name).
        Side effects: none.
        Exceptions: raises ValueError if name_div_text doesn't match
            _QUANTITY_AND_NAME_PATTERN.
        """
        match = _QUANTITY_AND_NAME_PATTERN.match(name_div_text)
        if match is None:
            raise ValueError(
                f"_parse_quantity_and_name: {name_div_text!r} does not match "
                f"the expected {_QUANTITY_AND_NAME_PATTERN.pattern!r} shape"
            )

        quantity, name = match.groups()
        return int(quantity), name


def scan_decklists_dir(decklists_dir: Path, metric: DecklistCardsMetric) -> Path:
    """Drive metric.accumulate() over every deck fragment in a
    decklists directory, then finalize it.

    Kept separate from DecklistCardsMetric itself: "how to walk
    fabtcg_decklists' saved fragments" is a different concern from
    "what this metric computes" (see DeckOutcomeMetric/
    scan_runs_jsonl's identical split, which this mirrors). Not yet
    promoted to a shared helper — single consumer today, same
    rule-of-three convention this project already applies elsewhere.

    Inputs:
        decklists_dir: directory of `<slug>.html` fragment files (e.g.
            data/raw/fabtcg_decklists/decklists/) — each file's stem is
            used as that deck's deck_slug.
        metric: the DecklistCardsMetric instance to drive. Its
            finalize() is called by this function (in a finally block
            — see below) — callers should not call it again themselves
            (finalize() is idempotent, so a redundant call is harmless,
            just unnecessary).
    Output: metric.finalize()'s return value (the output path).
    Side effects: reads every `*.html` file directly under
        decklists_dir; whatever metric.accumulate()/finalize() do (see
        those methods).
    Exceptions: raises if decklists_dir doesn't exist. Whatever
        metric.accumulate() raises for a malformed/unresolvable deck
        propagates — this function has no per-file best-effort skip of
        its own — but metric.finalize() still runs first (via
        try/finally), so the underlying ParquetWriter is always closed
        rather than left open on a mid-scan failure, even though the
        resulting file is then missing every deck from the failure
        point onward.

    Example:
        >>> card_binder = CardBinder.load(
        ...     [Path("data/final/cards/flesh_and_blood.jsonl")]
        ... )
        >>> metric = DecklistCardsMetric(card_binder)
        >>> scan_decklists_dir(Path("data/raw/fabtcg_decklists/decklists"), metric)
        PosixPath('data/final/decks/fab/decklists.parquet')
    """
    try:
        # Process each fragment file, in a stable (sorted) order.
        for fragment_path in sorted(decklists_dir.glob("*.html")):
            deck_slug = fragment_path.stem
            fragment_html = fragment_path.read_text(encoding="utf-8")
            metric.accumulate(deck_slug, fragment_html)
    finally:
        output_path = metric.finalize()

    return output_path
