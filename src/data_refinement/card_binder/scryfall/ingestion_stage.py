"""Translates a Scryfall oracle-cards dump into stored cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and
src/data_refinement/card_binder/README.md for the full design. This class is handed a live
CardBinder and owns its own duplicate-detection and
collision-resolution directly against it, rather than returning
IngestedCandidate data for a driver (build.py) to interpret centrally.
All source-specific identity-extraction knowledge lives here — this is
the one place that knows which raw JSON fields on a Scryfall row are
card identifiers versus printing/artwork metadata.

Confirmed by sampling data/raw/scryfall/oracle-cards-20260820090157.jsonl
directly:
  - Identity-bearing (extracted as secondary aliases): arena_id,
    mtgo_id, mtgo_foil_id, multiverse_ids (a list — 0/1/2+ entries).
  - NOT identity (deliberately excluded): id (printing-specific, not
    oracle_id), set_id, card_back_id, illustration_id (artwork/
    printing metadata); tcgplayer_id/cardmarket_id (commerce product
    ids, no current consumer).
  - Not every row has every field — e.g. Arena-illegal cards have no
    arena_id.

Design (see card_binder/README.md):
  - DEDUPLICATION: this is entirely the get_by_alias() lookup at the
    top of _ingest_row() — a hit means "this row is a duplicate of an
    already-stored card," a miss means "this is a genuinely new card."
    There is no other duplicate-detection logic anywhere in this
    file; oracle_id, resolved via
    binder.get_by_alias(self.SOURCE_GAME, DataSource.SCRYFALL,
    oracle_id), is a perfect natural key, so no name-based or
    heuristic fallback is needed (contrast pokemon_tcg's stage, which
    needs one since it has no equivalent natural key).
  - COLLISION RESOLUTION: once a duplicate is found, the incoming row
    wins (merge_strategies.keep_incoming_if_content_differs) whenever its
    lean content differs from the stored card's: a Scryfall dump is the authority for
    its own oracle_ids, and a byte-length richness comparison would be
    decided by URLs and prices, not card text. Identical content is left
    untouched, so re-ingesting the same dump reports no changes.
  - LEAN CONTENT: raw_content is not the raw row. It keeps what
    describes the card as a game object and drops printing, commerce,
    image and API-envelope fields, all URLs/IDs/dates, other cards'
    names (all_parts) and empty values, collapses legalities to the
    non-default statuses, and puts short identifying keys first and rules
    text last (see _lean_card_content and "What goes in raw_content" in
    card_binder/README.md). Identity aliases are still read from the RAW
    row, so dropping ids from raw_content loses no alias. The raw dump
    stays on disk under data/raw/scryfall.
  - source_game is NOT a parameter anywhere in this class — a stage
    always ingests exactly one game, so it's a class constant
    (SOURCE_GAME) instead. Passing it as an argument would let a
    caller construct a nonsensical call like
    ScryfallCardIngestionStage().ingest(path, binder) for a
    different game's binder without any way to catch the mismatch;
    tying it to the class instead makes that a non-issue by
    construction.
  - RARITY: rarity is per printing, and an oracle-cards row carries one
    arbitrary printing's. When the stage is given a default-cards dump (every
    printing, see __init__), a card stores the LOWEST rarity across its
    printings (_RARITY_BY_RESTRICTIVENESS), the same rule as
    cardvault_fabtcg's stage. A card with no default-cards printing keeps
    its oracle-cards row's rarity.
  - PRINTED NAMES: Arena names some printings differently from Scryfall's
    oracle name (the OM1 set: Scryfall's "Masked Meower" is printed
    "Skittering Kitten"), and 17lands spells the Arena name. From the same
    default-cards scan, every English printing whose printed_name differs
    from its name registers a (PRINTED_NAME, printed_name) alias on its
    card, so a name lookup can fall back to it (see
    card_lookup.uuid_for_name_or_front_face). A printed name shared by two
    cards is ambiguous and registers nothing. _MANUAL_NAME_ALIASES covers
    spellings no dump carries.
  - Every row's own primary alias (oracle_id) and every secondary
    alias it carries are registered on EVERY branch — including a row
    that matched an existing card but changed nothing: a losing/no-op
    row's identifier must never become a dead end for get_by_alias().
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, ClassVar, Iterable
from uuid import UUID, uuid4

from tqdm import tqdm

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.lean_content import (
    JsonObject,
    order_keys,
    strip_noise,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_MULTIVERSE_IDS_FIELD = "multiverse_ids"
_SINGLE_VALUE_ALIAS_FIELDS = {
    "arena_id": DataSource.ARENA,
    "mtgo_id": DataSource.MTGO,
    "mtgo_foil_id": DataSource.MTGO,
}

# Keys dropped wherever they appear. The dump holds one arbitrary printing
# per oracle_id, so these describe that printing, its commerce or the API
# envelope, not the card's rules. Kept on purpose: set, rarity (replaced by
# the lowest printing rarity, see _lean_card_content; labels masked-field
# dojos), digital,
# reserved and game_changer (true-only flags, their presence is the signal),
# produced_mana. Every id/uri/url key and every URL/UUID/date value is
# already removed by strip_noise.
_PRINTING_AND_ENVELOPE_KEYS = frozenset(
    {
        "object",
        "lang",
        "all_parts",  # names of other cards: noise, and leaks held-out cards
        "artist",
        "border_color",
        "frame",
        "frame_effects",
        "security_stamp",
        "finishes",
        "games",
        "collector_number",
        "image_status",
        "prices",
        "edhrec_rank",
        "penny_rank",
        "promo_types",
        "preview",
        "watermark",
        "flavor_text",
        "set_name",  # would give away a masked "set"
        "set_type",
        "foil",
        "nonfoil",
        "reprint",
        "booster",
        "highres_image",
        "story_spotlight",
    }
)
_LEADING_KEYS = (
    "name",
    "mana_cost",
    "cmc",
    "type_line",
    "power",
    "toughness",
    "loyalty",
    "defense",
    "colors",
    "color_identity",
    "keywords",
    "layout",
    "rarity",
    "set",
)
_TRAILING_KEYS = ("oracle_text", "card_faces")

# Spellings found in 17lands data that no Scryfall dump carries: the CSV
# header has the mojibake "B?" for "Bō". Maps spelling -> the card's name.
_MANUAL_NAME_ALIASES = {"Bespoke B?": "Bespoke B\u014d"}


@dataclass(frozen=True)
class _PrintingScan:
    """What one pass over a default-cards dump yields.

    lowest_rarities: oracle_id -> lowest rarity over the card's printings.
    oracle_id_by_printed_name: printed name -> oracle_id, for names that
        differ from the card's name and belong to exactly one card.
    """

    lowest_rarities: dict[str, str]
    oracle_id_by_printed_name: dict[str, str]


# From least to most restrictive. A rarity not listed (a future one) ranks
# after every listed one.
_RARITY_BY_RESTRICTIVENESS = (
    "common",
    "uncommon",
    "rare",
    "mythic",
    "special",
    "bonus",
)
_NOT_LEGAL = "not_legal"

# Scryfall layouts that are not a playable card a deck can contain -
# reminder/accessory objects that happen to share the oracle-cards dump
# with real cards. Excluded outright (see ingest()) rather than ingested
# and left for a downstream consumer to filter, because several of them
# share a name with a real card (e.g. a "Tarmogoyf" token alongside the
# real Tarmogoyf creature card) - leaving both in the binder makes any
# name-based lookup for that real card ambiguous (2+ matches) and it
# silently falls back to the Unknown sentinel, exactly as if the real
# card didn't exist. Not excluded: every layout that IS a real playable
# card, however unusually formatted (transform, saga, split, adventure,
# modal_dfc, class, mutate, flip, leveler, meld, prepare, etc).
_EXCLUDED_LAYOUTS = frozenset(
    {
        "token",
        "double_faced_token",
        "emblem",
        "scheme",
        "planar",
        "vanguard",
        "art_series",
        "front_card",
    }
)


class ScryfallCardIngestionStage:
    """Translates a Scryfall oracle-cards .jsonl dump directly into binder.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly.

    Runtime: about 15 s for 34,665 cards: 6 s scanning the 634 MB default-cards
    dump, 5 s for the oracle-cards dump, plus the save (measured 2026-10-08).
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG

    def __init__(self, find_printings_path: Callable[[], Path] | None = None) -> None:
        """
        Inputs:
            find_printings_path: called at the start of ingest() to get the
                Scryfall default-cards .jsonl file (one row per printing)
                that supplies each card's lowest rarity. A callable, not a
                path, so a missing file fails when ingest runs rather than
                when the stage is constructed. None: every card keeps its
                oracle-cards row's rarity.
        Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        self._find_printings_path = find_printings_path

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse a Scryfall oracle-cards .jsonl file, creating/updating
        cards directly on binder as a side effect.

        One _ingest_row() call per line of raw_path whose "layout" is
        not in _EXCLUDED_LAYOUTS (see that constant's own comment) - a
        row with an excluded layout is skipped entirely: no create(),
        no update, no alias registered.

        Inputs:
            raw_path: path to a Scryfall oracle-cards .jsonl file (one
                JSON object per line, e.g.
                data/raw/scryfall/oracle-cards-<timestamp>.jsonl).
            binder: the CardBinder to create/update cards on and
                register aliases against, as a side effect. Expected
                to already be loaded (e.g. via CardBinder.load()) by
                the caller — this method never calls load()/save()
                itself. Every card this call touches is stored under
                self.SOURCE_GAME (GameId.MTG) — there is no way to
                call this method for a different game.
        Output: nocab_uuid of every row that caused a create() or an
            actual content-changing replace() this call. A row that
            matched an existing card but changed nothing (an
            exact-duplicate re-fetch) is NOT included — see
            _ingest_row()'s docstring.
        Side effects: reads raw_path; creates/updates cards and
            registers aliases directly on binder, once per line.
            Prints a tqdm progress bar to stderr, sized against
            raw_path's byte size (not its line count, which isn't
            known up front without a separate full read). When a
            default-cards file is configured, reads it once first.
        Exceptions: raises if raw_path doesn't exist, isn't valid
            JSONL, or a line is missing "oracle_id" or "name" (see
            _ingest_row()); whatever find_printings_path or reading the
            default-cards file raises.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> stage = ScryfallCardIngestionStage()
            >>> changed_uuids = stage.ingest(
            ...     Path("data/raw/scryfall/oracle-cards-20260820090157.jsonl"),
            ...     binder,
            ... )
        """
        changed_uuids = []
        scan = self._scan_printings()
        total_bytes = raw_path.stat().st_size
        with open(raw_path, "r", encoding="utf-8") as raw_file, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=f"scryfall ingest: {raw_path.name}",
        ) as progress:
            for line in raw_file:
                progress.update(len(line.encode("utf-8")))
                row = json.loads(line)
                if row.get("layout") in _EXCLUDED_LAYOUTS:
                    continue
                rarity = scan.lowest_rarities.get(row["oracle_id"], row.get("rarity"))
                result = self._ingest_row(row, binder, rarity)
                if result is not None:
                    changed_uuids.append(result)
        self._register_name_aliases(scan, binder)
        return changed_uuids

    def _scan_printings(self) -> _PrintingScan:
        """Read the default-cards file once for lowest rarities and printed names.

        Inputs: none (uses the configured find_printings_path).
        Output: a _PrintingScan; both dicts are empty if no default-cards
            file is configured. Rows without a top-level oracle_id (a few
            multi-face printings) are skipped.
        Side effects: reads the default-cards file once, with a progress bar.
        Exceptions: whatever find_printings_path or reading the file raises.
        """
        if self._find_printings_path is None:
            return _PrintingScan({}, {})
        rarities_by_card: dict[str, str] = {}
        oracle_ids_by_printed_name: dict[str, set[str]] = {}
        printings_path = self._find_printings_path()
        with open(printings_path, "r", encoding="utf-8") as printings_file, tqdm(
            total=printings_path.stat().st_size,
            unit="B",
            unit_scale=True,
            desc=f"scryfall printings: {printings_path.name}",
        ) as progress:
            for line in printings_file:
                progress.update(len(line.encode("utf-8")))
                row = json.loads(line)
                oracle_id = row.get("oracle_id")
                if oracle_id is None or row.get("layout") in _EXCLUDED_LAYOUTS:
                    continue
                known = rarities_by_card.get(oracle_id)
                rarities_by_card[oracle_id] = _least_restrictive(
                    [row["rarity"]] if known is None else [known, row["rarity"]]
                )
                printed_name = row.get("printed_name")
                if (
                    printed_name is not None
                    and printed_name != row.get("name")
                    and row.get("lang", "en") == "en"
                ):
                    oracle_ids_by_printed_name.setdefault(printed_name, set()).add(
                        oracle_id
                    )
        unambiguous = {
            printed_name: next(iter(oracle_ids))
            for printed_name, oracle_ids in oracle_ids_by_printed_name.items()
            if len(oracle_ids) == 1
        }
        return _PrintingScan(rarities_by_card, unambiguous)

    def _register_name_aliases(self, scan: _PrintingScan, binder: CardBinder) -> None:
        """Register every printed-name and manual name alias on its card.

        Inputs: scan (this ingest's printing scan), binder (holding the
            cards just ingested).
        Output: none.
        Side effects: registers (PRINTED_NAME, name) aliases on binder; a
            name whose card is not in the binder is skipped.
        Exceptions: none.
        """
        for printed_name, oracle_id in scan.oracle_id_by_printed_name.items():
            card = binder.get_by_alias(self.SOURCE_GAME, DataSource.SCRYFALL, oracle_id)
            if card is not None:
                binder.register_alias(
                    self.SOURCE_GAME,
                    DataSource.PRINTED_NAME,
                    printed_name,
                    card.nocab_uuid,
                )
        for spelling, card_name in _MANUAL_NAME_ALIASES.items():
            matches = binder.get_by_name(self.SOURCE_GAME, card_name)
            if len(matches) == 1:
                binder.register_alias(
                    self.SOURCE_GAME,
                    DataSource.PRINTED_NAME,
                    spelling,
                    matches[0].nocab_uuid,
                )

    def _ingest_row(
        self, row: dict, binder: CardBinder, rarity: str | None
    ) -> UUID | None:
        """Create-or-merge one Scryfall row directly against binder.

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
            binder: the CardBinder to read from and write to.
            rarity: the rarity to store (the card's lowest printing rarity),
                or None for a card with none.
        Output: the stored/canonical nocab_uuid for this row IF this
            call caused a create() or an actual content-changing
            replace(); None if this row matched an existing card but
            changed nothing.
        Side effects: creates or updates exactly one card on binder;
            registers one or more aliases on binder.
        Exceptions: raises if row is missing "oracle_id" or "name".
        """
        oracle_id = row["oracle_id"]
        existing = binder.get_by_alias(self.SOURCE_GAME, DataSource.SCRYFALL, oracle_id)

        if existing is None:
            # New card: build a fresh GenericCard and create it.
            card = self._build_card(row, oracle_id, rarity)
            binder.create(card)
            stored_uuid = card.nocab_uuid
            changed = True
        else:
            # Existing card: the incoming row wins if its content differs.
            candidate = self._build_card(row, oracle_id, rarity)
            merged = merge_strategies.keep_incoming_if_content_differs(
                existing, candidate
            )
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(
            self.SOURCE_GAME, DataSource.SCRYFALL, oracle_id, stored_uuid
        )
        for data_source, source_id in self._extract_aliases(row):
            binder.register_alias(self.SOURCE_GAME, data_source, source_id, stored_uuid)

        return stored_uuid if changed else None

    def _build_card(self, row: dict, oracle_id: str, rarity: str | None) -> GenericCard:
        """Build a fresh GenericCard for one Scryfall row.

        Private helper — single consumer is _ingest_row(), from both
        its create-path and its throwaway-candidate-for-merge path
        (see that method's docstring — the candidate's own nocab_uuid
        is never actually stored on the merge path, only ever
        existing's is).

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
            oracle_id: row["oracle_id"], passed in rather than
                re-read, since callers already have it at hand.
            rarity: the rarity to store, or None for none.
        Output: a new GenericCard with a freshly minted nocab_uuid and the
            row's lean content (see _lean_card_content) as raw_content.
        Side effects: none.
        Exceptions: raises if row is missing "name".
        """
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=row["name"],
            raw_content=_lean_card_content(row, rarity),
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id=oracle_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )

    def _extract_aliases(self, row: dict) -> list[tuple[DataSource, str]]:
        """Extract every known secondary identifier present on a Scryfall row.

        Private helper — single consumer is _ingest_row(). Checks a
        fixed allowlist (arena_id, mtgo_id, mtgo_foil_id, each entry
        of multiverse_ids — see this module's docstring for why these
        four and not others) and tolerates any of them being absent —
        this is NOT the same as _ingest_row()'s required oracle_id;
        every entry here is optional per row.

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
        Output: one (DataSource, source_id) tuple per identity-bearing
            field actually present on row, in no particular required
            order. Empty list if row has none of the allowlisted
            fields.
        Side effects: none.
        Exceptions: none.
        """
        aliases = []

        for field_name, data_source in _SINGLE_VALUE_ALIAS_FIELDS.items():
            if field_name in row:
                aliases.append((data_source, str(row[field_name])))

        for multiverse_id in row.get(_MULTIVERSE_IDS_FIELD, []):
            aliases.append((DataSource.GATHERER, str(multiverse_id)))

        return aliases


def _least_restrictive(rarities: Iterable[str]) -> str:
    """The rarity that is least restrictive (see _RARITY_BY_RESTRICTIVENESS).

    Unlisted rarities rank after every listed one, ties between them break
    alphabetically, so the result never depends on input order.

    Inputs: rarities (Iterable[str]): at least one rarity.
    Output: str. Side effects: none.
    Exceptions: ValueError if rarities is empty.

    Example:
        >>> _least_restrictive(["mythic", "uncommon", "special"])
        'uncommon'
    """

    def restrictiveness(rarity: str) -> tuple[int, str]:
        if rarity in _RARITY_BY_RESTRICTIVENESS:
            return _RARITY_BY_RESTRICTIVENESS.index(rarity), rarity
        return len(_RARITY_BY_RESTRICTIVENESS), rarity

    return min(rarities, key=restrictiveness)


def _lean_card_content(row: JsonObject, rarity: str | None) -> JsonObject:
    """Reduce a Scryfall oracle-cards row to what describes the card.

    Applies lean_content.strip_noise with this source's printing/envelope
    keys, collapses legalities, and orders keys (identity and short fields
    first, oracle_text and card_faces last), including inside each face.

    Inputs: row (JsonObject): one parsed Scryfall oracle-cards line.
        rarity (str | None): the rarity to store in place of the row's own
        (None: store none).
    Output: a new JsonObject; row itself is not modified.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> _lean_card_content(
        ...     {"name": "Bolt", "id": "x", "cmc": 1.0, "rarity": "rare"}, "common"
        ... )
        {'name': 'Bolt', 'cmc': 1, 'rarity': 'common'}
    """
    content = strip_noise(row, extra_noise_keys=_PRINTING_AND_ENVELOPE_KEYS)
    content.pop("rarity", None)
    if rarity:
        content["rarity"] = rarity
    legalities = content.get("legalities")
    if isinstance(legalities, dict):
        legal_formats = _legal_formats_by_status(legalities)
        if legal_formats:
            content["legalities"] = legal_formats
        else:
            del content["legalities"]
    faces = content.get("card_faces")
    if isinstance(faces, list):
        content["card_faces"] = [
            (
                order_keys(face, _LEADING_KEYS, _TRAILING_KEYS)
                if isinstance(face, dict)
                else face
            )
            for face in faces
        ]
    return order_keys(content, _LEADING_KEYS, _TRAILING_KEYS)


def _legal_formats_by_status(legalities: dict) -> JsonObject:
    """Invert {format: status} into {status: [formats]}, omitting not_legal.

    Inputs: legalities (dict[str, str]): Scryfall's legalities object.
    Output: JsonObject mapping status to a list of format names, e.g.
        {"legal": ["modern"], "banned": ["legacy"]}; empty if nothing is
        legal, banned or restricted.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> _legal_formats_by_status({"modern": "legal", "pauper": "not_legal"})
        {'legal': ['modern']}
    """
    formats_by_status: dict[str, list[str]] = {}
    for format_name, status in legalities.items():
        if status != _NOT_LEGAL:
            formats_by_status.setdefault(status, []).append(format_name)
    return {status: list(formats) for status, formats in formats_by_status.items()}
