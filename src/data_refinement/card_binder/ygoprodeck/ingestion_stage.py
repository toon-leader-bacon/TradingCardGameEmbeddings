"""Translates YGOPRODeck's card table (feregrino/yugioh-cards) into the
Yu-Gi-Oh binder: the Yu-Gi-Oh card source.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface, and card_binder/pool_mirror.py for the
mirror-the-whole-pool steps. The data:
src/data_retrieval/huggingface_datasets/ygoprodeck_repos.py
(feregrino/yugioh-cards, rebuilt weekly from YGOPRODeck's API,
CC-BY-SA-3.0).

RAW SHAPE: raw_path is the yugioh-cards mirror directory, complete
(HuggingFaceDatasetDownloader.is_complete_mirror), holding:
    cards.csv: one row per card, keyed by "id" (the passcode, or an id
        YGOPRODeck minted for a card without one, > 99,999,999). Numbers
        are written as floats ("2500.0"); an ATK/DEF of "?" is "-1.0";
        linkmarkers is a Python list literal ("['Top', 'Bottom-Left']").
    cards_variants.csv: original,variant: alternate-artwork passcodes,
        which decks use but cards.csv doesn't list.

WHICH ROWS ARE CARDS: every row but a Token (made mid-game) or a Skill
Card (Speed Duel; it sits outside the deck): user, 2026-11-07. TCG,
OCG-only and the few undated rows are all kept.

IDENTITY: the row. cards.csv already holds one row per card (14,653
rows, 14,653 ids, 14,653 names on 2026-11-02), so nothing is grouped.
The passcode is the card's DataSource.YGOPRODECK alias, spelled by
passcode.py's passcode_id() (the deck stage, deck_box/ygoprodeck_decks/,
spells it the same way). Each alternate-artwork variant is aliased to
its original's card, except: a variant whose original is not a kept row
(a Token's, or one cards.csv lacks) is unused, and a variant that is
also some row's own id is skipped, so a row's own passcode always names
that row's card.

The table is the whole card pool, so the binder mirrors it: a row
reuses the stored card its passcode names (unless an earlier row of
this ingest took it), and after every row is stored, Yu-Gi-Oh cards no
row took are deleted (never the Unknown sentinel). A table with no card
rows raises instead.

The Yu-Gi-Oh binder is built fresh from this source (re-ingest, don't
migrate): run it on an empty binder, not on one tcgjson built, whose
cards carry tcgjson aliases and name-matched passcodes.

Collision policy: merge_strategies.keep_incoming_if_content_differs.

LEAN CONTENT (_lean_card_content), keys in this order:
    name, type ("XYZ Monster", "Spell Card": YGOPRODeck's frame type),
    attribute, monster_type (a monster's race, "Warrior"), property (a
    Spell/Trap's race, "Quick-Play"), level, rank (an Xyz monster's
    level column), link_rating (linkval), link_arrows (linkmarkers, as a
    list), scale, atk, def, text (desc; a Pendulum's "[ Pendulum Effect
    ]"/"[ Monster Effect ]" sections kept).
A row is a monster when its type contains "Monster". Numbers become
ints; ATK/DEF -1 becomes "?". A Link monster's level (always 0) is
dropped: it has none. Dropped: archetype (a YGOPRODeck tag, not
printed: user, 2026-11-07), set and rarity (per printing, in
cards_cardsets.csv; a metric's job), treated_as (restates the text),
has_effect (restates type), image URLs, ban lists, formats, staple,
views/votes, release dates, konami_id.
"""

import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Iterator
from uuid import UUID, uuid4

from tqdm import tqdm

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.lean_content import order_keys, strip_noise
from src.data_refinement.card_binder.pool_mirror import (
    delete_cards_except,
    store_mirrored_card,
)
from src.data_refinement.card_binder.ygoprodeck.passcode import passcode_id
from src.data_retrieval.huggingface_datasets.downloader import (
    HuggingFaceDatasetDownloader,
)
from src.data_retrieval.huggingface_datasets.ygoprodeck_repos import (
    YGOPRODECK_CARDS_REPO_ID,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.json_types import JsonObject, JsonValue

_CARDS_FILENAME = "cards.csv"
_VARIANTS_FILENAME = "cards_variants.csv"

# The cards.csv columns read
_CARD_COLUMNS = (
    "id", "name", "type", "desc", "atk", "def", "level", "race",
    "attribute", "scale", "linkval", "linkmarkers",
)  # fmt: skip
_VARIANT_COLUMNS = ("original", "variant")

# cards.csv "type" values that are not deck cards (see the module docstring)
_EXCLUDED_TYPES = frozenset({"Token", "Skill Card"})
_MONSTER_MARKER = "Monster"
_XYZ_MARKER = "XYZ"
_LINK_MARKER = "Link"

# Kept keys, in serialization order (text is appended last)
_LEADING_KEYS = (
    "name",
    "type",
    "attribute",
    "monster_type",
    "property",
    "level",
    "rank",
    "link_rating",
    "link_arrows",
    "scale",
    "atk",
    "def",
)
_TEXT_KEY = "text"

# How cards.csv writes an ATK/DEF of "?"
_UNKNOWN_STAT = -1
_UNKNOWN_STAT_TEXT = "?"


@dataclass(frozen=True)
class YgoprodeckCard:
    """One card row of cards.csv, parsed: its passcode (an alias id),
    its name, and its lean raw_content."""

    passcode: str
    name: str
    content: JsonObject


class YgoprodeckCardIngestionStage:
    """YGOPRODeck's cards.csv -> the Yu-Gi-Oh binder, mirrored: one card
    per row, passcodes (and their alternate artworks) as aliases.

    Single-consumer to src/data_refinement/card_binder/.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.YUGIOH
    DEFAULT_RAW_PATH: ClassVar[Path] = HuggingFaceDatasetDownloader.raw_data_dir_for(
        YGOPRODECK_CARDS_REPO_ID
    )

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Make binder's Yu-Gi-Oh cards mirror cards.csv's deck cards.

        Inputs: raw_path (Path, the yugioh-cards mirror directory),
            binder (CardBinder, written as self.SOURCE_GAME).
        Output: nocab_uuid of every card created or content-changed
            (deleted cards are not listed).
        Side effects: reads cards.csv and cards_variants.csv; creates,
            replaces and deletes Yu-Gi-Oh cards; registers passcode
            aliases; prints a tqdm progress bar.
        Exceptions: FileNotFoundError if raw_path isn't a complete
            mirror or a file is missing; ValueError if a header lacks a
            needed column, an id isn't a passcode, a cell isn't the
            expected shape, or no row is a card.

        Example:
            >>> YgoprodeckCardIngestionStage().ingest(
            ...     YgoprodeckCardIngestionStage.DEFAULT_RAW_PATH,
            ...     CardBinder.load([]))[:1]
            [UUID('...')]
        """
        result: list[UUID] = []

        # Validate inputs and read the deck cards and their variants
        HuggingFaceDatasetDownloader.require_complete_mirror(raw_path)
        cards = list(_read_cards(raw_path / _CARDS_FILENAME))
        if not cards:
            raise ValueError(f"{raw_path / _CARDS_FILENAME} has no deck cards")
        variants = _variants_by_original(
            raw_path / _VARIANTS_FILENAME, {card.passcode for card in cards}
        )

        # Store every card, each on a stored card no earlier row took
        taken: set[UUID] = set()
        for card in tqdm(cards, desc="ygoprodeck yugioh", unit="card"):
            stored_uuid, changed = self._store_card(
                card, variants.get(card.passcode, ()), binder, taken
            )
            taken.add(stored_uuid)
            if changed:
                result.append(stored_uuid)

        # Drop the cards the table no longer has
        delete_cards_except(binder, self.SOURCE_GAME, taken)
        return result

    def _store_card(
        self,
        card: YgoprodeckCard,
        variant_passcodes: tuple[str, ...],
        binder: CardBinder,
        taken: set[UUID],
    ) -> tuple[UUID, bool]:
        """Create-or-merge one card (store_mirrored_card); register its
        passcode and variant_passcodes as YGOPRODECK aliases of it.

        Inputs: card (YgoprodeckCard), variant_passcodes (tuple[str,
            ...], its alternate artworks' alias ids), binder
            (CardBinder), taken (set[UUID], uuids earlier rows stored
            into; read only).
        Output: (stored nocab_uuid, whether this call created or changed
            the card).
        Side effects: creates or replaces one card; registers aliases.
        Exceptions: none beyond CardBinder's own.
        """
        existing = self._existing_card(card, binder, taken)
        stored_uuid, changed = store_mirrored_card(
            binder, existing, self._build_card(card)
        )
        for passcode in (card.passcode, *variant_passcodes):
            binder.register_alias(
                self.SOURCE_GAME, DataSource.YGOPRODECK, passcode, stored_uuid
            )
        return stored_uuid, changed

    def _existing_card(
        self, card: YgoprodeckCard, binder: CardBinder, taken: set[UUID]
    ) -> GenericCard | None:
        """The stored card whose own passcode is card's, unless an
        earlier row of this ingest took it. A passcode that aliases a
        card only as its alternate artwork (its provenance source_id is
        another passcode) names no card here, so a former variant listed
        as its own row never takes its old original's uuid.

        Inputs: card (YgoprodeckCard), binder (CardBinder), taken
            (set[UUID]).
        Output: GenericCard | None (None: create a new card).
        Side effects: none. Exceptions: none.
        """
        stored = binder.get_by_alias(
            self.SOURCE_GAME, DataSource.YGOPRODECK, card.passcode
        )
        if stored is None or stored.nocab_uuid in taken:
            return None
        if stored.provenance.source_id != card.passcode:
            return None
        return stored

    def _build_card(self, card: YgoprodeckCard) -> GenericCard:
        """A fresh GenericCard for card: a new nocab_uuid, provenance
        DataSource.YGOPRODECK with source_id the passcode.

        Inputs: card (YgoprodeckCard). Output: GenericCard.
        Side effects: none. Exceptions: none.
        """
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=card.name,
            raw_content=card.content,
            provenance=Provenance(
                data_source=DataSource.YGOPRODECK,
                source_id=card.passcode,
                fetched_at=datetime.now(timezone.utc),
            ),
        )


def _read_cards(cards_path: Path) -> Iterator[YgoprodeckCard]:
    """Every deck-card row of cards.csv (type not in _EXCLUDED_TYPES),
    parsed, in file order.

    Inputs: cards_path (Path).
    Output: YgoprodeckCard per deck-card row, lazily.
    Side effects: reads cards_path.
    Exceptions: FileNotFoundError; ValueError if the header lacks a
        _CARD_COLUMNS column, an id isn't a passcode, or a cell isn't
        the expected shape.

    Example:
        >>> next(_read_cards(Path(".../cards.csv"))).passcode
        '80181649'
    """
    for row in _csv_rows(cards_path, _CARD_COLUMNS):
        if row["type"].strip() in _EXCLUDED_TYPES:
            continue
        try:
            content = _lean_card_content(row)
            passcode = passcode_id(row["id"])
        except ValueError as error:
            raise ValueError(f"{cards_path.name} id {row['id']!r}: {error}") from error
        yield YgoprodeckCard(
            passcode=passcode, name=str(content["name"]), content=content
        )


def _variants_by_original(
    variants_path: Path, card_passcodes: set[str]
) -> dict[str, tuple[str, ...]]:
    """cards_variants.csv as original passcode -> its variant passcodes
    (alias ids, file order), for originals in card_passcodes only,
    leaving out any variant in card_passcodes (a row's own id; see the
    module docstring's IDENTITY). A variant listed under two originals
    stays with the first.

    Inputs: variants_path (Path), card_passcodes (set[str], every kept
        row's alias id).
    Output: dict[str, tuple[str, ...]]. Side effects: reads the file.
    Exceptions: FileNotFoundError; ValueError if an id isn't a passcode
        or the header lacks original or variant.

    Example:
        >>> _variants_by_original(Path(".../cards_variants.csv"), {"1546123"})
        {'1546123': ('1546124',)}
    """
    result: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()
    for row in _csv_rows(variants_path, _VARIANT_COLUMNS):
        original, variant = passcode_id(row["original"]), passcode_id(row["variant"])
        if original not in card_passcodes or variant in card_passcodes:
            continue
        if variant in assigned:
            continue
        assigned.add(variant)
        result[original] = (*result.get(original, ()), variant)
    return result


def _csv_rows(
    path: Path, required_columns: tuple[str, ...]
) -> Iterator[dict[str, str]]:
    """path's rows as {column: cell} for required_columns, after
    checking the header has them; a short row's missing cells read as "".

    Inputs: path (Path), required_columns (tuple[str, ...]).
    Output: rows, lazily. Side effects: reads path.
    Exceptions: FileNotFoundError if path is missing; ValueError if the
        header lacks a required column.
    """
    with open(path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        header = reader.fieldnames or []
        missing = [column for column in required_columns if column not in header]
        if missing:
            raise ValueError(f"{path.name}: no {', '.join(missing)} column")
        for row in reader:
            yield {column: row.get(column) or "" for column in required_columns}


def _lean_card_content(row: dict[str, str]) -> JsonObject:
    """One cards.csv row reduced to the card as a game object (see the
    module docstring's LEAN CONTENT).

    Inputs: row (dict[str, str], column -> cell text).
    Output: a new JsonObject, keys in _LEADING_KEYS order then text,
        empty values removed.
    Side effects: none.
    Exceptions: ValueError if a numeric cell isn't a number or
        linkmarkers isn't a list literal of strings.

    Example:
        >>> _lean_card_content({"name": "Decode Talker",
        ...     "type": "Link Monster", "race": "Cyberse",
        ...     "attribute": "DARK", "level": "0.0", "linkval": "3.0",
        ...     "linkmarkers": "['Top', 'Bottom-Left']", "atk": "2300.0",
        ...     "def": "", "scale": "", "desc": "2+ Effect Monsters"})
        {'name': 'Decode Talker', 'type': 'Link Monster',
         'attribute': 'DARK', 'monster_type': 'Cyberse', 'link_rating': 3,
         'link_arrows': ['Top', 'Bottom-Left'], 'atk': 2300,
         'text': '2+ Effect Monsters'}
    """
    card_type = row["type"].strip()
    content: JsonObject = {
        "name": row["name"].strip(),
        "type": card_type,
        "attribute": row["attribute"].strip(),
        **_race_field(card_type, row["race"].strip()),
        **_level_and_link_fields(card_type, row),
        "atk": _stat(row["atk"]),
        "def": _stat(row["def"]),
        _TEXT_KEY: _clean_text(row["desc"]),
    }
    return order_keys(strip_noise(content), _LEADING_KEYS, (_TEXT_KEY,))


def _race_field(card_type: str, race: str) -> JsonObject:
    """race under its key for card_type: {"monster_type": race} for a
    monster, {"property": race} for a Spell or Trap.

    Inputs: card_type (str, cards.csv type), race (str).
    Output: a one-key JsonObject. Side effects: none. Exceptions: none.

    Example:
        >>> _race_field("Spell Card", "Quick-Play")
        {'property': 'Quick-Play'}
    """
    if _MONSTER_MARKER in card_type:
        return {"monster_type": race}
    return {"property": race}


def _level_and_link_fields(card_type: str, row: dict[str, str]) -> JsonObject:
    """The level column under its key for card_type: "rank" for an Xyz
    monster, nothing for a Link monster (its link_rating and
    link_arrows instead), else "level"; scale for any card with one.

    Inputs: card_type (str), row (dict[str, str]).
    Output: JsonObject (only keys with a value).
    Side effects: none.
    Exceptions: ValueError if a numeric cell isn't a whole number or
        linkmarkers isn't a list literal of strings.

    Example:
        >>> _level_and_link_fields("XYZ Monster", {"level": "4.0", "scale": "",
        ...     "linkval": "", "linkmarkers": ""})
        {'rank': 4}
    """
    result: JsonObject = {}
    level = _whole_number(row["level"])
    if _LINK_MARKER in card_type:
        result["link_rating"] = _whole_number(row["linkval"])
        arrows: list[JsonValue] = [*_link_arrows(row["linkmarkers"])]
        result["link_arrows"] = arrows
    elif _XYZ_MARKER in card_type:
        result["rank"] = level
    else:
        result["level"] = level
    result["scale"] = _whole_number(row["scale"])
    return {key: value for key, value in result.items() if value not in (None, [])}


def _stat(cell: str) -> int | str | None:
    """An ATK/DEF cell as an int, "?" for _UNKNOWN_STAT, None if empty.

    Inputs: cell (str). Output: int | str | None. Side effects: none.
    Exceptions: ValueError if cell isn't a whole number.
    """
    number = _whole_number(cell)
    if number == _UNKNOWN_STAT:
        return _UNKNOWN_STAT_TEXT
    return number


def _whole_number(cell: str) -> int | None:
    """A float-written count ("4.0") as an int, None if empty.

    Inputs: cell (str). Output: int | None. Side effects: none.
    Exceptions: ValueError if cell isn't a whole number.
    """
    text = cell.strip()
    if not text:
        return None
    number = float(text)
    if not number.is_integer():
        raise ValueError(f"not a whole number: {cell!r}")
    return int(number)


def _link_arrows(cell: str) -> list[str]:
    """linkmarkers ("['Top', 'Bottom-Left']") as a list, [] if empty;
    parsed with ast.literal_eval (a literal, never code).

    Inputs: cell (str). Output: list[str]. Side effects: none.
    Exceptions: ValueError if cell isn't a list literal of strings.
    """
    text = cell.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"linkmarkers is not a list literal: {cell!r}") from error
    if not isinstance(parsed, list) or not all(isinstance(a, str) for a in parsed):
        raise ValueError(f"linkmarkers is not a list of strings: {cell!r}")
    return parsed


def _clean_text(desc: str) -> str:
    """desc with Windows line endings made "\\n" and outer whitespace
    stripped; section markers and inner line breaks kept.

    Inputs: desc (str). Output: str. Side effects: none.
    Exceptions: none.
    """
    return desc.replace("\r\n", "\n").replace("\r", "\n").strip()
