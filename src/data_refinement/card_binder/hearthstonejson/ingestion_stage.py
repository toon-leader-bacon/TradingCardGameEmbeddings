"""Translates one HearthstoneJSON build snapshot (cards.json) into stored
cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface, and spire_codex/ingestion_stage.py for the
reference shape this follows. HearthstoneJSON ships one complete card
pool per game build (data/raw/hearthstonejson/<build id>.json); this stage
reads exactly the one file it is given. scripts/run_card_binder_ingestion.py
passes the newest build (highest build id); older builds are not merged in.

Which rows are cards (checked on build 251951: 35,807 rows):
  - only collectible rows (8,154): the rest are tokens, hero powers,
    enchantments, Tavern Brawl and Battlegrounds rows;
  - minus set HERO_SKINS (755): cosmetic alternate heroes with no cost or
    text, not deck cards. 7,399 rows remain.

Identity: one card per distinct game object. Reprints repeat a card under
several dbfIds (CORE, CORE_HIDDEN, LEGACY, VANILLA, WONDERS ...); 1,019 of
the 1,213 repeated names are the same game object. Rows are grouped by
their lean content without set and rarity (_identity_key), so a reprint
collapses into one card while a rebalanced version (VANILLA's 5-mana Holy
Nova vs. LEGACY's 3-mana one) stays its own card. Every dbfId of a group
is registered as a DataSource.HEARTHSTONEJSON alias (dbfId is what deck
codes use). set and rarity come from the group's lowest dbfId, i.e. its
original printing (rule 10 in card_binder/README.md).

The build is the whole card pool, so the binder mirrors it. Grouping is
content-derived, so it can change between builds (a rebalance splits a
group, a matching rebalance merges two). On an existing binder:
  - a group reuses the stored card its own lowest dbfId names, else the
    one any of its dbfIds names, unless an earlier group of this same
    ingest already took that card (then it gets a new card);
  - after every group is stored, Hearthstone cards no group took (dropped
    from the pool, or left behind by a merge) are deleted. The Unknown
    sentinel is never deleted.
A card that leaves the pool and later returns gets a new nocab_uuid. A
build with no deck cards at all raises instead of emptying the binder.

Collision policy: merge_strategies.keep_incoming_if_content_differs - the
build passed in is authoritative.

LEAN CONTENT (_lean_card_content): an allow-list of game fields
(_LEADING_KEYS + "text"). Dropped: ids, artist, flavor, how-to-earn text,
collectible, elite (= LEGENDARY), race (= races[0]), mechanics and
referencedTags (structured restatements of the text; a future mechanics
metric must read the raw build), overload and spellDamage (restate the
text), heroPowerDbfId/questReward/countAsCopyOfDbfId (references to
other cards), multiClassGroup (classes says it), faction, techLevel and
the battlegrounds keys (other modes), collectionText, targetingArrowText,
hideStats/hideCost and display flags. durability is dropped when 0
(weapons keep their durability in health in this build), and a runeCost
keeps only its nonzero runes. text loses display markup (<b>, <i>, [x],
the $/# spell-damage markers, {0}/{1} placeholders, the "@"-separated
in-game progress suffix, non-breaking spaces and manual line breaks).
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
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

# Cosmetic hero portraits: collectible, but not cards anyone puts in a deck
_COSMETIC_SETS = frozenset({"HERO_SKINS"})

# Kept keys, in serialization order (text is appended last)
_LEADING_KEYS = (
    "name",
    "cost",
    "attack",
    "health",
    "durability",
    "armor",
    "runeCost",
    "type",
    "cardClass",
    "classes",
    "rarity",
    "set",
    "races",
    "spellSchool",
)
_TEXT_KEY = "text"
# Printing-level fields left out of a card's identity (see module docstring)
_PRINTING_KEYS = ("set", "rarity")

# <b>, </b>, <i>, </i>, the [x] manual-layout marker, {0}/{1} placeholders
_DISPLAY_MARKUP = re.compile(r"</?[bi]>|\[x\]|\{\d\}")
# "$3"/"#3": a number scaled by spell damage/healing bonuses; keep the number
_SCALED_NUMBER_MARKER = re.compile(r"[$#](?=\d)")
# Text after "@" is an in-game progress note ("@ ({0} left!)@ (Ready!)")
_PROGRESS_SEPARATOR = "@"
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class _CardGroup:
    """One game object and every build row that prints it.

    content: the lean raw_content (set and rarity from the first row).
    dbf_ids: every row's dbfId, as str, lowest (original printing) first.
    """

    content: JsonObject
    dbf_ids: tuple[str, ...]


class HearthstoneJsonCardIngestionStage:
    """Translates one HearthstoneJSON build file directly into binder.

    Single-consumer to src/data_refinement/card_binder/ - no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse one build's cards.json and make binder's Hearthstone cards
        mirror it: one card per game object.

        Inputs:
            raw_path: one HearthstoneJSON build file (a JSON array of card
                rows, e.g. data/raw/hearthstonejson/251951.json).
            binder: the CardBinder to write into (as self.SOURCE_GAME).
        Output: nocab_uuid of every card created or content-changed
            (deleted cards are not listed).
        Side effects: reads raw_path; creates/replaces/deletes Hearthstone
            cards and registers dbfId aliases on binder; prints a tqdm
            progress bar.
        Exceptions: raises if raw_path is missing or not a JSON array, or
            a kept row lacks "dbfId" or "name"; ValueError if the build
            has no deck cards (rather than deleting every stored card).

        Example:
            >>> binder = CardBinder.load([])
            >>> HearthstoneJsonCardIngestionStage().ingest(
            ...     Path("data/raw/hearthstonejson/251951.json"), binder
            ... )
        """
        with open(raw_path, "r", encoding="utf-8") as raw_file:
            rows = json.load(raw_file)

        groups = _card_groups(rows)
        if not groups:
            raise ValueError(f"{raw_path} has no collectible deck cards")

        # Store every group, each on a card no earlier group has taken
        changed_uuids = []
        taken: set[UUID] = set()
        for group in tqdm(groups, desc="hearthstonejson", unit="card"):
            stored_uuid, changed = self._store_group(group, binder, taken)
            taken.add(stored_uuid)
            if changed:
                changed_uuids.append(stored_uuid)

        # Drop the cards this build no longer has
        self._delete_untaken_cards(binder, taken)
        return changed_uuids

    def _store_group(
        self, group: _CardGroup, binder: CardBinder, taken: set[UUID]
    ) -> tuple[UUID, bool]:
        """Create-or-merge one game object; register every dbfId of the
        group as an alias on every branch.

        Private helper - single consumer is ingest().

        Inputs: group (_CardGroup), binder (CardBinder), taken (uuids
            earlier groups of this ingest stored into; read only).
        Output: (stored nocab_uuid, whether this call created or changed
            the card).
        Side effects: creates or replaces one card; registers aliases.
        Exceptions: none beyond CardBinder's own.
        """
        existing = self._existing_card(group, binder, taken)
        candidate = self._build_card(group)

        if existing is None:
            binder.create(candidate)
            stored_uuid = candidate.nocab_uuid
            changed = True
        else:
            merged = merge_strategies.keep_incoming_if_content_differs(
                existing, candidate
            )
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        for dbf_id in group.dbf_ids:
            binder.register_alias(
                self.SOURCE_GAME, DataSource.HEARTHSTONEJSON, dbf_id, stored_uuid
            )
        return stored_uuid, changed

    def _existing_card(
        self, group: _CardGroup, binder: CardBinder, taken: set[UUID]
    ) -> GenericCard | None:
        """The stored card group should update: the one its lowest dbfId
        names, else the first one any of its dbfIds names, skipping cards
        an earlier group of this ingest already took.

        Private helper - single consumer is _store_group().

        Inputs: group, binder, taken (see _store_group).
        Output: GenericCard | None (None: create a new card).
        Side effects: none. Exceptions: none.
        """
        for dbf_id in group.dbf_ids:
            card = binder.get_by_alias(
                self.SOURCE_GAME, DataSource.HEARTHSTONEJSON, dbf_id
            )
            if card is not None and card.nocab_uuid not in taken:
                return card
        return None

    def _delete_untaken_cards(self, binder: CardBinder, taken: set[UUID]) -> None:
        """Delete every Hearthstone card no group took, except the Unknown
        sentinel.

        Private helper - single consumer is ingest().

        Inputs: binder (CardBinder), taken (uuids stored this ingest).
        Output: none.
        Side effects: deletes cards from binder (their aliases were
            already moved onto taken cards, or name dbfIds this build no
            longer has).
        Exceptions: none beyond CardBinder's own.
        """
        keep = taken | {CardBinder.unknown_card_uuid(self.SOURCE_GAME)}
        stale = [
            uuid for uuid in binder.all_uuids(self.SOURCE_GAME) if uuid not in keep
        ]
        for uuid in stale:
            binder.delete(uuid)

    def _build_card(self, group: _CardGroup) -> GenericCard:
        """A fresh GenericCard for group (a new nocab_uuid; a merge keeps
        the existing one).

        Private helper - single consumer is _store_group().

        Inputs: group (_CardGroup). Output: GenericCard whose source_id is
            the group's original dbfId. Side effects: none.
        Exceptions: none.
        """
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=str(group.content["name"]),
            raw_content=group.content,
            provenance=Provenance(
                data_source=DataSource.HEARTHSTONEJSON,
                source_id=group.dbf_ids[0],
                fetched_at=datetime.now(timezone.utc),
            ),
        )


def _card_groups(rows: list[JsonObject]) -> list[_CardGroup]:
    """Group a build's deck-card rows into one _CardGroup per game object.

    Inputs: rows (the build's JSON array).
    Output: list of _CardGroup, in order of each group's lowest dbfId;
        each group's content takes set/rarity from that lowest-dbfId row.
    Side effects: none.
    Exceptions: KeyError / TypeError if a kept row lacks an int dbfId.

    Example:
        >>> [g.dbf_ids for g in _card_groups(rows)][:2]
        [('1', '1650'), ('2',)]
    """
    deck_rows = sorted((row for row in rows if _is_deck_card(row)), key=_dbf_id)
    groups: dict[str, _CardGroup] = {}
    for row in deck_rows:
        content = _lean_card_content(row)
        key = _identity_key(content)
        group = groups.get(key, _CardGroup(content, ()))
        groups[key] = _CardGroup(group.content, (*group.dbf_ids, str(_dbf_id(row))))
    return list(groups.values())


def _dbf_id(row: JsonObject) -> int:
    """row's dbfId, checked to be an int.

    Inputs: row. Output: int. Side effects: none.
    Exceptions: KeyError if missing; TypeError if not an int.
    """
    dbf_id = row["dbfId"]
    if not isinstance(dbf_id, int):
        raise TypeError(f"dbfId must be an int, got {dbf_id!r}")
    return dbf_id


def _is_deck_card(row: JsonObject) -> bool:
    """Whether row is a collectible card a deck can hold (not cosmetic).

    Inputs: row. Output: bool. Side effects: none. Exceptions: none.
    """
    return row.get("collectible") is True and row.get("set") not in _COSMETIC_SETS


def _identity_key(content: JsonObject) -> str:
    """content without its printing-level keys, as a canonical JSON string.

    Inputs: content (lean raw_content). Output: str.
    Side effects: none. Exceptions: none.
    """
    game_object = {k: v for k, v in content.items() if k not in _PRINTING_KEYS}
    return json.dumps(game_object, sort_keys=True)


def _lean_card_content(row: JsonObject) -> JsonObject:
    """Reduce one build row to the card as a game object (see the module
    docstring for what is kept and why).

    Inputs: row (one JSON object from the build).
    Output: a new JsonObject: allow-listed keys in _LEADING_KEYS order,
        then text; empty values removed; durability 0 and zero runes
        removed; text cleaned of display markup.
    Side effects: none. Exceptions: none.

    Example:
        >>> _lean_card_content({"name": "Holy Smite", "cost": 1,
        ...     "type": "SPELL", "text": "Deal $3 damage\\nto a minion.",
        ...     "artist": "X"})
        {'name': 'Holy Smite', 'cost': 1, 'type': 'SPELL', 'text': 'Deal 3 damage to a minion.'}
    """
    kept: JsonObject = {
        key: row[key] for key in (*_LEADING_KEYS, _TEXT_KEY) if key in row
    }
    if kept.get("durability") == 0:
        del kept["durability"]
    rune_cost = kept.get("runeCost")
    if isinstance(rune_cost, dict):
        kept["runeCost"] = {rune: n for rune, n in rune_cost.items() if n}
    text = kept.get(_TEXT_KEY)
    if isinstance(text, str):
        kept[_TEXT_KEY] = _clean_text(text)
    return order_keys(strip_noise(kept), _LEADING_KEYS, (_TEXT_KEY,))


def _clean_text(text: str) -> str:
    """Card text without display markup: the "@" progress suffix, <b>/<i>
    tags, [x], {0}/{1} placeholders, the $/# marks on scaled numbers,
    non-breaking spaces and layout line breaks.

    Inputs: text (str). Output: str. Side effects: none. Exceptions: none.

    Example:
        >>> _clean_text("[x]<b>Battlecry:</b> Summon a{1} {0} Jade\\nGolem.@ (@)")
        'Battlecry: Summon a Jade Golem.'
    """
    text = text.split(_PROGRESS_SEPARATOR)[0]
    text = _DISPLAY_MARKUP.sub("", text)
    text = _SCALED_NUMBER_MARKER.sub("", text)
    return _WHITESPACE_RUN.sub(" ", text).strip()
