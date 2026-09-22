"""Translates a spire-codex cards.json dump into stored cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and plans/card_binder_v2.md
for the design this implements. This class is handed a live CardBinder
and owns its own duplicate-detection and collision-resolution directly
against it — see src/data_refinement/card_binder/scryfall/ingestion_stage.py
for the reference shape this follows.

This stage reads a *single* file — like ScryfallCardIngestionStage —
since spire-codex ships its entire card list as one JSON array, not
per-set/per-page files.

Confirmed by sampling the live data/raw/spire_codex/cards.json (577
records):
  - "id" (e.g. "ABRASIVE", or "STRIKE_IRONCLAD"/"STRIKE_SILENT"/etc.
    per character) is present and unique across every record —
    already a perfect natural key, so identity is a plain
    get_by_alias() lookup with no heuristic fallback needed. This is
    what lets "name" stay the raw, un-disambiguated value even though
    "Strike"/"Defend" repeat once per character (5 rows each): under
    the pre-card_binder_v2 design, CardBinder itself enforced
    (source_game, name) uniqueness, which would have silently
    collapsed those 5 distinct characters' cards down to one survivor
    — this stage used to work around that with a
    _disambiguated_name() helper that renamed them to e.g. "Strike
    (Ironclad)". That workaround is no longer needed or present: since
    identity is keyed on "id" (already per-character-unique) rather
    than name, all 5 "Strike" rows correctly get their own nocab_uuid
    regardless of sharing a plain name, and get_by_name("Strike") on
    the built CardBinder correctly returns all 5.
  - No other field plays a role comparable to Scryfall's
    arena_id/mtgo_id/multiverse_ids — no extra aliases are ever
    registered for this source, only each row's own id.

Collision policy is merge_strategies.keep_incoming_if_content_differs,
same as ScryfallCardIngestionStage: id is a perfect natural key, so a newer
dump is authoritative for its own ids.

LEAN CONTENT: raw_content is not the raw row (see _lean_card_content and
"What goes in raw_content" in card_binder/README.md). Most fields are null
on most cards and are dropped; description_raw (the unrendered template
of description), vars (internal template variables), powers_applied (a
structured copy of the text), keywords_key/type_key/rarity_key (copies of
keywords/type/rarity), compendium_order, image urls, spawns_cards and
sources (references to other cards and monsters) are dropped too. upgrade
(a structured diff) is kept only when upgrade_description, its rendered
form, is missing. Color tags like [gold] are stripped; game symbols like
[energy:1] are kept. can_be_generated_in_combat is tri-state (null is the
default, an explicit false is meaningful), so a false is preserved. The
raw row's id is still the identity/alias.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

from tqdm import tqdm

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.lean_content import (
    JsonObject,
    map_strings,
    order_keys,
    strip_noise,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# Dropped wherever they appear (image urls and ids are removed by
# strip_noise already); see the module docstring for why each.
_REDUNDANT_KEYS = frozenset(
    {
        "description_raw",
        "vars",
        "powers_applied",
        "keywords_key",
        "type_key",
        "rarity_key",
        "compendium_order",
        "spawns_cards",
        "sources",
    }
)
# Short identifying fields first; rendered text and nested variants last.
_LEADING_KEYS = (
    "name",
    "cost",
    "star_cost",
    "is_x_cost",
    "type",
    "rarity",
    "color",
    "target",
    "keywords",
    "tags",
    "multiplayer_only",
    "can_be_generated_in_combat",
)
_TRAILING_KEYS = ("description", "upgrade_description", "upgrade", "type_variants")
# [gold]...[/gold], [blue]...[/blue]: color tags. [energy:1] has a colon and
# stays: it is a game symbol, not styling.
_COLOR_TAG = re.compile(r"\[/?[a-z]+\]")
_TRI_STATE_KEY = "can_be_generated_in_combat"


class SpireCodexCardIngestionStage:
    """Translates a spire-codex cards.json file directly into binder.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse a spire-codex cards.json file, creating/updating cards
        directly on binder as a side effect.

        One _ingest_row() call per element of raw_path's top-level
        JSON array — no additional logic beyond calling that and
        collecting the non-None results.

        Inputs:
            raw_path: path to a spire-codex cards.json file (a single
                JSON array of card objects, e.g.
                data/raw/spire_codex/cards.json).
            binder: the CardBinder to create/update cards on and
                register aliases against, as a side effect. Every card
                this call touches is stored under self.SOURCE_GAME
                (GameId.SLAY_THE_SPIRE_2).
        Output: nocab_uuid of every row that caused a create() or an
            actual content-changing replace() this call. A row that
            matched an existing card but changed nothing (an
            exact-duplicate re-fetch) is NOT included.
        Side effects: reads raw_path; creates/updates cards and
            registers aliases directly on binder. Prints a tqdm
            progress bar to stderr, one tick per row (raw_path is
            already fully loaded as one JSON array before this loop
            starts, so its total is known up front).
        Exceptions: raises if raw_path doesn't exist, isn't valid
            JSON, isn't a JSON array, or an element is missing "id" or
            "name".

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/slay_the_spire_2.jsonl")])
            >>> stage = SpireCodexCardIngestionStage()
            >>> changed_uuids = stage.ingest(
            ...     Path("data/raw/spire_codex/cards.json"), binder
            ... )
        """
        with open(raw_path, "r", encoding="utf-8") as raw_file:
            rows = json.load(raw_file)

        changed_uuids = []
        for row in tqdm(rows, desc="spire_codex ingest", unit="card"):
            result = self._ingest_row(row, binder)
            if result is not None:
                changed_uuids.append(result)
        return changed_uuids

    def _ingest_row(self, row: dict, binder: CardBinder) -> UUID | None:
        """Create-or-merge one spire-codex row directly against binder.

        Private helper — single consumer is ingest(). THIS METHOD IS
        WHERE DEDUPLICATION HAPPENS: the identity check —
        binder.get_by_alias(self.SOURCE_GAME, DataSource.SPIRE_CODEX,
        row["id"]) — IS the duplicate check. A hit means row is a
        duplicate of the returned card; a miss means row is new.

        If no existing card resolves: builds a fresh GenericCard and
        calls binder.create().

        If an existing card resolves: builds the same kind of
        GenericCard as a throwaway candidate, then calls
        merge_strategies.keep_incoming_if_content_differs(existing,
        candidate). Compares the result to existing BY VALUE (dataclass
        equality):
        if different, calls binder.replace(existing.nocab_uuid,
        merged).

        Regardless of branch: registers row's own id alias against
        whichever uuid ended up stored — on every branch, not only the
        content-changing ones.

        Inputs:
            row: one parsed JSON object from raw_path's array.
            binder: the CardBinder to read from and write to.
        Output: the stored/canonical nocab_uuid for this row IF this
            call caused a create() or an actual content-changing
            replace(); None if this row matched an existing card but
            changed nothing.
        Side effects: creates or updates exactly one card on binder;
            registers one alias on binder.
        Exceptions: raises if row is missing "id" or "name".
        """
        card_id = row["id"]
        existing = binder.get_by_alias(
            self.SOURCE_GAME, DataSource.SPIRE_CODEX, card_id
        )

        if existing is None:
            card = self._build_card(row, card_id)
            binder.create(card)
            stored_uuid = card.nocab_uuid
            changed = True
        else:
            candidate = self._build_card(row, card_id)
            merged = merge_strategies.keep_incoming_if_content_differs(
                existing, candidate
            )
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(
            self.SOURCE_GAME, DataSource.SPIRE_CODEX, card_id, stored_uuid
        )

        return stored_uuid if changed else None

    def _build_card(self, row: dict, card_id: str) -> GenericCard:
        """Build a fresh GenericCard for one spire-codex row.

        Private helper — single consumer is _ingest_row(), from both
        its create-path and its throwaway-candidate-for-merge path.

        Inputs:
            row: one parsed JSON object from raw_path's array.
            card_id: row["id"], passed in rather than re-read.
        Output: a new GenericCard with a freshly minted nocab_uuid and the
            row's lean content (see _lean_card_content) as raw_content.
        Side effects: none.
        Exceptions: raises if row is missing "name".
        """
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=row["name"],
            raw_content=_lean_card_content(row),
            provenance=Provenance(
                data_source=DataSource.SPIRE_CODEX,
                source_id=card_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )


def _lean_card_content(row: JsonObject) -> JsonObject:
    """Reduce a spire-codex card row to what describes the card.

    Inputs: row (JsonObject): one element of cards.json.
    Output: a new JsonObject: nulls, ids, urls and the redundant keys
        removed, color tags stripped from text, upgrade kept only when
        upgrade_description is missing, an explicit
        can_be_generated_in_combat=false preserved, identity keys first and
        the rendered text last.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> _lean_card_content({"id": "X", "name": "Strike", "cost": 1,
        ...     "damage": None, "type_key": "Attack"})
        {'name': 'Strike', 'cost': 1}
    """
    content = strip_noise(row, extra_noise_keys=_REDUNDANT_KEYS)
    content = map_strings(content, lambda text: _COLOR_TAG.sub("", text))
    if "upgrade_description" in content:
        content.pop("upgrade", None)
    if row.get(_TRI_STATE_KEY) is False:
        content[_TRI_STATE_KEY] = False
    return order_keys(content, _LEADING_KEYS, _TRAILING_KEYS)
