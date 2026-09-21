"""Translates sumpfork/dominiontabs' two raw card files into stored cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and
src/data_retrieval/dominiontabs/downloader.py for why this source's raw
data is split across two files rather than one flat record:
`cards_db.json` (819 entries, a JSON list) holds language-neutral
fields keyed by `card_tag`; `cards_en_us.json` (884 entries, a JSON
dict keyed by card name) holds English rules text. This stage joins
them by `card_tag` == cards_en_us.json's dict key.

Confirmed by sampling both live raw files directly
(data/raw/dominiontabs/cards_db.json, cards_en_us.json):
  - The join matches for 817 of 819 cards_db.json entries. The 2
    misses ("Soothsayer BB2DE", "Walled Village BB2DE") are German
    big-box promo variants of cards already ingested under their plain
    card_tag - not missing cards, just missing English text for these
    two specific variant rows. Handled by falling back to no
    description rather than raising - see _ingest_card().
  - cards_en_us.json's other 117 keys with no cards_db.json match are
    NOT cards: they're either group/category header text (e.g.
    "Boons", "menagerie events" - these match cards_db.json's
    group_tag/group_top values, not any card_tag) or alternate name
    spellings of split cards cards_db.json already represents as
    separate card_tag rows (e.g. "Border Guard / Horn / Lantern" vs.
    cards_db.json's separate "Border Guard"/"Horn"/"Lantern" rows).
    Since this stage iterates cards_db.json (not cards_en_us.json), none
    of these 117 keys are ever visited or ingested.

Per explicit instruction: only six raw fields are kept verbatim in
raw_content (name, types, cost, description, potcost, debtcost) - see
_build_raw_content(). Every other field on either raw file
(cardset_tags, group_tag, group_top, count, randomizer, image, extra,
notes) is deliberately excluded from this stage - either
divider-generator-specific bookkeeping with no use here (group_tag,
group_top, count, randomizer, image), or text not judged useful
(extra, notes). cardset_tags is left out of ingestion for now, not
deleted from the raw file - a later stage that wants "which
expansion(s) is this card in" can still read it directly from
data/raw/dominiontabs/cards_db.json.

`cost`/`potcost`/`debtcost` are kept as the raw strings dominiontabs
itself uses (e.g. cost "6*" for a variable-cost card, "" for an
uncosted Landmark/Way/Boon/etc.) - not parsed into int here, per
explicit instruction not to reshape this data more than necessary.
`description`'s `<br>`/`<n>` tokens are likewise left untouched.

Design, matching scryfall/ingestion_stage.py's shape:
  - DEDUPLICATION: card_tag is dominiontabs' own perfect natural key
    (unique per cards_db.json entry) - the identity lookup is
    binder.get_by_alias(self.SOURCE_GAME, DataSource.DOMINIONTABS,
    card_tag). No name-based or heuristic fallback is needed.
  - COLLISION RESOLUTION: merge_strategies.keep_longer_content, same
    policy every other stage in this container uses.
  - Every entry's card_tag is registered as its alias on EVERY
    branch of _ingest_card() - including a no-op branch - per
    card_binder/README.md's "a losing/no-op row's identifier must
    never become a dead end" rule.
  - No secondary identifier system exists in this data (unlike e.g.
    Scryfall's arena_id/mtgo_id/multiverse_ids) - card_tag is the
    only identifier dominiontabs' raw files carry, so card_tag is the
    only alias ever registered.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_OPTIONAL_DB_FIELDS: tuple[str, ...] = ("potcost", "debtcost")
# Present only on some cards_db.json entries (potcost: Alchemy cards;
# debtcost: Dark Ages/Renaissance cards) — copied into raw_content only
# when db_entry actually has them, unlike "types"/"cost" below which
# are always attempted.


class DominionTabsCardIngestionStage:
    """Translates dominiontabs' cards_db.json + cards_en_us.json directly into binder.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.DOMINION

    CARDS_DB_FILENAME: ClassVar[str] = "cards_db.json"
    CARDS_EN_US_FILENAME: ClassVar[str] = "cards_en_us.json"

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse raw_path's two dominiontabs files, creating/updating
        cards directly on binder as a side effect.

        One _ingest_card() call per entry of cards_db.json — that file,
        not cards_en_us.json, drives iteration, since it's the one with
        a real per-card identity (card_tag); see this module's
        docstring for why cards_en_us.json's extra keys are never
        visited.

        Inputs:
            raw_path: path to a directory containing both
                CARDS_DB_FILENAME and CARDS_EN_US_FILENAME (e.g.
                data/raw/dominiontabs/, as written by
                DominionTabsCardDownloader).
            binder: the CardBinder to create/update cards on and
                register aliases against, as a side effect. Expected
                to already be loaded by the caller — this method never
                calls load()/save() itself. Every card this call
                touches is stored under self.SOURCE_GAME
                (GameId.DOMINION).
        Output: nocab_uuid of every cards_db.json entry that caused a
            create() or an actual content-changing replace() this
            call. An entry that matched an existing card but changed
            nothing is NOT included — see _ingest_card()'s docstring.
        Side effects: reads raw_path/CARDS_DB_FILENAME and
            raw_path/CARDS_EN_US_FILENAME; creates/updates cards and
            registers aliases directly on binder, once per
            cards_db.json entry.
        Exceptions: raises if either file is missing under raw_path,
            isn't valid JSON, or a cards_db.json entry is missing
            "card_tag" or "types".

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/dominion.jsonl")])
            >>> stage = DominionTabsCardIngestionStage()
            >>> changed_uuids = stage.ingest(Path("data/raw/dominiontabs"), binder)
        """
        db_entries = json.loads(
            (raw_path / self.CARDS_DB_FILENAME).read_text(encoding="utf-8")
        )
        text_by_card_tag = json.loads(
            (raw_path / self.CARDS_EN_US_FILENAME).read_text(encoding="utf-8")
        )

        changed_uuids = []
        for db_entry in db_entries:
            result = self._ingest_card(db_entry, text_by_card_tag, binder)
            if result is not None:
                changed_uuids.append(result)
        return changed_uuids

    def _ingest_card(
        self,
        db_entry: dict,
        text_by_card_tag: dict,
        binder: CardBinder,
    ) -> UUID | None:
        """Create-or-merge one cards_db.json entry directly against binder.

        Inputs:
            db_entry: one parsed JSON object from cards_db.json.
            text_by_card_tag: the full parsed cards_en_us.json dict
                (keyed by card name) — looked up by db_entry["card_tag"],
                which misses for 2 of 819 entries (German promo
                variants — see this module's docstring); a miss is
                tolerated, not an error.
            binder: the CardBinder to read from and write to.
        Output: the stored/canonical nocab_uuid for this entry IF this
            call caused a create() or an actual content-changing
            replace(); None if this entry matched an existing card but
            changed nothing.
        Side effects: creates or updates exactly one card on binder;
            registers db_entry["card_tag"] as a
            (self.SOURCE_GAME, DataSource.DOMINIONTABS) alias on
            binder, on every branch (including a no-op) — see this
            module's docstring.
        Exceptions: raises if db_entry is missing "card_tag" or "types"
            (the latter via _build_card() -> _build_raw_content()).
        """
        card_tag = db_entry["card_tag"]
        text_entry = text_by_card_tag.get(card_tag)
        existing = binder.get_by_alias(
            self.SOURCE_GAME, DataSource.DOMINIONTABS, card_tag
        )

        if existing is None:
            # New card: build a fresh GenericCard and create it.
            card = self._build_card(db_entry, text_entry, card_tag)
            binder.create(card)
            stored_uuid = card.nocab_uuid
            changed = True
        else:
            # Existing card: build a throwaway candidate and merge it.
            candidate = self._build_card(db_entry, text_entry, card_tag)
            merged = merge_strategies.keep_longer_content(existing, candidate)
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(
            self.SOURCE_GAME, DataSource.DOMINIONTABS, card_tag, stored_uuid
        )

        return stored_uuid if changed else None

    def _build_raw_content(self, db_entry: dict, text_entry: dict | None) -> dict:
        """Build the trimmed raw_content dict for one card.

        Private helper — single consumer is _build_card(). Keeps only
        name, types, cost, description, potcost, debtcost — see this
        module's docstring for why every other field on either raw
        file is excluded.

        Inputs:
            db_entry: one parsed JSON object from cards_db.json.
            text_entry: the matching cards_en_us.json value
                (`{"description": ..., "name": ..., ...}`), or None if
                no match was found (see _ingest_card()).
        Output: a dict with keys "name", "types", "cost", "description",
            and, only when present on db_entry, "potcost"/"debtcost".
            "description" is "" when text_entry is None. cost/potcost/
            debtcost are kept as dominiontabs' own raw strings,
            unparsed. description's `<br>`/`<n>` tokens are left as-is.
        Side effects: none.
        Exceptions: raises if db_entry is missing "card_tag" or "types".
        """
        raw_content: dict = {
            "name": db_entry["card_tag"],
            "types": db_entry["types"],
            "cost": db_entry.get("cost", ""),
            "description": text_entry["description"] if text_entry is not None else "",
        }
        for optional_field in _OPTIONAL_DB_FIELDS:
            if optional_field in db_entry:
                raw_content[optional_field] = db_entry[optional_field]
        return raw_content

    def _build_card(
        self,
        db_entry: dict,
        text_entry: dict | None,
        card_tag: str,
    ) -> GenericCard:
        """Build a fresh GenericCard for one dominiontabs card.

        Private helper — single consumer is _ingest_card(), from both
        its create-path and its throwaway-candidate-for-merge path.

        Inputs:
            db_entry: one parsed JSON object from cards_db.json.
            text_entry: see _build_raw_content().
            card_tag: db_entry["card_tag"], passed in rather than
                re-read, since callers already have it at hand.
        Output: a new GenericCard with a freshly minted nocab_uuid,
            name=card_tag, raw_content=self._build_raw_content(...).
        Side effects: none.
        Exceptions: raises whatever _build_raw_content() raises.
        """
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=card_tag,
            raw_content=self._build_raw_content(db_entry, text_entry),
            provenance=Provenance(
                data_source=DataSource.DOMINIONTABS,
                source_id=card_tag,
                fetched_at=datetime.now(timezone.utc),
            ),
        )
