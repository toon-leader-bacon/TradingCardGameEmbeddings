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
    two specific variant rows. Such entries are skipped, see SKIPPED
    below.
  - cards_en_us.json's other 117 keys with no cards_db.json match are
    NOT cards: they're either group/category header text (e.g.
    "Boons", "menagerie events" - these match cards_db.json's
    group_tag/group_top values, not any card_tag) or alternate name
    spellings of split cards cards_db.json already represents as
    separate card_tag rows (e.g. "Border Guard / Horn / Lantern" vs.
    cards_db.json's separate "Border Guard"/"Horn"/"Lantern" rows).
    Since this stage iterates cards_db.json (not cards_en_us.json), none
    of these 117 keys are ever visited or ingested.

LEAN CONTENT (see "What goes in raw_content" in card_binder/README.md).
raw_content is built from a whitelist, since this source is two small,
cleanly keyed files: name (the card_tag), types, cost, potcost, debtcost
and description. Everything else is left out on purpose:
  - cardset_tags: which expansion(s) a card is in. Not the card itself, and
    SetMaskMetric reads it straight from the raw cards_db.json, so nothing
    is lost. A good classification metric, not binder content.
  - extra: the long rulings/clarifications (median 337 characters against a
    106-character description). Not text on the card; a different kind of
    data (like Scryfall's rulings, which are also left out).
  - notes: only used to say which entries are unused or fan-made; see
    "NOT DECIDED HERE" below.
  - randomizer, group_tag, group_top, count, image: the divider generator's
    own bookkeeping (kingdom-randomizer flag, pile grouping and sizes, an
    image file name). randomizer is mostly implied by types (Events,
    Landmarks, Ways... are never kingdom cards); group_tag names other
    cards in a split pile, a cross-card reference.
cost/potcost/debtcost are kept as dominiontabs' own strings (cost "6*" for
a variable cost, "" for a card with no coin cost) and cost is ALWAYS
present, even when empty: CostRegressionMetric indexes raw_content["cost"]
directly and reads "" plus a potcost as a real coin cost of 0
(Transmute, Vineyard), so dropping an empty cost would raise a KeyError
there. types keeps its order, since TypeMaskMetric reads types[0].
description is the card text with the divider generator's layout markup
turned into plain text (see _clean_description): <br>, <n> and <line>
become a newline, <left>/<center>/<justify> start a new line, <u>/<i>/<b>
are removed, and the symbol tags <VP>, <*COIN*> and <*POTION*> become the
words VP, Coin and Potion.

SKIPPED: a cards_db.json entry with no English text is not ingested. Today
that is exactly two entries, "Soothsayer BB2DE" and "Walled Village BB2DE":
German big-box duplicates of cards that are already ingested under their
plain card_tag, which would otherwise become junk-named cards with no
description. (SetMaskMetric skips raw entries that have no binder card, so
it copes.)

NOT DECIDED HERE, still ingested: the 3 fan-made "Animals" cards (notes say
"Fan expansion") and the 13 count == "0" entries that are pile headers
rather than cards (Augurs, Clashes, ... "Settlers - Bustling Village", plus
"Start Deck").

Design, matching scryfall/ingestion_stage.py's shape:
  - DEDUPLICATION: card_tag is dominiontabs' own perfect natural key
    (unique per cards_db.json entry) - the identity lookup is
    binder.get_by_alias(self.SOURCE_GAME, DataSource.DOMINIONTABS,
    card_tag). No name-based or heuristic fallback is needed.
  - COLLISION RESOLUTION: merge_strategies.keep_incoming_if_content_differs.
    card_tag is a perfect natural key, so a newer dump is authoritative
    for its own cards; unchanged content is left alone.
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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_SYMBOL_TAG = re.compile(r"<\*?(VP|COIN|POTION)\*?>")
_SYMBOL_WORDS = {"VP": "VP", "COIN": "Coin", "POTION": "Potion"}
_LINE_BREAK_TAG = re.compile(r"<(?:br|n|line)>")
_BLOCK_OPEN_TAG = re.compile(r"<(?:left|center|justify)>")
_FORMATTING_TAG = re.compile(r"</?(?:u|i|b|left|center|justify)>")
_SPACES = re.compile(r"[ \t\u00a0]+")
_SPACES_AROUND_NEWLINE = re.compile(r" ?\n ?")
_BLANK_LINES = re.compile(r"\n{3,}")

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
            cards_db.json entry that has English text (entries without any
            are skipped).
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
            if db_entry["card_tag"] not in text_by_card_tag:
                continue  # no English text: a German duplicate, see module doc
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
                which ingest() has already checked is present (entries
                without English text are skipped there).
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
        text_entry = text_by_card_tag[card_tag]
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
            merged = merge_strategies.keep_incoming_if_content_differs(
                existing, candidate
            )
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(
            self.SOURCE_GAME, DataSource.DOMINIONTABS, card_tag, stored_uuid
        )

        return stored_uuid if changed else None

    def _build_raw_content(self, db_entry: dict, text_entry: dict) -> dict:
        """Build the lean raw_content dict for one card.

        Private helper — single consumer is _build_card(). Keeps only
        name, types, cost, potcost, debtcost and description — see this
        module's docstring for why every other field on either raw file
        is excluded, and why cost is always present.

        Inputs:
            db_entry: one parsed JSON object from cards_db.json.
            text_entry: the matching cards_en_us.json value
                (`{"description": ..., "name": ..., ...}`).
        Output: a dict with keys "name", "types", "cost" (always, "" when
            the card has none), then "potcost"/"debtcost" only when
            db_entry has them, then "description" (cleaned by
            _clean_description) only when there is English text for it.
        Side effects: none.
        Exceptions: raises if db_entry is missing "card_tag" or "types".
        """
        raw_content: dict = {
            "name": db_entry["card_tag"],
            "types": list(db_entry["types"]),
            "cost": db_entry.get("cost", ""),
        }
        for optional_field in _OPTIONAL_DB_FIELDS:
            if optional_field in db_entry:
                raw_content[optional_field] = db_entry[optional_field]
        description = _clean_description(text_entry["description"])
        if description:
            raw_content["description"] = description
        return raw_content

    def _build_card(
        self,
        db_entry: dict,
        text_entry: dict,
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


def _clean_description(text: str) -> str:
    """Turn a dominiontabs description's layout markup into plain text.

    Inputs: text (str): a cards_en_us.json "description", e.g.
        "+2 Cards<br>+1 Action<line>When you discard this...".
    Output: str: <br>, <n> and <line> as newlines; <left>, <center> and
        <justify> start a new line; <u> becomes a space (Knights lists
        "5 Coin<u>Dame Anna</u>") and </u>, <i>, <b> and closing tags are
        removed; <VP>/<*VP*>, <*COIN*> and <*POTION*> become VP, Coin and
        Potion; runs of spaces (and non-breaking spaces) collapse, spaces
        around newlines go, three or more newlines become two, and the
        ends are stripped.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> _clean_description("+2 Cards<br>+1 Action<n>Gain <*COIN*>.")
        '+2 Cards\n+1 Action\nGain Coin.'
    """
    text = _SYMBOL_TAG.sub(lambda match: _SYMBOL_WORDS[match.group(1)], text)
    text = _LINE_BREAK_TAG.sub("\n", text)
    text = _BLOCK_OPEN_TAG.sub("\n", text)
    text = text.replace("<u>", " ")
    text = _FORMATTING_TAG.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _SPACES_AROUND_NEWLINE.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()
