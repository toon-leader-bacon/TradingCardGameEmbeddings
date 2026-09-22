"""Translates cardvault.fabtcg.com's public_card_data.csv dump into
stored cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, card_binder/README.md
for the full design this implements, and
src/data_refinement/card_binder/spire_codex/ingestion_stage.py for the
reference shape this follows most closely (a single static file with a
perfect natural key, no heuristic identity fallback needed).

Confirmed by sampling the full live data/raw/cardvault_fabtcg/
public_card_data.csv (46,660 rows — one row per print: card x
set/reprint x print_language x finish):
  - card_id (e.g. "10000-year-reunion-1") is stable across every
    reprint, every one of the 9 print_language values, and every
    finish/rarity of the same card (5,046 unique values across the
    whole corpus) — a perfect natural key, structurally identical to
    Scryfall's oracle_id. This is the PRIMARY identity/alias.
  - print_id (e.g. "MST131", "MST131-RF", "DE_MST131") identifies one
    specific printing — analogous to Scryfall's arena_id/
    multiverse_ids, registered as a SECONDARY alias alongside card_id.
  - face_1_true_name/face_2_true_name are language-invariant: no
    card_id has more than one distinct face_1_true_name value across
    its rows. face_2_true_name is only populated for genuine
    two-faced cards (489/46,660 rows, e.g. "A Drop in the Ocean //
    Inner Chi" — a permanent specialization flip side, same shape as
    an MTG DFC).
  - Every card_id has at least one print_language == "en" row.

LEAN CONTENT: raw_content is not the CSV row. Of its 56 columns only the
face_N_true_* columns (plus the parsed classes/talents/subtypes/types,
which repeat typebox) are card-level: every one is identical across all
prints of a card_id. Everything else describes a print (print_id, set,
product, artwork, artist, finish, flavor text) or is a per-print display
variant of a true_* column (face_N_name, face_N_rules_text,
face_N_typebox), so it is dropped. The one exception is rarity, which
varies between prints of the same card (28% of cards; e.g. common, marvel
and promo) but is a classic label for metrics: the stored "rarity" is the
LEAST RESTRICTIVE rarity across the card's English prints (see
_RARITY_BY_RESTRICTIVENESS), i.e. the "oracle" rarity, so it is
deterministic and does not depend on which print's row is read last. It is
found in a preliminary scan of the file, before pass 1. The lean content has short unprefixed
keys (name, typebox, pitch, cost, ... textbox last) for the front face and,
for a two-faced card, a nested back_face object. See _lean_card_content()
and "What goes in raw_content" in card_binder/README.md. Identity and
aliases are read from the RAW row.

DELIBERATE TWO-PASS DEPARTURE from every other existing stage's
single-pass shape (see e.g. ScryfallCardIngestionStage.ingest(), which
is one pass over one JSONL line at a time): this source carries the
same card's rules text/flavor text repeated once per print_language,
and letting a non-English row win the merge would make the canonical
stored name/rules_text non-English for no principled reason. So content
(pass 1) is built/merged from print_language == "en" rows only; every
other row (pass 2) only ever registers its own print_id as a secondary
alias against whichever card pass 1 already resolved for that card_id
— never contributing to raw_content. This is safe by construction
because every card_id is confirmed to have an "en" row, so pass 2's
get_by_alias(card_id) lookup can never miss.
"""

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Iterable
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

_ENGLISH_PRINT_LANGUAGE = "en"

# Lean key -> column name after "face_<n>_", in output order: identity and
# short fields first, the long text box last. Only card-level columns; the
# parsed types/classes/talents/subtypes columns are omitted because typebox
# already says the same thing (91-99% are literally contained in it).
_FACE_COLUMNS = {
    "name": "true_name",
    "typebox": "true_typebox",
    "pitch": "true_pitch",
    "pitch_chi": "true_pitch_chi",
    "color": "true_color",
    "cost": "true_cost",
    "power": "true_power",
    "defense": "true_defense",
    "life": "true_life",
    "intellect": "true_intellect",
    "traitbox": "true_traitbox",
    "textbox": "true_textbox",
}
_LINE_BREAK_MARKUP = "{br}"

# From least to most restrictive: how hard a card at that rarity is to get.
# A judgment call about FaB (edit here if it is wrong): tokens and basics are
# free, then the booster rarities in order, then the limited treatments
# (marvel, promo, gold) which are the most restrictive. A rarity not listed
# (a future one) ranks after every listed one.
_RARITY_BY_RESTRICTIVENESS = (
    "token",
    "basic",
    "common",
    "rare",
    "super-rare",
    "majestic",
    "legendary",
    "fabled",
    "marvel",
    "promo",
    "promo-marvel",
    "gold",
    "gold-marvel",
)


class CardVaultFabtcgCardIngestionStage:
    """Translates a cardvault.fabtcg.com public_card_data.csv file directly into binder.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse a cardvault.fabtcg.com public_card_data.csv file,
        creating/updating cards directly on binder as a side effect.

        A preliminary scan (no progress bar) collects each card's least
        restrictive English rarity, then two passes over raw_path's rows,
        see this module's docstring for why. Pass 1 (English rows) does all content
        creation/merging; pass 2 (every other language) only registers
        additional print_id aliases against whatever pass 1 already
        resolved.

        Inputs:
            raw_path: path to a cardvault.fabtcg.com
                public_card_data.csv file (one row per print, e.g.
                data/raw/cardvault_fabtcg/public_card_data.csv).
            binder: the CardBinder to create/update cards on and
                register aliases against, as a side effect. Expected
                to already be loaded (e.g. via CardBinder.load()) by
                the caller — this method never calls load()/save()
                itself. Every card this call touches is stored under
                self.SOURCE_GAME (GameId.FLESH_AND_BLOOD) — there is
                no way to call this method for a different game.
        Output: nocab_uuid of every row that caused a create() or an
            actual content-changing replace() this call. A row that
            matched an existing card but changed nothing (an
            exact-duplicate re-fetch), and every pass-2 (non-English)
            row, is NOT included — see _ingest_content_row()'s
            docstring.
        Side effects: reads raw_path (twice — once per pass); creates/
            updates cards and registers aliases directly on binder.
            Prints one tqdm progress bar to stderr per pass, each sized
            against raw_path's byte size (not its row count, which
            isn't known up front without a separate full read).
        Exceptions: raises if raw_path doesn't exist, isn't a
            well-formed CSV with the expected header, or a row is
            missing "card_id", "print_id", "print_language", or
            "face_1_true_name" (see _ingest_content_row()/
            _register_print_alias()).

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/flesh_and_blood.jsonl")])
            >>> stage = CardVaultFabtcgCardIngestionStage()
            >>> changed_uuids = stage.ingest(
            ...     Path("data/raw/cardvault_fabtcg/public_card_data.csv"), binder
            ... )
        """
        changed_uuids = []
        total_bytes = raw_path.stat().st_size
        rarities = self._least_restrictive_rarities(raw_path)

        # Pass 1: English rows only — all content creation/merging happens here.
        # Opened in binary + wrapped in TextIOWrapper ourselves (rather than
        # open(..., "r") directly) so we can track raw_bytes.tell() - the
        # TextIOWrapper csv.DictReader actually iterates disables .tell()
        # once next() has been called on it (OSError: "telling position
        # disabled by next() call"), but the underlying binary buffer has
        # no such restriction.
        with open(raw_path, "rb") as raw_bytes, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=f"cardvault_fabtcg ingest (pass 1/2): {raw_path.name}",
        ) as progress:
            with io.TextIOWrapper(raw_bytes, encoding="utf-8") as raw_file:
                bytes_read = 0
                for row in csv.DictReader(raw_file):
                    position = raw_bytes.tell()
                    progress.update(position - bytes_read)
                    bytes_read = position

                    if row["print_language"] != _ENGLISH_PRINT_LANGUAGE:
                        continue
                    result = self._ingest_content_row(
                        row, binder, rarities.get(row["card_id"], "")
                    )
                    if result is not None:
                        changed_uuids.append(result)

        # Pass 2: every other language — alias-only, no content changes.
        with open(raw_path, "rb") as raw_bytes, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=f"cardvault_fabtcg ingest (pass 2/2): {raw_path.name}",
        ) as progress:
            with io.TextIOWrapper(raw_bytes, encoding="utf-8") as raw_file:
                bytes_read = 0
                for row in csv.DictReader(raw_file):
                    position = raw_bytes.tell()
                    progress.update(position - bytes_read)
                    bytes_read = position

                    if row["print_language"] == _ENGLISH_PRINT_LANGUAGE:
                        continue
                    self._register_print_alias(row, binder)

        return changed_uuids

    def _ingest_content_row(
        self, row: dict, binder: CardBinder, rarity: str
    ) -> UUID | None:
        """Create-or-merge one English cardvault.fabtcg.com row directly against binder.

        Private helper — single consumer is ingest()'s pass 1. Mirrors
        SpireCodexCardIngestionStage._ingest_row()'s shape: identity is
        get_by_alias(card_id), collision policy is
        merge_strategies.keep_incoming_if_content_differs (every English
        print of a card has the same lean content, so it rarely matters),
        and both card_id
        (primary) and this row's own print_id (secondary) are
        registered on every branch — including a no-op merge — per
        card_binder/README.md's AliasLedger "register on every branch"
        convention.

        Inputs:
            row: one parsed CSV row (a dict, via csv.DictReader) with
                print_language == "en".
            binder: the CardBinder to read from and write to.
            rarity: the card's least restrictive English rarity ("" if
                none of its rows has one).
        Output: the stored/canonical nocab_uuid for this row IF this
            call caused a create() or an actual content-changing
            replace(); None if this row matched an existing card but
            changed nothing.
        Side effects: creates or updates exactly one card on binder;
            registers two aliases (card_id, print_id) on binder.
        Exceptions: raises if row is missing "card_id", "print_id", or
            "face_1_true_name".
        """
        card_id = row["card_id"]
        existing = binder.get_by_alias(
            self.SOURCE_GAME, DataSource.CARDVAULT_FABTCG, card_id
        )

        if existing is None:
            card = self._build_card(row, card_id, rarity)
            binder.create(card)
            stored_uuid = card.nocab_uuid
            changed = True
        else:
            candidate = self._build_card(row, card_id, rarity)
            merged = merge_strategies.keep_incoming_if_content_differs(
                existing, candidate
            )
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(
            self.SOURCE_GAME, DataSource.CARDVAULT_FABTCG, card_id, stored_uuid
        )
        binder.register_alias(
            self.SOURCE_GAME, DataSource.CARDVAULT_FABTCG, row["print_id"], stored_uuid
        )

        return stored_uuid if changed else None

    def _least_restrictive_rarities(self, raw_path: Path) -> dict[str, str]:
        """Each card_id's least restrictive rarity over its English rows.

        Inputs: raw_path (Path): the public_card_data.csv file.
        Output: dict[str, str]: card_id -> rarity. A card whose English rows
            all have an empty rarity has no entry.
        Side effects: reads raw_path once.
        Exceptions: raises if raw_path is unreadable or a row lacks
            "card_id", "print_language" or "rarity".
        """
        rarities_by_card: dict[str, set[str]] = {}
        with open(raw_path, "r", newline="", encoding="utf-8") as raw_file:
            for row in csv.DictReader(raw_file):
                if row["print_language"] == _ENGLISH_PRINT_LANGUAGE and row["rarity"]:
                    rarities_by_card.setdefault(row["card_id"], set()).add(
                        row["rarity"]
                    )
        return {
            card_id: _least_restrictive(rarities)
            for card_id, rarities in rarities_by_card.items()
        }

    def _register_print_alias(self, row: dict, binder: CardBinder) -> None:
        """Register one non-English row's print_id against its already-ingested card.

        Private helper — single consumer is ingest()'s pass 2. Never
        builds or merges a GenericCard — pass 1 is guaranteed (per
        this module's docstring) to have already created the card for
        row["card_id"], so this only needs to resolve it and register
        one more secondary alias.

        Inputs:
            row: one parsed CSV row (a dict, via csv.DictReader) with
                print_language != "en".
            binder: the CardBinder to read from and write to.
        Output: none.
        Side effects: registers one alias (print_id) on binder.
        Exceptions: raises if row is missing "card_id" or "print_id",
            or if row["card_id"] resolves to no existing card (would
            indicate the "every card_id has an en row" corpus
            assumption this module's docstring documents no longer
            holds).
        """
        card_id = row["card_id"]
        existing = binder.get_by_alias(
            self.SOURCE_GAME, DataSource.CARDVAULT_FABTCG, card_id
        )
        if existing is None:
            raise ValueError(
                f"_register_print_alias: no card resolves for card_id={card_id!r} — "
                "pass 1 should have already created one from this card_id's 'en' row"
            )

        binder.register_alias(
            self.SOURCE_GAME,
            DataSource.CARDVAULT_FABTCG,
            row["print_id"],
            existing.nocab_uuid,
        )

    def _build_card(self, row: dict, card_id: str, rarity: str) -> GenericCard:
        """Build a fresh GenericCard for one English cardvault.fabtcg.com row.

        Private helper — single consumer is _ingest_content_row(),
        from both its create-path and its throwaway-candidate-for-merge
        path. name is row["face_1_true_name"], joined as
        f"{face_1} // {face_2}" when row.get("face_2_true_name") is
        truthy — matching Scryfall's existing DFC naming convention.

        Inputs:
            row: one parsed CSV row (a dict, via csv.DictReader) with
                print_language == "en".
            card_id: row["card_id"], passed in rather than re-read,
                since callers already have it at hand.
            rarity: the card's least restrictive English rarity.
        Output: a new GenericCard with a freshly minted nocab_uuid and the
            row's lean content (see _lean_card_content) as raw_content.
        Side effects: none.
        Exceptions: raises if row is missing "face_1_true_name".
        """
        name = row["face_1_true_name"]
        if row.get("face_2_true_name"):
            name = f"{name} // {row['face_2_true_name']}"

        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=name,
            raw_content=_lean_card_content(row, rarity),
            provenance=Provenance(
                data_source=DataSource.CARDVAULT_FABTCG,
                source_id=card_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )


def _least_restrictive(rarities: Iterable[str]) -> str:
    """The rarity that is least restrictive (see _RARITY_BY_RESTRICTIVENESS).

    Unlisted rarities rank after every listed one, and ties between them
    break alphabetically, so the result never depends on input order.

    Inputs: rarities (Iterable[str]): at least one rarity.
    Output: str.
    Side effects: none.
    Exceptions: ValueError if rarities is empty.

    Example:
        >>> _least_restrictive(["promo", "marvel", "common"])
        'common'
    """

    def restrictiveness(rarity: str) -> tuple[int, str]:
        if rarity in _RARITY_BY_RESTRICTIVENESS:
            return _RARITY_BY_RESTRICTIVENESS.index(rarity), rarity
        return len(_RARITY_BY_RESTRICTIVENESS), rarity

    return min(rarities, key=restrictiveness)


def _lean_card_content(row: dict[str, str], rarity: str) -> JsonObject:
    """Reduce one English CSV row to the card-level content of the card.

    Inputs: row (dict[str, str]): one parsed CSV row. rarity (str): the
        card's least restrictive English rarity, "" if it has none.
    Output: a new JsonObject: the front face's fields under short keys
        (see _FACE_COLUMNS), the rarity, plus a nested "back_face" object
        when the card has a second face. Empty columns are omitted.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> _lean_card_content(
        ...     {"face_1_true_name": "Bolt", "face_1_true_cost": "1"}, "common"
        ... )
        {'name': 'Bolt', 'rarity': 'common', 'cost': '1'}
    """
    content = _face_content(row, 1)
    if rarity:
        content["rarity"] = rarity
    if row.get("face_2_true_name"):
        content["back_face"] = _face_content(row, 2)
    return order_keys(
        content,
        leading=("name", "typebox", "rarity"),
        trailing=("textbox", "back_face"),
    )


def _face_content(row: dict[str, str], face_number: int) -> JsonObject:
    """One face's card-level fields, empties dropped, {br} turned into a
    newline (the same convention gwent_one's stage uses for its text)."""
    fields: JsonObject = {
        lean_key: row.get(f"face_{face_number}_{column}", "")
        for lean_key, column in _FACE_COLUMNS.items()
    }
    textbox = fields["textbox"]
    if isinstance(textbox, str):
        fields["textbox"] = textbox.replace(_LINE_BREAK_MARKUP, "\n")
    return strip_noise(fields)
