"""The in-memory, multi-game card store with add-time uniqueness.

See src/data_refinement/README.md for this container's scope and
src/data_refinement/card_binder/README.md for the full design this
implements.

CardBinder enforces one invariant: at most one GenericCard per
(source_game, name). Enforcement happens at add() time, not at
get_by_name() query time — a validly-built binder is guaranteed
collision-free by construction, so get_by_name() stays a plain
GenericCard | None lookup with no ambiguity to resolve.

CardBinder is a Facade (PATTERNS.md) over the canonical card store
PLUS a privately owned AliasLedger — callers never see AliasLedger
directly, only CardBinder's own get_by_alias()/register_alias().
"""

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

from src.data_refinement.card_binder.alias_ledger import AliasLedger
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class AddOutcome(str, Enum):
    """What add() did with a candidate card.

    Inputs: none (enum).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    INSERTED = "inserted"  # no prior card existed for this name
    CONTENT_UPDATED = "content_updated"  # candidate's raw_content/provenance
    # replaced the existing card's (existing nocab_uuid kept)
    KEPT_EXISTING = "kept_existing"  # existing card's raw_content/provenance
    # kept (candidate lost the richness comparison, or exactly tied)


@dataclass(frozen=True)
class AddResult:
    """The result of one CardBinder.add() call.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    stored_card: GenericCard  # canonical card for this name after add()
    outcome: AddOutcome


class CardBinder:
    """In-memory, multi-game card store, keyed by nocab_uuid/alias/name.

    Single-consumer-per-instance is not the right frame here — this is
    the shared, generic lookup capability every data_refinement stage
    and training itself reads from (see this container's README).
    Construction is always empty (__init__ takes no
    arguments); load() is the sole populated-construction entry point.
    """

    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/cards")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{game}.jsonl"

    def __init__(self) -> None:
        """Construct an empty binder, with an empty owned AliasLedger.

        Prefer load([]) over calling this directly when the intent is
        "start empty for an ingestion run" — this constructor exists
        because load() is itself built on top of it.

        Inputs: none.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._cards_by_uuid: dict[UUID, GenericCard] = {}
        self._uuid_by_name: dict[tuple[GameId, str], UUID] = {}
        self._alias_ledger = AliasLedger()

    def get_by_uuid(self, nocab_uuid: UUID) -> GenericCard | None:
        """Look up a card by this system's own identity.

        Inputs:
            nocab_uuid: identity to look up.
        Output: the matching GenericCard, or None if no card has this
            nocab_uuid.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> binder.get_by_uuid(some_uuid)
        """
        return self._cards_by_uuid.get(nocab_uuid)

    def get_by_alias(
        self, source_game: GameId, data_source: DataSource, source_id: str
    ) -> GenericCard | None:
        """Look up a card by an external system's identifier for it.

        Replaces the earlier get_by_source_id — resolves through the
        internal AliasLedger, then get_by_uuid(). Works regardless of
        whether source_id belonged to the card that "won" a name
        collision at add() time or the one that "lost" — both remain
        resolvable to the one surviving card (see add()'s docstring).

        Inputs:
            source_game: which game's namespace to look in.
            data_source: which external system minted source_id (e.g.
                DataSource.SCRYFALL, DataSource.ARENA).
            source_id: that system's own id (e.g. a Scryfall oracle_id
                or Arena numeric card id).
        Output: the matching GenericCard, or None if no card is
            registered under this (source_game, data_source, source_id).
        Side effects: none.
        Exceptions: none.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> binder.get_by_alias(GameId.MTG, DataSource.ARENA, "76497")
        """
        nocab_uuid = self._alias_ledger.resolve(source_game, data_source, source_id)
        if nocab_uuid is None:
            return None
        return self.get_by_uuid(nocab_uuid)

    def get_by_name(self, source_game: GameId, name: str) -> GenericCard | None:
        """Look up a card by name.

        Guaranteed at most one match — CardBinder enforces uniqueness
        at add() time, not here (see this module's docstring and
        add()'s).

        Inputs:
            source_game: which game's name namespace to look in.
            name: card name to look up.
        Output: the matching GenericCard, or None if no card is
            registered under this (source_game, name).
        Side effects: none.
        Exceptions: none.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> binder.get_by_name(GameId.MTG, "Secret Invasion")
        """
        nocab_uuid = self._uuid_by_name.get((source_game, name))
        if nocab_uuid is None:
            return None
        return self.get_by_uuid(nocab_uuid)

    def all_uuids(self) -> Iterable[UUID]:
        """Every nocab_uuid currently stored in this binder.

        Inputs: none.
        Output: every nocab_uuid this binder currently holds a card
            for, in no particular guaranteed order.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> list(binder.all_uuids())
        """
        return self._cards_by_uuid.keys()

    def get_by_name_regex(self, source_game: GameId, pattern: str) -> list[GenericCard]:
        """Look up every card whose name matches a regex pattern.

        Generic, source-agnostic search — ported unchanged from the
        pre-refactor CardRegistry. See
        src/data_refinement/seventeenlands/game_data_metrics/ and
        draft_data_metrics/ for the 17lands-specific fallback logic
        that calls this — NOT this method's concern.

        Inputs:
            source_game: which game's cards to search.
            pattern: a regular expression, matched against each
                candidate card's name via re.match(pattern, name).
        Output: every GenericCard in source_game whose name matches
            pattern, in no particular guaranteed order.
        Side effects: none.
        Exceptions: raises re.error if pattern is not a valid regular
            expression.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> binder.get_by_name_regex(GameId.MTG, r"^Bruce Banner( //.*)?$")
        """
        return [
            card
            for card in self._cards_by_uuid.values()
            if card.source_game == source_game and re.match(pattern, card.name)
        ]

    def add(self, candidate: GenericCard) -> AddResult:
        """Add or merge a candidate card, enforcing the uniqueness invariant.

        Collision/richness comparison logic is ported unchanged from
        the pre-refactor CardRegistry.add(), not redesigned: if no
        card exists yet for candidate's
        (source_game, name), it's inserted as-is. If one does, the
        richer raw_content wins (ties favor the existing card); the
        WINNING card's raw_content AND provenance both come from
        whichever card won (provenance moves atomically with
        raw_content, same as source_id does today) — but the
        EXISTING card's nocab_uuid is always kept, regardless of which
        content wins.

        Regardless of outcome, this method registers candidate's own
        provenance (data_source, source_id) into the internal
        AliasLedger via register_alias() — even on KEPT_EXISTING, so a
        losing candidate's primary identifier never becomes a dead end
        for get_by_alias(). Registering any EXTRA identifiers beyond
        provenance (e.g. an ingested candidate's arena_id) is the
        caller's job — see register_alias(), called once per extra
        identifier by build.py's driver, not by add() itself.

        Inputs:
            candidate: a GenericCard to add, typically fresh from a
                CardIngestionStage (with its own newly minted
                nocab_uuid, which may or may not end up canonical).
        Output: an AddResult naming the stored (canonical) card for
            this name and which of the three outcomes occurred.
        Side effects: mutates this binder's in-memory indices and its
            owned AliasLedger.
        Exceptions: none expected — candidate is assumed to already be
            a valid GenericCard.

        Example:
            >>> binder = CardBinder()
            >>> result = binder.add(candidate_card)
            >>> result.outcome
            <AddOutcome.INSERTED: 'inserted'>
        """
        existing = self._find_existing_by_name(candidate.source_game, candidate.name)

        if existing is None:
            stored = candidate
            self._store_card(stored)
            outcome = AddOutcome.INSERTED
        elif self._is_richer(candidate, existing):
            stored = replace(
                existing,
                raw_content=candidate.raw_content,
                provenance=candidate.provenance,
            )
            self._store_card(stored)
            outcome = AddOutcome.CONTENT_UPDATED
        else:
            stored = existing
            outcome = AddOutcome.KEPT_EXISTING

        self.register_alias(
            candidate.source_game,
            candidate.provenance.data_source,
            candidate.provenance.source_id,
            stored.nocab_uuid,
        )

        return AddResult(stored_card=stored, outcome=outcome)

    def register_alias(
        self,
        source_game: GameId,
        data_source: DataSource,
        source_id: str,
        nocab_uuid: UUID,
    ) -> None:
        """Register one external identifier against an already-stored card.

        Driver-only — same "construction-time concern" status as
        add()/load()/save(); training never calls this. Used by build.py's
        driver once per IngestedCandidate.aliases entry, after add()
        has already stored the candidate.

        Inputs:
            source_game: which game's namespace this identifier
                belongs to.
            data_source: which external system minted source_id.
            source_id: that system's own id.
            nocab_uuid: the nocab_uuid this identifier should resolve
                to — MUST already be a card this binder holds.
        Output: none.
        Side effects: mutates this binder's owned AliasLedger.
        Exceptions: raises ValueError if nocab_uuid isn't a card this
            binder currently holds — an AliasLedger entry pointing at
            a phantom uuid would be a silent data-integrity bug.

        Example:
            >>> binder = CardBinder()
            >>> result = binder.add(candidate_card)
            >>> nocab_uuid = result.stored_card.nocab_uuid
            >>> binder.register_alias(GameId.MTG, DataSource.ARENA, "76497", nocab_uuid)
        """
        if nocab_uuid not in self._cards_by_uuid:
            raise ValueError(f"register_alias: {nocab_uuid} is not a card this binder holds")
        self._alias_ledger.register(source_game, data_source, source_id, nocab_uuid)

    @staticmethod
    def load(paths: list[Path]) -> "CardBinder":
        """Load and merge one or more per-game JSONL files into one binder.

        Merge/collision semantics (including per-path uuid remapping
        for cross-file collisions) are ported unchanged from the
        pre-refactor CardRegistry.load(). Generalized to read each path's
        sibling <game>.alias_ledger.jsonl file (renamed from
        <game>.aliases.jsonl) and re-register every entry via
        register_alias() — rather than writing the internal
        AliasLedger directly — so the same phantom-uuid guard
        register_alias() enforces applies during load() too.

        load([]) — an empty list — is the sanctioned way to start a
        fresh, empty binder.

        Inputs:
            paths: zero or more paths to data/final/cards/<game>.jsonl-
                shaped files. Files for different games can be mixed
                in one call.
        Output: a populated CardBinder containing every card from
            every given path, deduplicated per add()'s rules, with
            every alias (if any sibling alias_ledger files are
            present) also resolvable.
        Side effects: reads each path in paths, and each one's sibling
            alias_ledger file if present.
        Exceptions: raises if any path in paths doesn't exist or
            contains a line that doesn't deserialize to a valid
            GenericCard.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> empty_binder = CardBinder.load([])
        """
        binder = CardBinder()
        for path in paths:
            # Maps this path's OWN nocab_uuids (as written by whatever
            # save() produced this file) to the card actually stored in
            # the merged binder after add() — which may carry a
            # different nocab_uuid, if this path's card lost a
            # cross-file collision to an earlier-loaded path.
            original_uuid_to_stored_card: dict[UUID, GenericCard] = {}

            with open(path, "r", encoding="utf-8") as binder_file:
                for line in binder_file:
                    row = json.loads(line)
                    original_uuid = UUID(row["nocab_uuid"])
                    card = GenericCard(
                        nocab_uuid=original_uuid,
                        source_game=GameId(row["source_game"]),
                        name=row["name"],
                        raw_content=row["raw_content"],
                        provenance=Provenance(
                            data_source=DataSource(row["provenance"]["data_source"]),
                            source_id=row["provenance"]["source_id"],
                            fetched_at=datetime.fromisoformat(row["provenance"]["fetched_at"]),
                        ),
                    )
                    result = binder.add(card)
                    original_uuid_to_stored_card[original_uuid] = result.stored_card

            alias_ledger_path = CardBinder._alias_ledger_path(path)
            if alias_ledger_path.exists():
                with open(alias_ledger_path, "r", encoding="utf-8") as ledger_file:
                    for line in ledger_file:
                        alias_row = json.loads(line)
                        original_uuid = UUID(alias_row["nocab_uuid"])
                        stored_card = original_uuid_to_stored_card.get(original_uuid)
                        if stored_card is None:
                            continue
                        binder.register_alias(
                            stored_card.source_game,
                            DataSource(alias_row["data_source"]),
                            alias_row["source_id"],
                            stored_card.nocab_uuid,
                        )
        return binder

    @staticmethod
    def default_output_path(source_game: GameId) -> Path:
        """The conventional save() path for one game's binder file.

        A recommended default, not an enforced requirement — callers
        remain free to save() to any path; this exists so a caller
        (e.g. a Dojo defaulting its own metrics_path/binder location)
        can compute the same conventional path this project already
        uses elsewhere, instead of re-typing the string.

        Inputs:
            source_game: which game's conventional path to compute.
        Output: DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME, formatted
            with source_game (e.g. data/final/cards/mtg.jsonl).
        Side effects: none — purely a path computation.
        Exceptions: none.

        Example:
            >>> CardBinder.default_output_path(GameId.MTG)
            PosixPath('data/final/cards/mtg.jsonl')
        """
        return CardBinder.DEFAULT_OUTPUT_DIR / CardBinder.DEFAULT_OUTPUT_NAME.format(
            game=source_game.value
        )

    def save(self, path: Path, source_game: GameId) -> None:
        """Write this binder's cards for one game to path, as JSONL.

        Only cards with card.source_game == source_game are written.
        Writes the main file, then delegates to
        self._alias_ledger.save(...) for the sibling
        <game>.alias_ledger.jsonl file (renamed from
        <game>.aliases.jsonl) — CardBinder no longer writes alias
        entries itself; AliasLedger owns its own file format.

        Inputs:
            path: destination file (e.g.
                data/final/cards/<game>.jsonl). Overwritten if it
                already exists.
            source_game: which game's subset of this binder to write.
        Output: none.
        Side effects: creates path's parent directory if missing;
            writes/overwrites path and its sibling alias_ledger file.
        Exceptions: raises on failure to write either file.

        Example:
            >>> binder.save(Path("data/final/cards/mtg.jsonl"), GameId.MTG)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as binder_file:
            for card in self._cards_by_uuid.values():
                if card.source_game != source_game:
                    continue
                row = {
                    "nocab_uuid": str(card.nocab_uuid),
                    "source_game": card.source_game.value,
                    "name": card.name,
                    "raw_content": card.raw_content,
                    "provenance": {
                        "data_source": card.provenance.data_source.value,
                        "source_id": card.provenance.source_id,
                        "fetched_at": card.provenance.fetched_at.isoformat(),
                    },
                }
                binder_file.write(json.dumps(row) + "\n")

        self._alias_ledger.save(self._alias_ledger_path(path), source_game)

    def _find_existing_by_name(self, source_game: GameId, name: str) -> GenericCard | None:
        """Look up the currently-stored card for (source_game, name), if any.

        Private helper — single consumer is add(). Ported unchanged from the
        pre-refactor CardRegistry.

        Inputs:
            source_game: which game's name namespace to look in.
            name: card name to look up.
        Output: the currently-stored GenericCard for this name, or
            None if none exists yet.
        Side effects: none.
        Exceptions: none.
        """
        return self.get_by_name(source_game, name)

    def _is_richer(self, candidate: GenericCard, existing: GenericCard) -> bool:
        """Decide whether candidate's content should replace existing's.

        Private helper — single consumer is add(). Ported unchanged
        from the pre-refactor CardRegistry (compares raw_content, an
        exact tie MUST return False).

        Inputs:
            candidate: the newly-seen card being considered.
            existing: the card currently stored for this name.
        Output: True if candidate should replace existing's
            raw_content/provenance, False otherwise (including a tie).
        Side effects: none.
        Exceptions: none.
        """
        candidate_size = len(json.dumps(candidate.raw_content, sort_keys=True))
        existing_size = len(json.dumps(existing.raw_content, sort_keys=True))
        return candidate_size > existing_size

    def _store_card(self, card: GenericCard) -> None:
        """Insert or overwrite card in the uuid and name indices.

        Private helper — single consumer is add(). Ported unchanged from the
        pre-refactor CardRegistry. Does not touch
        AliasLedger — see register_alias(), called separately.

        Inputs:
            card: the card to store as canonical for its
                (source_game, name) and nocab_uuid.
        Output: none.
        Side effects: mutates this binder's uuid and name indices.
        Exceptions: none.
        """
        self._cards_by_uuid[card.nocab_uuid] = card
        self._uuid_by_name[(card.source_game, card.name)] = card.nocab_uuid

    @staticmethod
    def _alias_ledger_path(path: Path) -> Path:
        """Compute the sibling alias-ledger path for a main binder path.

        Private helper — shared by load() and save(). Renamed from
        _alias_path/".aliases" suffix — e.g.
        data/final/cards/mtg.jsonl -> data/final/cards/mtg.alias_ledger.jsonl.

        Inputs:
            path: a main binder file path (e.g.
                data/final/cards/mtg.jsonl).
        Output: the corresponding alias-ledger file path.
        Side effects: none — purely a path computation.
        Exceptions: none.
        """
        return path.with_suffix(".alias_ledger" + path.suffix)
