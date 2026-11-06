"""Translates one tcgjson catalog into stored cards, for any tcgjson game:
one subclass per game names its TcgjsonGameSpec.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface. The pipeline is generic: read_catalog
(catalog.py) -> group_printings -> build_card_content per group
(printing_groups.py) -> store. What differs per game is data, in its
spec (game_spec.py).

Why a subclass per game rather than a spec argument: the Protocol needs
SOURCE_GAME as a ClassVar, so the game must be known per class.
__init_subclass__ sets SOURCE_GAME from SPEC.game, so a subclass states
its game once. The bare base has no SPEC and is never registered.

Identity: one card per PrintingGroup (base name or card number; see
printing_groups.py). Every product id of the group is registered as a
DataSource.TCGJSON alias (TCGplayer's productId), on every branch, and
for a game whose cards are their card numbers, each of the group's
card_numbers as a DataSource.PRINTED_NUMBER alias, the ids its deck
lists use ("OP01-024"); Provenance.source_id is the representative
printing's product id.

The catalog is the game's whole card pool, so the binder mirrors it, as
in hearthstonejson/:
  - a group reuses the stored card its lowest (original) product id
    names, else the first any of its ids names, unless an earlier group
    of this same ingest already took that card (then it gets a new
    card);
  - after every group is stored, the game's cards no group took are
    deleted. The Unknown sentinel is never deleted.
A catalog with no card groups raises instead of emptying the binder.

Collision policy: merge_strategies.keep_incoming_if_content_differs - the
catalog passed in is authoritative.

Runtime (in memory, weekly-20260831): under 2 s per game, except
Yu-Gi-Oh (46k products) at about 5 s.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID, uuid4

from tqdm import tqdm

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.tcgjson.catalog import (
    TcgjsonCatalog,
    read_catalog,
)
from src.data_refinement.card_binder.tcgjson.game_spec import (
    DIGIMON_SPEC,
    LORCANA_SPEC,
    ONE_PIECE_SPEC,
    RIFTBOUND_SPEC,
    STAR_WARS_UNLIMITED_SPEC,
    UNION_ARENA_SPEC,
    YUGIOH_SPEC,
    TcgjsonGameSpec,
)
from src.data_refinement.card_binder.tcgjson.printing_groups import (
    PrintingGroup,
    build_card_content,
    group_printings,
    representative_printing,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_FULL_CATALOG_SUFFIX = ".full.json.gz"


class TcgjsonCardIngestionStage:
    """Translates one tcgjson catalog directly into binder (one subclass
    per game sets SPEC).

    Single-consumer to src/data_refinement/card_binder/ - no other
    container depends on this class directly.
    """

    SPEC: ClassVar[TcgjsonGameSpec]
    SOURCE_GAME: ClassVar[GameId]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Set SOURCE_GAME from the subclass's SPEC.

        Inputs: kwargs (passed on). Output: none.
        Side effects: sets cls.SOURCE_GAME.
        Exceptions: TypeError if the subclass sets no SPEC.
        """
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get("SPEC")
        if not isinstance(spec, TcgjsonGameSpec):
            raise TypeError(f"{cls.__name__} must set SPEC to a TcgjsonGameSpec")
        cls.SOURCE_GAME = spec.game

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Read one catalog and make binder's cards of this game mirror
        it: one card per printing group.

        Inputs:
            raw_path: a <slug>.full.json.gz, or a release directory
                holding <SPEC.slug>.full.json.gz.
            binder: the CardBinder to write into (as SOURCE_GAME).
        Output: nocab_uuid of every card created or content-changed
            (deleted cards are not listed).
        Side effects: reads raw_path; creates/replaces/deletes this
            game's cards and registers product-id and card-number
            aliases on binder;
            prints a tqdm progress bar.
        Exceptions: FileNotFoundError if the catalog is missing; whatever
            read_catalog raises; ValueError if the catalog has no card
            groups (rather than deleting every stored card).

        Example:
            >>> LorcanaTcgjsonCardIngestionStage().ingest(
            ...     Path("data/raw/tcgjson/20260831T174333Z"), CardBinder.load([]))
        """
        result: list[UUID] = []
        catalog = read_catalog(self._catalog_path(raw_path))
        groups = group_printings(catalog, self.SPEC)
        if not groups:
            raise ValueError(f"{raw_path} has no {self.SPEC.slug} card products")

        # Store every group, each on a card no earlier group has taken
        taken: set[UUID] = set()
        for group in tqdm(groups, desc=f"tcgjson {self.SPEC.slug}", unit="card"):
            stored_uuid, changed = self._store_group(group, catalog, binder, taken)
            taken.add(stored_uuid)
            if changed:
                result.append(stored_uuid)

        # Drop the cards this catalog no longer has
        self._delete_untaken_cards(binder, taken)
        return result

    def _catalog_path(self, raw_path: Path) -> Path:
        """raw_path itself, or <raw_path>/<SPEC.slug>.full.json.gz if it
        is a directory.
        Inputs: raw_path. Output: Path. Side effects: stats raw_path.
        Exceptions: none (a missing file fails in read_catalog)."""
        if raw_path.is_dir():
            return raw_path / f"{self.SPEC.slug}{_FULL_CATALOG_SUFFIX}"
        return raw_path

    def _store_group(
        self,
        group: PrintingGroup,
        catalog: TcgjsonCatalog,
        binder: CardBinder,
        taken: set[UUID],
    ) -> tuple[UUID, bool]:
        """Create-or-merge one card; register every product id and card
        number of the group as an alias on every branch.
        Inputs: group, catalog, binder, taken (uuids earlier groups
            stored into; read only).
        Output: (stored nocab_uuid, whether this call created or changed
            the card).
        Side effects: creates or replaces one card; registers aliases.
        Exceptions: none beyond CardBinder's own."""
        existing = self._existing_card(group, binder, taken)
        candidate = self._build_card(group, catalog)

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

        for product in group.products:
            binder.register_alias(
                self.SOURCE_GAME,
                DataSource.TCGJSON,
                str(product.product_id),
                stored_uuid,
            )
        for number in group.card_numbers:
            binder.register_alias(
                self.SOURCE_GAME, DataSource.PRINTED_NUMBER, number, stored_uuid
            )
        return stored_uuid, changed

    def _existing_card(
        self, group: PrintingGroup, binder: CardBinder, taken: set[UUID]
    ) -> GenericCard | None:
        """The stored card group should update: the first its product ids
        name, lowest id first, skipping cards already taken.
        Inputs: group, binder, taken. Output: GenericCard | None.
        Side effects: none. Exceptions: none."""
        for product in group.products:
            card = binder.get_by_alias(
                self.SOURCE_GAME, DataSource.TCGJSON, str(product.product_id)
            )
            if card is not None and card.nocab_uuid not in taken:
                return card
        return None

    def _build_card(self, group: PrintingGroup, catalog: TcgjsonCatalog) -> GenericCard:
        """A fresh GenericCard for group (a new nocab_uuid; a merge keeps
        the existing one): build_card_content, provenance from the
        representative printing.
        Inputs: group, catalog. Output: GenericCard.
        Side effects: none. Exceptions: none."""
        representative = representative_printing(group, catalog, self.SPEC)
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=group.name,
            raw_content=build_card_content(group, catalog, self.SPEC),
            provenance=Provenance(
                data_source=DataSource.TCGJSON,
                source_id=str(representative.product_id),
                fetched_at=datetime.now(timezone.utc),
            ),
        )

    def _delete_untaken_cards(self, binder: CardBinder, taken: set[UUID]) -> None:
        """Delete every SOURCE_GAME card no group took, except the
        Unknown sentinel.
        Inputs: binder, taken. Output: none.
        Side effects: deletes cards from binder.
        Exceptions: none beyond CardBinder's own."""
        keep = taken | {CardBinder.unknown_card_uuid(self.SOURCE_GAME)}
        stale = [
            uuid for uuid in binder.all_uuids(self.SOURCE_GAME) if uuid not in keep
        ]
        for uuid in stale:
            binder.delete(uuid)


class LorcanaTcgjsonCardIngestionStage(TcgjsonCardIngestionStage):
    """Disney Lorcana's cards from its tcgjson catalog (LORCANA_SPEC)."""

    SPEC = LORCANA_SPEC


class YugiohTcgjsonCardIngestionStage(TcgjsonCardIngestionStage):
    """Yu-Gi-Oh!'s cards from its tcgjson catalog (YUGIOH_SPEC)."""

    SPEC = YUGIOH_SPEC


class DigimonTcgjsonCardIngestionStage(TcgjsonCardIngestionStage):
    """The Digimon Card Game's cards from its tcgjson catalog (DIGIMON_SPEC)."""

    SPEC = DIGIMON_SPEC


class OnePieceTcgjsonCardIngestionStage(TcgjsonCardIngestionStage):
    """The One Piece Card Game's cards from its tcgjson catalog (ONE_PIECE_SPEC)."""

    SPEC = ONE_PIECE_SPEC


class StarWarsUnlimitedTcgjsonCardIngestionStage(TcgjsonCardIngestionStage):
    """Star Wars: Unlimited's cards from its tcgjson catalog (STAR_WARS_UNLIMITED_SPEC)."""

    SPEC = STAR_WARS_UNLIMITED_SPEC


class UnionArenaTcgjsonCardIngestionStage(TcgjsonCardIngestionStage):
    """Union Arena's cards from its tcgjson catalog (UNION_ARENA_SPEC)."""

    SPEC = UNION_ARENA_SPEC


class RiftboundTcgjsonCardIngestionStage(TcgjsonCardIngestionStage):
    """Riftbound's cards from its tcgjson catalog (RIFTBOUND_SPEC)."""

    SPEC = RIFTBOUND_SPEC
