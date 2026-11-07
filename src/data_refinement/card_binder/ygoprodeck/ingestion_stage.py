"""Registers YGOPRODeck passcodes as aliases of the Yu-Gi-Oh binder's cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface. This stage creates and changes no card:
the Yu-Gi-Oh cards come from tcgjson (card_binder/tcgjson/), and this
only teaches the binder each card's passcodes, the id every Yu-Gi-Oh
deck source names cards by (YGOPRODeck decks, EDOPro/YGOPro .ydk files,
Master Duel Meta). The data: src/data_retrieval/huggingface_datasets/
ygoprodeck_repos.py (feregrino/yugioh-cards, CC-BY-SA-3.0).

RAW SHAPE: raw_path is the yugioh-cards mirror directory, complete
(HuggingFaceDatasetDownloader.is_complete_mirror), holding:
    cards.csv: one row per YGOPRODeck card; only "id" (the passcode, or
        an id YGOPRODeck minted for a card without one) and "name" are
        read.
    cards_variants.csv: original,variant: alternate-artwork passcodes,
        which decks use but cards.csv doesn't list.

MATCHING (CardNameIndex): a cards.csv name is compared with the binder's
names under three keys in turn (_NAME_KEYS): the exact name, the
casefolded name, then its ASCII letters and digits only (after
decomposing accents), since TCGplayer drops the symbols YGOPRODeck
keeps ("Maliss <P> Chessy Cat" / "Maliss P Chessy Cat", "Cupsy☆Yummy" /
"CupsyYummy", "Mädchen" / "Madchen"). The first key with any candidate
decides: exactly one card is a match, several are no match (never a
looser key's guess). The Unknown sentinel is never a candidate.
Measured 2026-11-06: 13,932 of 14,653 names match (46 by casefold, 56
by letters and digits); the misses are OCG-only and not-yet-TCG cards,
tokens, skill cards and some TCG cards TCGplayer's catalog lacks.

ALIASES: register_alias(YUGIOH, YGOPRODECK, passcode_id(id), uuid) for
each matched row, and for each variant whose original matched, the
variant's id to the same card.

STALENESS: AliasLedger is last-write-wins with no unregister, and a
tcgjson re-run can mint a new uuid for a card. Every run therefore
re-registers every matched passcode, repointing it at the card's
current uuid. A passcode that matched before and has no single
candidate now keeps its old alias: to a deleted card it reads as no
match (get_by_alias -> None, indistinguishable from never aliased); to
a live card it stays that card. The run logs how many unmatched
passcodes keep a live earlier alias.

RUN ORDER: after tcgjson_yugioh, against the same binder file;
scripts/run_card_binder_ingestion.py runs this straight after it.

RETURN VALUE: [] (no card is created or content-changed; an alias is
not a content change, per CardIngestionStage.ingest()).

PRECONDITION: binder holds Yu-Gi-Oh cards; raises ValueError otherwise.
"""

import csv
import logging
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterator
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.card_binder.ygoprodeck.passcode import passcode_id
from src.data_retrieval.huggingface_datasets.downloader import (
    HuggingFaceDatasetDownloader,
)
from src.data_retrieval.huggingface_datasets.ygoprodeck_repos import (
    YGOPRODECK_CARDS_REPO_ID,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_CARDS_FILENAME = "cards.csv"
_VARIANTS_FILENAME = "cards_variants.csv"
_UNMATCHED_LOG_LIMIT = 30


@dataclass(frozen=True)
class YgoprodeckCardRow:
    """One cards.csv row, as far as this stage reads it: the alias id
    (passcode_id of "id") and the card's name."""

    passcode: str
    name: str


@dataclass(frozen=True)
class PasscodeVariant:
    """One cards_variants.csv row: an alternate-artwork passcode and the
    original passcode it prints, both as alias ids."""

    original: str
    variant: str


class CardNameIndex:
    """Yu-Gi-Oh card name -> the one binder card it names, if exactly
    one (see the module docstring's MATCHING), built once from the
    binder.

    Single-consumer to YgoprodeckPasscodeAliasStage.
    """

    def __init__(self, card_lookup: CardLookup) -> None:
        """
        Inputs: card_lookup (the Yu-Gi-Oh binder).
        Output: none (constructor).
        Side effects: reads every Yu-Gi-Oh card once (one index per
            _NAME_KEYS key, and card_count).
        Exceptions: none.
        """
        unknown_uuid = CardBinder.unknown_card_uuid(GameId.YUGIOH)
        self._uuids_by_key: list[dict[str, list[UUID]]] = [
            defaultdict(list) for _ in _NAME_KEYS
        ]
        self._card_count = 0
        for card in card_lookup.all_cards(GameId.YUGIOH):
            if card.nocab_uuid == unknown_uuid:
                continue
            self._card_count += 1
            for name_key, index in zip(_NAME_KEYS, self._uuids_by_key):
                key = name_key(card.name)
                if key:  # a name of symbols only has no letters-and-digits key
                    index[key].append(card.nocab_uuid)

    @property
    def card_count(self) -> int:
        """How many Yu-Gi-Oh cards the index holds, the Unknown sentinel
        excluded (0 means tcgjson_yugioh hasn't run).

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return self._card_count

    def card_uuid(self, name: str) -> UUID | None:
        """The uuid of the one card name names, or None for no or
        several candidates.

        Inputs: name. Output: UUID | None. Side effects: none.
        Exceptions: none.

        Example:
            >>> CardNameIndex(binder).card_uuid("Dark Magician")
            UUID('...')
        """
        # The first key with any candidate decides
        for name_key, index in zip(_NAME_KEYS, self._uuids_by_key):
            key = name_key(name)
            candidates = index.get(key, []) if key else []
            if candidates:
                return candidates[0] if len(candidates) == 1 else None
        return None


def _letters_and_digits(name: str) -> str:
    """name casefolded, accents decomposed, keeping ASCII letters and
    digits only ("Maliss <P> Chessy Cat" -> "malisspchessycat").

    Inputs: name. Output: str (may be "" for a name of symbols only).
    Side effects: none. Exceptions: none.
    """
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(ch for ch in decomposed if ch.isascii() and ch.isalnum())


# CardNameIndex's keys, strictest first (see the module docstring)
_NAME_KEYS: tuple[Callable[[str], str], ...] = (
    lambda name: name,
    str.casefold,
    _letters_and_digits,
)


class YgoprodeckPasscodeAliasStage:
    """YGOPRODeck's cards.csv -> DataSource.YGOPRODECK aliases on the
    Yu-Gi-Oh binder's existing cards.

    Single-consumer to src/data_refinement/card_binder/.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.YUGIOH
    DEFAULT_RAW_PATH: ClassVar[Path] = HuggingFaceDatasetDownloader.raw_data_dir_for(
        YGOPRODECK_CARDS_REPO_ID
    )

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Register every matched passcode (and its variants) as an
        alias of its binder card.

        Inputs: raw_path (the yugioh-cards mirror directory), binder
            (the Yu-Gi-Oh binder, already holding tcgjson's cards).
        Output: [] (no card created or changed).
        Side effects: reads cards.csv and cards_variants.csv; registers
            aliases on binder; logs a summary (matched, unmatched,
            variants, stale aliases) and the most common unmatched
            names.
        Exceptions: FileNotFoundError if raw_path isn't a complete
            mirror or a file is missing; ValueError if binder holds no
            Yu-Gi-Oh card or a row's id isn't a passcode.

        Example:
            >>> YgoprodeckPasscodeAliasStage().ingest(
            ...     YgoprodeckPasscodeAliasStage.DEFAULT_RAW_PATH, binder)
            []
        """
        result: list[UUID] = []

        # Validate inputs
        HuggingFaceDatasetDownloader.require_complete_mirror(raw_path)
        names = CardNameIndex(binder)
        if names.card_count == 0:
            raise ValueError(
                "the Yu-Gi-Oh binder holds no cards: run tcgjson_yugioh first"
            )

        # Alias each passcode whose name names one card
        uuid_by_passcode: dict[str, UUID] = {}
        unmatched_names: Counter[str] = Counter()
        kept_alias_count = 0
        for row in self._read_cards(raw_path / _CARDS_FILENAME):
            card_uuid = names.card_uuid(row.name)
            if card_uuid is None:
                unmatched_names[row.name] += 1
                if self._has_live_alias(row.passcode, binder):
                    kept_alias_count += 1
                continue
            uuid_by_passcode[row.passcode] = card_uuid
            binder.register_alias(
                self.SOURCE_GAME, DataSource.YGOPRODECK, row.passcode, card_uuid
            )

        # Alias each alternate artwork to its original's card
        variant_count = self._register_variants(
            self._read_variants(raw_path / _VARIANTS_FILENAME),
            uuid_by_passcode,
            binder,
        )

        self._log_summary(
            len(uuid_by_passcode), variant_count, unmatched_names, kept_alias_count
        )
        return result

    def _read_cards(self, cards_path: Path) -> Iterator[YgoprodeckCardRow]:
        """Every cards.csv row, parsed.

        Inputs: cards_path. Output: rows, in file order.
        Side effects: reads the file lazily.
        Exceptions: FileNotFoundError; ValueError if a row's id isn't a
            passcode or the header lacks id or name.
        """
        for row in _csv_rows(cards_path, ("id", "name")):
            yield YgoprodeckCardRow(
                passcode=passcode_id(row["id"]), name=row["name"].strip()
            )

    def _read_variants(self, variants_path: Path) -> list[PasscodeVariant]:
        """Every cards_variants.csv row, parsed.

        Inputs: variants_path. Output: the variants, in file order.
        Side effects: reads the file.
        Exceptions: FileNotFoundError; ValueError if an id isn't a
            passcode or the header lacks original or variant.
        """
        return [
            PasscodeVariant(
                original=passcode_id(row["original"]),
                variant=passcode_id(row["variant"]),
            )
            for row in _csv_rows(variants_path, ("original", "variant"))
        ]

    def _register_variants(
        self,
        variants: list[PasscodeVariant],
        uuid_by_passcode: dict[str, UUID],
        binder: CardBinder,
    ) -> int:
        """Alias each variant whose original matched to the original's
        card.

        Inputs: variants, uuid_by_passcode (this run's matches),
            binder. Output: how many variants were aliased.
        Side effects: registers aliases on binder.
        Exceptions: none.
        """
        result = 0
        for variant in variants:
            card_uuid = uuid_by_passcode.get(variant.original)
            if card_uuid is None:
                continue
            binder.register_alias(
                self.SOURCE_GAME, DataSource.YGOPRODECK, variant.variant, card_uuid
            )
            result += 1
        return result

    def _has_live_alias(self, passcode: str, binder: CardBinder) -> bool:
        """Whether passcode already aliases a live card (an earlier
        run's match this run can't repeat).

        Inputs: passcode (alias id), binder. Output: bool.
        Side effects: none. Exceptions: none.
        """
        existing = binder.get_by_alias(
            self.SOURCE_GAME, DataSource.YGOPRODECK, passcode
        )
        return existing is not None

    def _log_summary(
        self,
        matched_count: int,
        variant_count: int,
        unmatched_names: Counter[str],
        kept_alias_count: int,
    ) -> None:
        """Log the counts at INFO and the most common unmatched names at
        WARNING. kept_alias_count covers cards.csv passcodes only: a
        variant of an unmatched original isn't checked.

        Inputs: the counts and unmatched_names. Output: none.
        Side effects: log lines. Exceptions: none.
        """
        _logger.info(
            "YGOPRODeck passcodes: %d aliased, %d variants aliased, %d names "
            "unmatched (%d of them keep an earlier run's alias)",
            matched_count,
            variant_count,
            sum(unmatched_names.values()),
            kept_alias_count,
        )
        if unmatched_names:
            _logger.warning(
                "YGOPRODeck names with no single binder card: %s",
                unmatched_names.most_common(_UNMATCHED_LOG_LIMIT),
            )


def _csv_rows(
    path: Path, required_columns: tuple[str, ...]
) -> Iterator[dict[str, str]]:
    """path's rows as dicts, after checking its header has
    required_columns; a short row's missing cells read as "".

    Inputs: path, required_columns. Output: rows, lazily.
    Side effects: reads path.
    Exceptions: FileNotFoundError if path is missing; ValueError if the
        header lacks a required column.
    """
    with open(path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [c for c in required_columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: no {', '.join(missing)} column")
        for row in reader:
            yield {column: row.get(column) or "" for column in required_columns}
