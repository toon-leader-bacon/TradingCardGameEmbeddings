"""Translates spire-codex's run export into Slay the Spire 2 decks, directly
into a DeckBox.

RAW SHAPE: data/raw/spire_codex/runs/page_NNNNN.jsonl.gz, written by
src/data_retrieval/spire_codex/run_downloader.py - one run per line, the
game's own run record. Each row carries "run_hash" (a hex string, the
run's identity) and "players" (a list); players[i]["deck"] is a list of
{"id": "CARD.<NAME>", ...} entries, duplicates meaningful (copy counts).
Also read by src/data_refinement/metrics/sts2_runs/ (raw_files(), runs()
and deck_uuid()), whose deck-level metric rows point at the decks this
stage stores; changing either changes those metrics.

WHAT IS KEPT: one final deck per player of every run, by default. The
project favors data quantity over quality, so abandoned runs
(was_abandoned, ~16% of the 2026-09 export) are kept too, though a quit
run's deck is not a finished one and early quits are mostly starter
decks. keep_abandoned=False drops them; a finer filter (drop only runs
abandoned before the first boss) is noted in deck_box/TODO.md.

CARD ALIAS LOOKUP: "CARD." is stripped from each id, which is then looked
up as a spire_codex alias of the StS2 binder. A miss logs an error and
falls back to the game's Unknown sentinel card (seeded by
CardBinder.ensure_unknown_card()), so a deck keeps its size.

IDEMPOTENT RE-RUNS: a deck's identity is uuid5(DECK_NAMESPACE,
f"{run_id}:{player_index}"), so re-seeing a run updates its deck rather
than duplicating it, and a run with several players gets one distinct
deck per player. A re-seen deck whose card multiset is unchanged is not
reported as changed.

PROVENANCE: each created or updated deck carries a Provenance
(DataSource.SPIRE_CODEX, source_id=f"{run_id}:{player_index}"),
refreshed on every content-changing update.

STREAMING: pages are gzip-compressed NDJSON, read one line at a time.

HISTORY: this stage once subclassed a stage for sts2runs.com's dump
(deck data source STS2RUNS). That source is dead and its code is in
archive/sts2runs/. Its 6,796 decks remain in the StS2 deck box with
DataSource.STS2RUNS provenance, which is why that enum value stays.
"""

import gzip
import io
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Iterator
from uuid import UUID, uuid5

from tqdm import tqdm

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_retrieval.spire_codex.run_downloader import SpireCodexRunDownloader
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_PAGE_GLOB = "page_*.jsonl.gz"
_CARD_ID_PREFIX = "CARD."


class SpireCodexRunsDeckExtractionStage:
    """spire-codex run export pages -> final GenericDecks (StS2)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2
    DEFAULT_RAW_PATH: ClassVar[Path] = SpireCodexRunDownloader.DEFAULT_RAW_DATA_DIR
    RUN_ID_KEY: ClassVar[str] = "run_hash"
    DECK_DATA_SOURCE: ClassVar[DataSource] = DataSource.SPIRE_CODEX
    # Fixed, arbitrary - never regenerate (deck identity depends on it)
    DECK_NAMESPACE: ClassVar[UUID] = UUID("3b7e1d52-9c4a-4f06-b8d1-6e2a0c9f5d17")
    DECK_NAME_PREFIX: ClassVar[str] = "spire_codex run"

    def __init__(self, keep_abandoned: bool = True) -> None:
        """Inputs: keep_abandoned (store the decks of abandoned runs).
        Side effects: none. Exceptions: none."""
        self._keep_abandoned = keep_abandoned

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Create or update a deck on box for every player of every kept run.

        Inputs:
            raw_path: the pages directory, or one page file;
                DEFAULT_RAW_PATH when None.
            box: the DeckBox to create/update decks on. Every deck is
                stored under SOURCE_GAME.
            card_lookup: must already have SOURCE_GAME's Unknown sentinel
                card seeded, in addition to spire_codex's cards.
        Output: nocab_uuid of every (run, player) whose card list this
            call created or changed. A re-seen (run, player) with an
            identical card multiset is NOT included.
        Side effects: reads every page through runs() (one progress bar
            each); creates/updates decks on box; logs one error per card
            that falls back to the Unknown sentinel.
        Exceptions: FileNotFoundError as raw_files(); raises if a page is
            not valid gzip or NDJSON, or a line lacks RUN_ID_KEY or
            "players". RuntimeError if the Unknown sentinel is not on
            card_lookup.

        Example:
            >>> changed = SpireCodexRunsDeckExtractionStage().extract(
            ...     None, box, binder
            ... )
        """
        changed_uuids: list[UUID] = []
        for path in self.raw_files(raw_path):
            for row in self.runs(path):
                if self._keeps_run(row):
                    changed_uuids.extend(self._extract_run(row, box, card_lookup))
        return changed_uuids

    def raw_files(self, raw_path: Path | None) -> list[Path]:
        """Every page file under raw_path, in name order (or raw_path
        itself when it is one page file).

        Inputs: raw_path (the pages directory, or one page file;
            DEFAULT_RAW_PATH when None).
        Output: list[Path] of page files.
        Side effects: lists the directory.
        Exceptions: FileNotFoundError if raw_path is missing or holds no
            page file.

        Example:
            >>> SpireCodexRunsDeckExtractionStage().raw_files(None)[0].name
            'page_00000.jsonl.gz'
        """
        path = raw_path or self.DEFAULT_RAW_PATH
        pages = [path] if path.is_file() else sorted(path.glob(_PAGE_GLOB))
        if not pages:
            raise FileNotFoundError(f"no {_PAGE_GLOB} run pages at {path}")
        return pages

    def runs(self, path: Path) -> Iterator[dict]:
        """Every run in one gzip-compressed NDJSON page, parsed, in file
        order (blank lines skipped). Also read by the sts2_runs metrics,
        so the raw file format is read in one place.

        Inputs: path (Path), one file from raw_files().
        Output: Iterator[dict], one raw run JSON object per line.
        Side effects: reads path one line at a time, decompressing as it
            goes. Prints a tqdm progress bar to stderr over path's
            COMPRESSED size, advanced by compressed bytes consumed, so it
            reaches exactly 100% at EOF.
        Exceptions: raises if path doesn't exist, isn't a valid gzip
            file, or a line isn't valid JSON.

        Example:
            >>> next(stage.runs(Path("data/raw/spire_codex/runs/page_00000.jsonl.gz")))[
            ...     "players"
            ... ]
        """
        total_compressed_bytes = path.stat().st_size
        # Wrap the compressed file ourselves so compressed_file.tell()
        # tracks compressed bytes consumed; GzipFile.tell() would report
        # the decompressed position, whose total is unknown up front.
        with open(path, "rb") as compressed_file, tqdm(
            total=total_compressed_bytes,
            unit="B",
            unit_scale=True,
            desc=f"{self.DECK_DATA_SOURCE.value} runs: {path.name}",
        ) as progress:
            with io.TextIOWrapper(
                gzip.GzipFile(fileobj=compressed_file), encoding="utf-8"
            ) as raw_file:
                bytes_read = 0
                for line in raw_file:
                    position = compressed_file.tell()
                    progress.update(position - bytes_read)
                    bytes_read = position

                    if line.strip():
                        yield json.loads(line)

    def deck_uuid(self, run_id: int | str, player_index: int) -> UUID:
        """The deterministic nocab_uuid for one (run, player) pair.

        Public so the sts2_runs metrics can point at the deck this stage
        stored for a run without re-deriving the namespace. uuid5 (not
        uuid4): the same pair always gives the same uuid, so a re-seen
        pair updates rather than duplicates.

        Inputs:
            run_id: one run's RUN_ID_KEY field; an int and its str give
                the same uuid.
            player_index: that player's position in the run's "players".
        Output: a uuid stable forever for the pair.
        Side effects: none. Exceptions: none.

        Example:
            >>> stage.deck_uuid("ab12", 0) == stage.deck_uuid("ab12", 0)
            True
        """
        return uuid5(self.DECK_NAMESPACE, f"{run_id}:{player_index}")

    def _keeps_run(self, row: dict) -> bool:
        """False for an abandoned run when keep_abandoned is off (see the
        module docstring).
        Inputs: row. Output: bool. Side effects: none. Exceptions: none."""
        return self._keep_abandoned or not row.get("was_abandoned", False)

    def _extract_run(
        self, row: dict, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Create-or-update every player's deck from one run. This is where
        idempotency happens: each deck_uuid is a pure function of
        (row[RUN_ID_KEY], player_index).

        Inputs:
            row: one parsed run, carrying RUN_ID_KEY and "players" (each
                player carrying "deck", a list of {"id": str, ...}).
            box: the DeckBox to read from and write to.
            card_lookup: used to look up each card id (see _card_uuid()).
        Output: deck_uuid for every player whose deck was created or whose
            update changed stored content.
        Side effects: creates or updates one deck per player, with a fresh
            Provenance.
        Exceptions: raises if row lacks RUN_ID_KEY or "players", or a
            player lacks "deck". Whatever _card_uuid() raises propagates.
        """
        run_id = row[self.RUN_ID_KEY]
        changed_uuids: list[UUID] = []

        # One deck per player; "players" is a real list
        for player_index, player in enumerate(row["players"]):
            deck_uuid = self.deck_uuid(run_id, player_index)
            card_nocab_uuids = [
                self._card_uuid(card_entry["id"], card_lookup)
                for card_entry in player["deck"]
            ]
            provenance = Provenance(
                data_source=self.DECK_DATA_SOURCE,
                source_id=f"{run_id}:{player_index}",
                fetched_at=datetime.now(timezone.utc),
            )

            existing = box.get_by_uuid(deck_uuid)
            if existing is None:
                box.create(
                    GenericDeck(
                        nocab_uuid=deck_uuid,
                        source_game=self.SOURCE_GAME,
                        name=f"{self.DECK_NAME_PREFIX} {run_id} player {player_index}",
                        card_nocab_uuids=card_nocab_uuids,
                        provenance=provenance,
                    )
                )
                changed_uuids.append(deck_uuid)
            # A multiset: compare copy counts, never list or set equality
            elif Counter(existing.card_nocab_uuids) != Counter(card_nocab_uuids):
                box.update(
                    deck_uuid, card_nocab_uuids=card_nocab_uuids, provenance=provenance
                )
                changed_uuids.append(deck_uuid)

        return changed_uuids

    def _card_uuid(self, raw_card_id: str, card_lookup: CardLookup) -> UUID:
        """The nocab_uuid for one deck entry's card id.

        Inputs: raw_card_id (an entry's "id", e.g. "CARD.SETUP_STRIKE").
        Output: the matching nocab_uuid, or on a miss the Unknown
            sentinel's.
        Side effects: logs one error on a miss.
        Exceptions: RuntimeError if the Unknown sentinel is not on
            card_lookup.
        """
        card = card_lookup.get_by_alias(
            self.SOURCE_GAME,
            DataSource.SPIRE_CODEX,
            raw_card_id.removeprefix(_CARD_ID_PREFIX),
        )
        if card is not None:
            return card.nocab_uuid

        _logger.error(
            "SpireCodexRunsDeckExtractionStage: unresolved card id %r - "
            "substituting the Unknown sentinel card",
            raw_card_id,
        )
        unknown_card = card_lookup.get_by_name_single(
            self.SOURCE_GAME, CardBinder.UNKNOWN_CARD_NAME, strict=False
        )
        if unknown_card is None:
            raise RuntimeError(
                f"SpireCodexRunsDeckExtractionStage: {self.SOURCE_GAME!r}'s "
                "Unknown sentinel card is not seeded - call "
                "CardBinder.ensure_unknown_card() before extract()"
            )
        return unknown_card.nocab_uuid
