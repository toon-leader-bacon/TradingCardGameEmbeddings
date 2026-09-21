"""Translates 17Lands' game_data CSVs into full constructed decks, directly into a DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface this implements, and
src/data_refinement/deck_box/sts_gg/extraction_stage.py for the sibling
this is structurally modeled on (idempotent uuid5-per-composite-key
identity, Unknown-sentinel-on-miss policy) and
src/data_refinement/deck_box/sts2runs/extraction_stage.py (tqdm
streaming over one large file).

RAW SHAPE: data/raw/17lands/game_data/<Expansion>.<FormatCode>.csv —
CONFIRMED against the live files (~133 of them, up to several GB each).
Unlike every other DeckExtractionStage in this container, there is no
per-row list of raw card entries. Instead, each CSV's HEADER carries
one "deck_<CardName>" column per card ever legal for that set/format,
and each row's value under that column is that card's COPY COUNT in
this game's constructed deck (deck_<name> sums to 40 across a row) —
the same column shape
src/data_refinement/metrics/seventeenlands/game_data/game_card_columns.py's
GameCardColumns already parses for that sibling container's metrics.
Because a header-derived count, not a presence flag, is this source's
only signal, extraction here expands each present deck_<name> column
into that many repeated nocab_uuid entries in the resulting
GenericDeck.card_nocab_uuids — this is what makes the result a genuine
FULL deck list (see src/data_refinement/deck_box/README.md's
multiset/copy-count section), not merely which cards were present.

CARD RESOLUTION: each deck_<name> column's bare <name> suffix is
matched against card_lookup the same way
GameCardColumns._match_uuid() already does for that sibling metrics
container: an exact card_lookup.get_by_name(self.SOURCE_GAME, name) match
if it returns exactly one card; else a
card_lookup.get_by_name_regex(self.SOURCE_GAME, f"^{re.escape(name)}( //.*)?$")
front-face fallback (a split/MDFC card is stored under its combined
name) if that returns exactly one card; else unresolved. THIS POLICY IS
DELIBERATELY RE-IMPLEMENTED HERE, not imported from
game_card_columns.py — deck_box never imports from metrics/ anywhere in
this codebase, and metrics/ itself already re-types this same policy
per raw source rather than sharing it (see that module's own
docstring); this file follows that established precedent one level
further, onto a second container entirely.

Because a set's legal card pool differs per file, the deck_<name>
column index is rebuilt fresh for every CSV — never shared or cached
across files.

UNRESOLVED CARDS: same policy as every sibling stage in this
container — an unresolved column substitutes the Unknown sentinel
card's nocab_uuid (at that row's own copy count, not just once) rather
than dropping the slot, logged loudly via the stdlib `logging` module.
PRECONDITION: card_lookup must already have GameId.MTG's Unknown
sentinel card seeded (CardBinder.ensure_unknown_card(self.SOURCE_GAME),
called once against a real CardBinder before any extract() call) —
this stage stays read-only (CardLookup, never a full CardBinder) and
does NOT create that card lazily; a missing sentinel raises
RuntimeError.

PER-GAME IDENTIFIER: game_data has no single unique-id column — this
project's already-established composite for this exact raw source
(see metrics/seventeenlands/game_data/README.md's own "per-game
identifier" section) is (draft_id: str, match_number: int, game_number:
int), read directly off each row. Deck identity is
uuid5(_DECK_NAMESPACE, f"{draft_id}:{match_number}:{game_number}") —
deterministic, so re-running extraction over the same CSV(s) updates
rather than duplicates a deck, same IDEMPOTENT RE-RUNS convention as
every sibling stage (box.get_by_uuid() first; box.create() on a
genuinely new composite key; box.update() only when the resolved
Counter(card_nocab_uuids) actually changed; a no-op otherwise). A
draft_id is assumed globally unique across every game_data CSV in this
directory (17Lands' own convention) — this stage never folds the
source file's expansion/format into the hashed identity, only into the
stored deck's human-readable name.

TAR-WRAPPED FILES, CONFIRMED against the live files: 10 of the 133
game_data CSVs (every AFR/KHM/MID/STX/VOW format) are not plain CSV on
disk — a since-fixed bug in SeventeenLandsDownloader.download_one()
(src/data_retrieval/seventeenlands/downloader.py) wrote these ones'
decompressed bytes straight to disk without noticing the decompressed
stream was itself a POSIX tar archive (wrapping one CSV member), not a
plain CSV. Future downloads are fixed at the source, but these 10
files, already on disk, still need to be read correctly without a
network re-fetch — extract() detects this per file (peeking for the
ustar magic at header byte offset 257, the same technique
download_one() now uses) and transparently reads through the tar
member instead of the raw file when needed. The on-disk quirk here is
already-decompressed (unlike download_one()'s in-flight gzip stream),
so the extracted tar member is a plain seekable file, not a
forward-only stream.

STREAMING: every file here is read in chunks (via pandas' own
chunksize, mirroring
src/data_refinement/metrics/seventeenlands/game_data/scanner.py's
scan_game_csv()) — the full-file-in-memory approach that function's own
docstring warns against is never used here either, tar-wrapped or not.

DIRECTORY HANDLING: raw_path may be a single CSV file or a directory of
them (see extraction.py's own documented flexibility) — DEFAULT_RAW_PATH
is SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data", the
whole directory. A directory's *.csv files are processed in sorted
order, each contributing independently to the one returned
changed_uuids list.

PROVENANCE: each created/updated deck carries a Provenance
(DataSource.SEVENTEENLANDS_GAME_DATA,
source_id=f"{draft_id}:{match_number}:{game_number}", fetched_at=now)
— this stage's own deck identity (see PER-GAME IDENTIFIER above), not
SCRYFALL (that's each card's own data source, used only for card
resolution). Refreshed on every content-changing update(), not just on
first create().
"""

import logging
import re
import tarfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, ClassVar, Iterable, Iterator
from uuid import UUID, uuid5

import pandas as pd
from tqdm import tqdm

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_retrieval.seventeenlands.downloader import SeventeenLandsDownloader
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_DECK_COLUMN_PREFIX = "deck_"

# I/O-efficiency knob only (see scan_game_csv()'s own chunk_size
# parameter for precedent) - never changes what _extract_row() sees,
# still one row at a time.
_CHUNK_SIZE = 100_000

# POSIX ustar header layout: the "ustar" magic sits at byte offset 257
# of every 512-byte tar header - same constant SeventeenLandsDownloader
# now checks at download time (see module docstring's TAR-WRAPPED FILES
# section for why this is re-checked here rather than assumed fixed).
_TAR_MAGIC_OFFSET = 257
_TAR_MAGIC = b"ustar"

# Fixed, arbitrary — never regenerate. Namespace for this stage's
# deterministic per-(draft_id, match_number, game_number) deck uuids
# (see module docstring's PER-GAME IDENTIFIER section). Deliberately
# distinct from every other uuid5 namespace already minted in this
# codebase (e.g. sts_gg/sts2runs/play_gwent's own _DECK_NAMESPACE
# constants).
_DECK_NAMESPACE = UUID("a3d8f1c2-4e5b-4a6c-9d7e-2f8b1c3a5e6f")


class SeventeenLandsGameDataDeckExtractionStage:
    """Translates 17Lands' game_data CSVs directly into a DeckBox.

    Single-consumer to src/data_refinement/deck_box/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data"
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Parse every game_data CSV under raw_path, creating/updating decks on box.

        One _extract_csv() call per CSV file — no additional logic
        beyond calling that (in sorted order, when raw_path is a
        directory) and concatenating the results.

        Inputs:
            raw_path: path to one game_data CSV, or a directory of them
                (e.g. data/raw/17lands/game_data/). Defaults to
                DEFAULT_RAW_PATH when None.
            box: the DeckBox to create/update decks on, as a side
                effect. Every deck this call touches is stored under
                self.SOURCE_GAME.
            card_lookup: must already have self.SOURCE_GAME's Unknown
                sentinel card seeded (see module docstring's
                PRECONDITION).
        Output: nocab_uuid of every (draft_id, match_number,
            game_number) whose resolved deck this call created or
            changed, across every processed file. A re-seen composite
            key whose resolved list is identical to what's already
            stored is NOT included.
        Side effects: reads every processed CSV (streamed, never fully
            in memory — see module docstring's STREAMING section);
            creates/updates decks directly on box; emits one
            logging.error() per column name that falls back to the
            Unknown sentinel; prints one tqdm progress bar per file to
            stderr.
        Exceptions: raises if raw_path doesn't exist, a file isn't
            parsable as CSV (or as a tar-wrapped CSV — see module
            docstring's TAR-WRAPPED FILES section), or a row is missing
            draft_id/match_number/game_number. Raises RuntimeError if
            self.SOURCE_GAME's Unknown sentinel card isn't found on
            card_lookup.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> box = DeckBox.load([Path("data/final/decks/mtg.jsonl")])
            >>> stage = SeventeenLandsGameDataDeckExtractionStage()
            >>> changed_uuids = stage.extract(None, box, binder)
        """
        path = raw_path if raw_path is not None else self.DEFAULT_RAW_PATH
        csv_paths = sorted(path.glob("*.csv")) if path.is_dir() else [path]

        changed_uuids: list[UUID] = []
        # One file at a time - each gets its own header-derived deck
        # column index (a set's legal card pool differs per file, see
        # module docstring) and its own tqdm progress bar.
        for csv_path in csv_paths:
            changed_uuids.extend(self._extract_csv(csv_path, box, card_lookup))
        return changed_uuids

    def _extract_csv(
        self, csv_path: Path, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Stream one game_data CSV (plain or tar-wrapped), creating/updating decks on box.

        Private helper — single consumer is extract(). Opens csv_path
        exactly once via _open_stream() (tar-aware): reads the header
        first (to build this file's own deck column index), seeks back
        to the start, then re-reads the same stream in chunks for the
        real pass — see _open_stream()'s own docstring for why this
        works uniformly for a tar-wrapped member and a plain file.

        Inputs:
            csv_path: one game_data CSV file.
            box: the DeckBox to read from and write to.
            card_lookup: used to resolve every deck_<name> column (see
                _deck_column_index()).
        Output: deck_uuid for every row whose create() was new or whose
            update() actually changed stored content, in this file
            alone.
        Side effects: reads csv_path; creates/updates decks on box;
            prints one tqdm progress bar to stderr, sized against this
            file's own content length (the tar member's size, when
            tar-wrapped — not the compressed/wrapped file's own size on
            disk).
        Exceptions: raises if csv_path isn't parsable as CSV (or
            tar-wrapped CSV), or a row is missing
            draft_id/match_number/game_number. Whatever
            _extract_row()/_deck_column_index() raise propagates.
        """
        changed_uuids: list[UUID] = []

        with self._open_stream(csv_path) as (binary_stream, total_bytes):
            header_columns = pd.read_csv(binary_stream, nrows=0).columns
            binary_stream.seek(0)
            deck_columns = self._deck_column_index(header_columns, card_lookup)

            with tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=f"seventeenlands_game_data extract: {csv_path.name}",
            ) as progress:
                bytes_read = 0
                for chunk in pd.read_csv(binary_stream, chunksize=_CHUNK_SIZE):
                    position = binary_stream.tell()
                    progress.update(position - bytes_read)
                    bytes_read = position

                    # Hand every row to _extract_row() one at a time
                    # (never the whole chunk at once).
                    for row in chunk.to_dict(orient="records"):
                        result = self._extract_row(row, box, card_lookup, deck_columns)
                        if result is not None:
                            changed_uuids.append(result)

        return changed_uuids

    def _extract_row(
        self,
        row: dict,
        box: DeckBox,
        card_lookup: CardLookup,
        deck_columns: list[tuple[str, UUID]],
    ) -> UUID | None:
        """Create-or-update one game_data row's deck directly against box.

        Private helper — single consumer is _extract_csv(). THIS METHOD
        IS WHERE IDEMPOTENCY HAPPENS: deck_uuid is a pure function of
        (row["draft_id"], row["match_number"], row["game_number"]) (see
        module docstring's PER-GAME IDENTIFIER section), so re-processing
        the same row always targets the same stored deck.

        Inputs:
            row: one game_data CSV row, dict-like — carrying at least
                draft_id/match_number/game_number/expansion/event_type,
                plus this file's deck_<name> columns.
            box: the DeckBox to read from and write to.
            card_lookup: used to resolve unresolved deck_<name> columns'
                Unknown-sentinel fallback (see
                _card_nocab_uuids_for_row()'s own docstring — the
                columns themselves are already resolved into
                deck_columns by the time this method runs).
            deck_columns: this CSV's own deck_<name> column index (see
                _deck_column_index()).
        Output: deck_uuid if this call caused a create() or an actual
            content-changing update(); None if this row matched an
            already-stored deck with an identical resolved card list.
        Side effects: creates or updates exactly one deck on box, with
            a freshly-computed Provenance (see module docstring's
            PROVENANCE section) — set on create(), and refreshed on a
            content-changing update() too.
        Exceptions: raises if row is missing
            draft_id/match_number/game_number. Whatever
            _card_nocab_uuids_for_row() raises propagates.
        """
        draft_id = row.get("draft_id", "unknown")
        match_number = row.get("match_number", -1)
        game_number = row.get("game_number", -1)
        deck_uuid = self._deck_uuid(
            draft_id, 
            match_number, 
            game_number
        )
        card_nocab_uuids = self._card_nocab_uuids_for_row(row, deck_columns)
        provenance = Provenance(
            data_source=DataSource.SEVENTEENLANDS_GAME_DATA,
            source_id=f"{draft_id}:{match_number}:{game_number}",
            fetched_at=datetime.now(timezone.utc),
        )

        existing = box.get_by_uuid(deck_uuid)
        if existing is None:
            # Brand-new (draft_id, match_number, game_number).
            box.create(
                GenericDeck(
                    nocab_uuid=deck_uuid,
                    source_game=self.SOURCE_GAME,
                    name=self._deck_name(row),
                    card_nocab_uuids=card_nocab_uuids,
                    provenance=provenance,
                )
            )
            return deck_uuid

        # card_nocab_uuids is a multiset - compare copy counts via
        # Counter, never list/set equality (same reasoning as every
        # sibling stage's own _extract_row/_extract_run).
        if Counter(existing.card_nocab_uuids) != Counter(card_nocab_uuids):
            box.update(
                deck_uuid, card_nocab_uuids=card_nocab_uuids, provenance=provenance
            )
            return deck_uuid

        # Re-seen row, identical resolved multiset - no-op.
        return None

    def _deck_column_index(
        self, header_columns: Iterable[str], card_lookup: CardLookup
    ) -> list[tuple[str, UUID]]:
        """Match every "deck_<name>" column in header_columns to a nocab_uuid.

        Private helper — single consumer is _extract_csv(). Every
        distinct <name> is resolved at most once per call (a local
        cache), via _card_uuid_for_name() — mirrors
        GameCardColumns.from_header()'s own column-parsing shape (see
        module docstring's CARD RESOLUTION section) without importing
        it.

        Inputs:
            header_columns: this CSV's own header (e.g.
                pandas.read_csv(path, nrows=0).columns) — every column,
                not just deck_-prefixed ones; non-matching columns are
                ignored.
            card_lookup: registry each distinct <name> is resolved
                against.
        Output: every "deck_<name>" column paired with its resolved (or
            Unknown-sentinel-substituted — see module docstring's
            UNRESOLVED CARDS section) nocab_uuid, in header_columns' own
            order.
        Side effects: emits one logging.error() per distinct
            unresolved <name> (via _card_uuid_for_name()).
        Exceptions: raises RuntimeError if self.SOURCE_GAME's Unknown
            sentinel card isn't found on card_lookup.
        """
        name_cache: dict[str, UUID] = {}
        deck_columns: list[tuple[str, UUID]] = []
        for column in header_columns:
            if not column.startswith(_DECK_COLUMN_PREFIX):
                continue
            name = column[len(_DECK_COLUMN_PREFIX) :]
            if name not in name_cache:
                name_cache[name] = self._card_uuid_for_name(name, card_lookup)
            deck_columns.append((column, name_cache[name]))
        return deck_columns

    def _card_uuid_for_name(self, name: str, card_lookup: CardLookup) -> UUID:
        """Resolve one bare card name to a nocab_uuid, falling back to Unknown.

        Private helper — single consumer is _deck_column_index(). Exact
        match first, then a split/MDFC front-face regex fallback, then
        the Unknown sentinel — see module docstring's CARD RESOLUTION
        and UNRESOLVED CARDS sections for the exact policy and why it's
        re-implemented here rather than imported.

        Inputs:
            name: one deck_<name> column's bare <name> suffix.
            card_lookup: registry to resolve name against.
        Output: the matching nocab_uuid, or (on a miss) the Unknown
            sentinel's nocab_uuid.
        Side effects: emits one logging.error() call on a miss.
        Exceptions: raises RuntimeError if even the Unknown sentinel
            isn't found on card_lookup.
        """
        exact_matches = card_lookup.get_by_name(self.SOURCE_GAME, name)
        if len(exact_matches) == 1:
            return exact_matches[0].nocab_uuid

        front_face_matches = card_lookup.get_by_name_regex(
            self.SOURCE_GAME, f"^{re.escape(name)}( //.*)?$"
        )
        if len(front_face_matches) == 1:
            return front_face_matches[0].nocab_uuid

        _logger.error(
            "SeventeenLandsGameDataDeckExtractionStage: unresolved card name "
            "%r — substituting the Unknown sentinel card",
            name,
        )
        unknown_card = card_lookup.get_by_name_single(
            self.SOURCE_GAME, CardBinder.UNKNOWN_CARD_NAME, strict=False
        )
        if unknown_card is None:
            raise RuntimeError(
                f"SeventeenLandsGameDataDeckExtractionStage: {self.SOURCE_GAME!r}'s "
                "Unknown sentinel card is not seeded — call "
                "CardBinder.ensure_unknown_card() before extract()"
            )
        return unknown_card.nocab_uuid

    def _card_nocab_uuids_for_row(
        self, row: dict, deck_columns: list[tuple[str, UUID]]
    ) -> list[UUID]:
        """Expand one row's deck_<name> copy counts into a full card multiset.

        Private helper — single consumer is _extract_row(). THIS IS
        WHERE THE "FULL DECK LIST" HAPPENS: unlike
        GameCardColumns.present_uuids() (which samples presence once
        per qualifying card for that sibling container's per-card
        metrics), this method appends card_uuid once per unit of
        row[column_name]'s own count — see module docstring's opening
        RAW SHAPE section.

        Inputs:
            row: one game_data CSV row, dict-like.
            deck_columns: this CSV's own deck_<name> column index (see
                _deck_column_index()).
        Output: card_nocab_uuids — every deck_columns entry whose
            row[column_name] is a positive count, repeated that many
            times; a zero/NaN count contributes nothing. Order is not
            meaningful (see GenericDeck's own multiset docstring).
        Side effects: none.
        Exceptions: raises KeyError if a column in deck_columns is
            missing from row.
        """
        card_nocab_uuids: list[UUID] = []
        for column_name, card_uuid in deck_columns:
            count = row[column_name]
            if pd.notna(count) and count:
                card_nocab_uuids.extend([card_uuid] * int(count))
        return card_nocab_uuids

    def _deck_name(self, row: dict) -> str:
        """Build one row's human-readable GenericDeck.name.

        Private helper — single consumer is _extract_row(). Descriptive
        only — never part of this stage's hashed identity (see module
        docstring's PER-GAME IDENTIFIER section).

        Inputs:
            row: one game_data CSV row, dict-like — carrying at least
                expansion/event_type/draft_id/match_number/game_number.
        Output: e.g. "17lands game_data MSH.PremierDraft
            <draft_id>/<match_number>/<game_number> deck".
        Side effects: none.
        Exceptions: raises KeyError if row is missing any of the fields
            named above.
        """
        draft_id = row.get("draft_id", "unknown")
        match_number = row.get("match_number", -1)
        game_number = row.get("game_number", -1)
        return (
            f"17lands game_data {row['expansion']}.{row['event_type']} "
            f"{draft_id}/{match_number}/{game_number} deck"
        )

    @staticmethod
    def _deck_uuid(draft_id: str, match_number: int, game_number: int) -> UUID:
        """Compute the deterministic nocab_uuid for one (draft_id, match_number, game_number).

        Private helper — single consumer is _extract_row(). uuid5 (not
        uuid4): the same composite key must always produce the same
        uuid, across every extract() call, so a re-seen row updates
        rather than duplicates (see module docstring's PER-GAME
        IDENTIFIER section).

        Inputs:
            draft_id: this row's own draft_id.
            match_number: this row's own match_number.
            game_number: this row's own game_number.
        Output: a uuid unique to (this fixed namespace, draft_id,
            match_number, game_number) — stable forever for a given
            triple.
        Side effects: none.
        Exceptions: none.
        """
        return uuid5(_DECK_NAMESPACE, f"{draft_id}:{match_number}:{game_number}")

    @staticmethod
    def _is_tar_wrapped(csv_path: Path) -> bool:
        """Detect whether csv_path is actually a tar archive on disk.

        Private helper — single consumer is _open_stream(). Peeks the
        first 512 bytes and checks for the ustar magic at its known
        offset — same technique and constants
        SeventeenLandsDownloader.download_one() now uses at download
        time (see module docstring's TAR-WRAPPED FILES section).

        Inputs:
            csv_path: path to check.
        Output: True if csv_path's first 512 bytes carry a ustar tar
            header; False otherwise (a plain CSV).
        Side effects: none — reads only the first 512 bytes.
        Exceptions: none expected.
        """
        with open(csv_path, "rb") as probe_file:
            header = probe_file.read(512)
        return header[_TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + len(_TAR_MAGIC)] == (
            _TAR_MAGIC
        )

    @contextmanager
    def _open_stream(self, csv_path: Path) -> Iterator[tuple[IO[bytes], int]]:
        """Open csv_path for repeated binary reads, tar-wrapped or not.

        Private helper — single consumer is _extract_csv(). Dispatches
        on _is_tar_wrapped(): for a tar-wrapped file, opens it via
        tarfile.open(csv_path, mode="r") (random-access mode — the
        on-disk quirk here is already-decompressed, so this is a plain
        seekable file, unlike download_one()'s in-flight gzip stream)
        and yields its one member's extractfile() stream (which itself
        supports .seek()/.tell() since the underlying tar file does);
        for a plain file, yields a standard open(csv_path, "rb"). Both
        branches yield a stream _extract_csv() reads TWICE — once via
        pandas for the header (nrows=0), then seek(0) back to the
        start for the real chunked pass — so the returned stream must
        support seek(0) either way.

        Inputs:
            csv_path: the file to open.
        Output (yielded): (stream, total_bytes) — stream is the binary
            handle to read; total_bytes is the CSV content's own byte
            length (the tar member's declared size when tar-wrapped,
            not csv_path's on-disk size, which for a tar-wrapped file
            includes header/padding overhead).
        Side effects: opens csv_path (and, when tar-wrapped, the
            TarFile it belongs to); both are closed on context exit.
        Exceptions: raises ValueError if csv_path is tar-wrapped but
            its tar has no members, or its first member isn't a regular
            file (mirrors
            SeventeenLandsDownloader._extract_tar_member()'s own
            guards).

        Example:
            >>> with self._open_stream(csv_path) as (stream, total_bytes):
            ...     header = pd.read_csv(stream, nrows=0).columns
            ...     stream.seek(0)
        """
        if self._is_tar_wrapped(csv_path):
            with tarfile.open(csv_path, mode="r") as tar:
                member = tar.next()
                if member is None:
                    raise ValueError(
                        f"_open_stream: {csv_path} is tar-wrapped but has no members"
                    )
                member_file = tar.extractfile(member)
                if member_file is None:
                    raise ValueError(
                        f"_open_stream: {csv_path}'s first member "
                        f"{member.name!r} isn't a regular file"
                    )
                yield member_file, member.size
        else:
            with open(csv_path, "rb") as csv_file:
                yield csv_file, csv_path.stat().st_size
