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

CARD MATCHING: each deck_<name> column's bare <name> suffix is matched
with card_lookup.uuid_for_name_or_front_face() (a unique exact name
match, else a unique split/MDFC front-face match), the same policy the
17lands metrics use; else the card is unresolved.

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

ONE DECK PER DRAFT, CONTENT-ADDRESSED: each game_data row is one game,
and a draft plays 3-7 games with the same or nearly the same deck
(only sideboard swaps differ). Storing one deck per game multiplied
the box ~5x (~28M decks) and let 17lands dominate every deck-based
dojo, so this stage stores at most one deck per draft. That deck's
identity is deck_uuid_from_cards() (see
src/data_refinement/deck_ids.py) over the draft's canonical
game's fully-resolved card multiset — a CONTENT hash, not a function
of draft_id — so two different drafts that build the exact same
40-card deck collapse into ONE stored GenericDeck, letting the box
naturally deduplicate across drafts (this is deliberate: the box is
meant to hold unique decklists for external publication, train/test
splits over decks must not leak a decklist shared by two drafts across
the split, and per-draft/per-game popularity is tracked by a separate
metric instead of by storage duplication). A draft_id is assumed
globally unique across every game_data CSV in this directory (17Lands'
own convention) — this stage never folds the source file's
expansion/format into the hashed identity, only into the stored
deck's human-readable name.

CANONICAL GAME: the game whose cards are hashed for a draft is the one
from its lowest (build_index, match_number, game_number), its
_GameKey, which provenance.source_id records as
f"{draft_id}:{build_index}:{match_number}:{game_number}". build_index
is 17Lands' deck-build counter (0 = the deck as first built, +1 per
rebuild or sideboard change), so the hashed deck is the draft's
earliest played build (~30% of drafts have no build-0 game on
record). CONFIRMED against the live files: all 133 carry build_index,
but the 10 older tar-wrapped ones (see TAR-WRAPPED FILES) have no
match_number column and number games per match, so a draft there has
several game_number == 1 rows; match_number counts as 0 for those
files. Rows that still tie share a build_index, i.e. the same
decklist, so which one wins doesn't matter.

TWO-PHASE EXTRACTION, because content-addressed identity breaks the
old single-pass "look up by a draft-derived key" trick: the final
deck_uuid depends on the RESOLVED CARD LIST of whichever game turns
out to be canonical, which isn't known until every row of a draft has
been seen — so a row can no longer be looked up against "what's
currently stored for this draft" by a deterministic draft-derived key
(there isn't one anymore). _extract_csv() instead runs two phases per
file:
    1. Streaming accumulation (_accumulate_best_games(), still one
       pass over the CSV, still chunked, never the whole file in
       memory): for every row, compute its _GameKey and, only when
       that key improves on this draft's best key seen so far in this
       file, resolve the row's card multiset and keep
       (game_key, card_nocab_uuids, expansion, event_type) as that
       draft's new best — a dict keyed by draft_id, bounded by one
       file's distinct draft count (see ONE DECK PER DRAFT's own
       global-uniqueness assumption — this dict never needs to span
       files or process runs on that same assumption).
    2. Resolution/write (_write_best_games()), after the whole file
       has streamed: for each accumulated draft, hash its best game's
       card multiset into a deck_uuid and box.create_if_absent() the
       resulting GenericDeck — content-derived identity makes
       create_if_absent() the correct, naturally idempotent call (no
       more create()-vs-replace() branching): a draft whose resolved
       deck matches one already stored, from this draft or a
       different one, is just a no-op recurrence. This also makes
       crash-retry trivially safe — a retried file recomputes the
       identical accumulation from scratch and resubmits the same
       create_if_absent() calls, which no-op against anything already
       durably stored.

changed_uuids CONTRACT: because drafts no longer map 1:1 to stored
uuids (two drafts can legitimately target the same deck_uuid),
extract()/_extract_csv() return every deck_uuid that this call's
create_if_absent() calls actually targeted — whether that call stored
a genuinely new deck or merely recurred, no-op, against one already
present — across every processed file, each uuid listed once. A
deck_uuid that several drafts (or several extract() calls) all
resolve to still appears only once.

FRESH BOX REQUIRED AFTER AN IDENTITY CHANGE: an mtg.db written under
either of this stage's earlier identity schemes — the original
per-game namespace (a3d8f1c2-...) or the later per-draft namespace
(163ae839-...) — must be deleted before re-running under this
content-addressed scheme. None of those ids match deck_uuid_from_cards()
output, so a re-run would add content-addressed decks next to the
stale per-game/per-draft ones and save() would stamp the mixed box as
current.

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
The per-draft accumulation dict (see TWO-PHASE EXTRACTION) is the only
state held past a single chunk, and it holds one resolved card
multiset per distinct draft_id in the file, not per row.

DIRECTORY HANDLING: raw_path may be a single CSV file or a directory of
them (see extraction.py's own documented flexibility) — DEFAULT_RAW_PATH
is SeventeenLandsDownloader.DEFAULT_RAW_DATA_DIR / "game_data", the
whole directory. A directory's *.csv files are processed in sorted
order, each contributing independently to the one returned
changed_uuids list.

PROVENANCE: each created/recurred deck carries a Provenance
(DataSource.SEVENTEENLANDS_GAME_DATA,
source_id=f"{draft_id}:{build_index}:{match_number}:{game_number}" of
the canonical game, fetched_at=now) — not SCRYFALL (that's each card's
own data source, used only for card matching). On a create_if_absent()
recurrence the ALREADY-STORED provenance wins (create_if_absent()'s own
documented contract — the existing entry is returned unconditionally,
never overwritten), so a deck's stored provenance records whichever
draft/game first wrote it, not necessarily the one being processed now.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, NamedTuple
from uuid import UUID

import pandas as pd
from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.seventeenlands_game_data._card_columns import (
    _deck_column_index,
)
from src.data_refinement.deck_box.seventeenlands_game_data._game_key import _GameKey
from src.data_refinement.deck_box.seventeenlands_game_data._tar_aware_csv_stream import (
    _open_stream,
)
from src.data_refinement.deck_ids import deck_uuid_from_cards
from src.data_retrieval.seventeenlands.downloader import SeventeenLandsDownloader
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# I/O-efficiency knob only (see scan_game_csv()'s own chunk_size
# parameter for precedent) - never changes what _accumulate_best_games()
# sees, still one row at a time.
_CHUNK_SIZE = 100_000


class _BestGame(NamedTuple):
    """One draft's best-known (lowest _GameKey) game, mid-accumulation.

    Held in extract()'s per-file accumulation dict (see module
    docstring's TWO-PHASE EXTRACTION section) — just enough to write
    this draft's deck once the whole file has streamed, without
    retaining the raw row.
    """

    game_key: _GameKey
    card_nocab_uuids: list[UUID]
    expansion: str
    event_type: str


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
        """Parse every game_data CSV under raw_path, creating/recurring decks on box.

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
        Output: every deck_uuid this call's box.create_if_absent()
            calls targeted, across every processed file, each listed
            once — see module docstring's changed_uuids CONTRACT
            section (this includes no-op recurrences against an
            already-stored deck, not only genuinely new decks).
        Side effects: reads every processed CSV (streamed, never fully
            in memory — see module docstring's STREAMING section);
            creates decks on box (via create_if_absent()); emits one
            logging.error() per column name that falls back to the
            Unknown sentinel; prints one tqdm progress bar per file to
            stderr.
        Exceptions: raises if raw_path doesn't exist, a file isn't
            parsable as CSV (or as a tar-wrapped CSV — see module
            docstring's TAR-WRAPPED FILES section), or a row is missing
            draft_id/build_index/game_number. Raises RuntimeError if
            self.SOURCE_GAME's Unknown sentinel card isn't found on
            card_lookup.

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> box = DeckBox.load([Path("data/final/decks/mtg.db")])
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
        # A deck_uuid can recur across files (two drafts in different
        # files sharing a decklist, or the same draft split across
        # files) - report it once (see module docstring's
        # changed_uuids CONTRACT section).
        return list(dict.fromkeys(changed_uuids))

    def _extract_csv(
        self, csv_path: Path, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Stream one game_data CSV (plain or tar-wrapped), creating/recurring decks on box.

        Private helper — single consumer is extract(). Opens csv_path
        exactly once via _open_stream() (tar-aware): reads the header
        first (to build this file's own deck column index), seeks back
        to the start, then re-reads the same stream in chunks for the
        real pass — see _open_stream()'s own docstring for why this
        works uniformly for a tar-wrapped member and a plain file. Runs
        the two phases in module docstring's TWO-PHASE EXTRACTION
        section: _accumulate_best_games() over the whole streamed file,
        then _write_best_games() once, after the stream is exhausted.

        Inputs:
            csv_path: one game_data CSV file.
            box: the DeckBox to write decks onto.
            card_lookup: used to resolve every deck_<name> column (see
                _deck_column_index()).
        Output: deck_uuid for every draft in this file whose best game
            was written via box.create_if_absent(), in this file alone
            (two drafts may produce the same deck_uuid; extract()
            dedupes across files, and this method itself only ever
            calls create_if_absent() once per distinct draft_id, so no
            within-file dedup is needed here).
        Side effects: reads csv_path; creates decks on box (via
            create_if_absent()); prints one tqdm progress bar to
            stderr, sized against this file's own content length (the
            tar member's size, when tar-wrapped — not the
            compressed/wrapped file's own size on disk).
        Exceptions: raises if csv_path isn't parsable as CSV (or
            tar-wrapped CSV), or a row is missing
            draft_id/build_index/game_number. Whatever
            _accumulate_best_games()/_deck_column_index() raise
            propagates.
        """
        with _open_stream(csv_path) as (binary_stream, total_bytes):
            header_columns = pd.read_csv(binary_stream, nrows=0).columns
            binary_stream.seek(0)
            deck_columns = _deck_column_index(
                header_columns, card_lookup, self.SOURCE_GAME
            )

            best_games: dict[str, _BestGame] = {}
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

                    # Hand every row to _accumulate_best_games() one at
                    # a time (never the whole chunk at once).
                    for row in chunk.to_dict(orient="records"):
                        self._accumulate_best_games(row, deck_columns, best_games)

        return self._write_best_games(best_games, box)

    def _accumulate_best_games(
        self,
        row: dict,
        deck_columns: list[tuple[str, UUID]],
        best_games: dict[str, _BestGame],
    ) -> None:
        """Update best_games in place with row, if row's game is this draft's best yet.

        Private helper — single consumer is _extract_csv(), called once
        per streamed row (see module docstring's TWO-PHASE EXTRACTION
        section, phase 1). A row's card multiset is only resolved when
        its _GameKey actually improves on what's already accumulated
        for this draft_id — a worse-keyed row is dropped after a cheap
        _GameKey comparison, never paying for card resolution.

        Inputs:
            row: one game_data CSV row, dict-like — carrying at least
                draft_id/build_index/game_number/expansion/event_type
                (and match_number in files that have it), plus this
                file's deck_<name> columns.
            deck_columns: this CSV's own deck_<name> column index (see
                _deck_column_index()).
            best_games: this file's accumulation dict so far, keyed by
                draft_id — MUTATED in place: gains a new entry for a
                draft_id not yet seen, or has its entry replaced when
                row's _GameKey is strictly lower than the stored one.
                A tying or higher key leaves the existing entry
                untouched (ties share a build_index, i.e. the same
                decklist per module docstring's CANONICAL GAME section,
                so which tied row wins doesn't matter).
        Output: none (mutates best_games).
        Side effects: none beyond mutating best_games. Resolving row's
            card list may emit one logging.error() per unresolved
            column name, via _card_nocab_uuids_for_row().
        Exceptions: raises KeyError if row is missing
            draft_id/build_index/game_number, and ValueError if
            any key number is NaN (see _GameKey.from_row()).
        """
        # str(): pandas may infer a non-string dtype for a chunk's
        # draft_id column.
        draft_id = str(row["draft_id"])
        game_key = _GameKey.from_row(row)

        existing = best_games.get(draft_id)
        if existing is not None and game_key >= existing.game_key:
            return

        card_nocab_uuids = self._card_nocab_uuids_for_row(row, deck_columns)
        best_games[draft_id] = _BestGame(
            game_key=game_key,
            card_nocab_uuids=card_nocab_uuids,
            expansion=str(row["expansion"]),
            event_type=str(row["event_type"]),
        )

    def _write_best_games(
        self, best_games: dict[str, _BestGame], box: DeckBox
    ) -> list[UUID]:
        """Hash and store every accumulated draft's best game (phase 2).

        Private helper — single consumer is _extract_csv(), called once
        per file after the whole file has streamed (see module
        docstring's TWO-PHASE EXTRACTION section, phase 2).

        Inputs:
            best_games: this file's full accumulation dict, keyed by
                draft_id, as built by repeated _accumulate_best_games()
                calls.
            box: the DeckBox to write decks onto.
        Output: deck_uuid for every draft_id in best_games, in
            best_games' own iteration order (a draft_id's deck_uuid is
            always included, whether box.create_if_absent() stored it
            fresh or found it already present — see module docstring's
            changed_uuids CONTRACT section).
        Side effects: one box.create_if_absent() call per draft_id in
            best_games.
        Exceptions: whatever box.create_if_absent() raises propagates
            (see DeckBox's own module docstring's BATCHED COMMITS
            section).
        """
        written_uuids: list[UUID] = []
        for draft_id, best_game in best_games.items():
            deck_uuid = deck_uuid_from_cards(best_game.card_nocab_uuids)
            deck = GenericDeck(
                nocab_uuid=deck_uuid,
                source_game=self.SOURCE_GAME,
                name=self._deck_name(
                    best_game.expansion, best_game.event_type, draft_id
                ),
                card_nocab_uuids=best_game.card_nocab_uuids,
                provenance=Provenance(
                    data_source=DataSource.SEVENTEENLANDS_GAME_DATA,
                    source_id=best_game.game_key.source_id_for(draft_id),
                    fetched_at=datetime.now(timezone.utc),
                ),
            )
            box.create_if_absent(deck)
            written_uuids.append(deck_uuid)
        return written_uuids

    def _card_nocab_uuids_for_row(
        self, row: dict, deck_columns: list[tuple[str, UUID]]
    ) -> list[UUID]:
        """Expand one row's deck_<name> copy counts into a full card multiset.

        Private helper — single consumer is _accumulate_best_games().
        THIS IS WHERE THE "FULL DECK LIST" HAPPENS: unlike
        the game_data metrics' ZoneCounts.present() (which samples
        presence once per qualifying card for that sibling container's
        per-card metrics), this method appends card_uuid once per unit of
        row[column_name]'s own count — see module docstring's opening
        RAW SHAPE section.

        Inputs:
            row: one game_data CSV row, dict-like.
            deck_columns: this CSV's own deck_<name> column index (see
                _deck_column_index()).
        Output: card_nocab_uuids — every deck_columns entry whose
            row[column_name] is a positive count, repeated that many
            times; a zero/NaN count contributes nothing. Order is not
            meaningful (see GenericDeck's own multiset docstring) —
            deck_uuid_from_cards() sorts before hashing.
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

    def _deck_name(self, expansion: str, event_type: str, draft_id: str) -> str:
        """Build one draft's human-readable GenericDeck.name.

        Private helper — single consumer is _write_best_games().
        Descriptive only — never part of this stage's hashed identity
        (see module docstring's ONE DECK PER DRAFT, CONTENT-ADDRESSED
        section).

        Inputs:
            expansion: the draft's best game's expansion code.
            event_type: the draft's best game's event_type.
            draft_id: the draft's own draft_id.
        Output: e.g. "17lands game_data MSH.PremierDraft <draft_id>
            deck".
        Side effects: none.
        Exceptions: none.
        """
        return f"17lands game_data {expansion}.{event_type} {draft_id} deck"
