"""Translates sts2runs.com's community run dump into decks, directly into a DeckBox.

See src/data_refinement/deck_box/extraction.py for the shared
DeckExtractionStage interface this implements, and
src/data_refinement/deck_box/sts_gg/extraction_stage.py for the sibling
this is modeled on — same game (Slay the Spire 2), same
"CARD."-prefixed card id convention, same spire_codex card source, same
Unknown-sentinel-on-miss policy. The two real differences are RAW SHAPE
(below) and this dump being gzip-compressed on disk rather than plain
NDJSON.

RAW SHAPE: src/data_retrieval/sts2runs/downloader.py's monthly dump
(e.g. runs-all-before-2026-06.json.gz) — one run JSON object per line.
Confirmed against the live file: every row carries "_serverId" (int —
this row's own identity; NOT "id", unlike sts_gg) and "players" (a
list). CONFIRMED ACROSS THE ENTIRE CURRENT DUMP (6,796 rows, checked
in full): "players" is always exactly length 1 — but the schema itself
allows more, so this stage still treats it as a list and produces one
GenericDeck per player rather than assuming index 0 is the only entry.
Each players[i] carries "deck": a list of {"id": "CARD.<NAME>", ...}
entries, duplicates meaningful (copy counts) — identical shape to
sts_gg's own top-level "deck".

CARD RESOLUTION: identical to StsGgDeckExtractionStage — see that
file's own CARD RESOLUTION docstring section, not repeated here.

OPEN DESIGN QUESTION FOR REVIEW: this stage's _card_uuid()/_deck_uuid()
below would be textually identical to StsGgDeckExtractionStage's own
(same DataSource.SPIRE_CODEX, same "CARD." prefix, same Unknown
fallback) — this is the "actually the same logic" case PRINCIPLES.md
section 2 asks to centralize, not the "superficially similar" case
where duplication is fine (contrast PlayGwentDeckExtractionStage's own
_card_uuid, which resolves against a different DataSource entirely and
skips prefix-stripping). Left duplicated in this skeleton rather than
unilaterally introducing a shared spire_codex card-lookup module that
would also mean touching the already-approved sts_gg file — flagging
for a decision before implementation rather than deciding it here.

UNRESOLVED CARDS / IDEMPOTENT RE-RUNS: same policy as
StsGgDeckExtractionStage, extended one level: deck identity is
uuid5(_DECK_NAMESPACE, f"{run_id}:{player_index}") rather than just
run_id, so a future multi-player row would mint one stable, distinct
deck per player instead of colliding on one uuid — this costs nothing
for today's always-single-player reality (player_index is always 0)
and avoids a silent data loss if sts2runs' schema is ever actually used
the way it's declared.

STREAMING: the dump is gzip-compressed and large (~150MB decompressed,
6,796 lines) — extract() decompresses and reads it one line at a time
via gzip.open(..., "rt"), never loading the whole file into memory
(same convention as PlayGwentDeckExtractionStage's guides.jsonl
handling, applied to a .gz source directly rather than a
pre-extracted plain-text sibling). DEFAULT_RAW_PATH deliberately points
at the .gz STS2RunsDownloader.download() itself writes, not the
decompressed sibling STS2RunsDownloader.extract()/phase_1() produce —
the .gz is the one artifact guaranteed to exist and be current after
just download(), whereas the decompressed sibling depends on
extract()/phase_1() having also been run and not since gone stale.
"""

import gzip
import json
import logging
from collections import Counter
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse
from uuid import UUID, uuid5

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_retrieval.sts2runs.downloader import STS2RunsDownloader
from src.schema.card import GenericDeck
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_STS2RUNS_CARD_ID_PREFIX = "CARD."

# Fixed, arbitrary — never regenerate. Namespace for this stage's
# deterministic per-(run, player) deck uuids (see module docstring's
# IDEMPOTENT RE-RUNS section). Deliberately distinct from sts_gg's own
# _DECK_NAMESPACE — these are two different raw sources and must never
# collide on the same deck uuid even if a run id happened to match.
_DECK_NAMESPACE = UUID("6c9f8e2d-3a4b-4c5d-8e9f-0a1b2c3d4e5f")


class Sts2RunsDeckExtractionStage:
    """Translates sts2runs.com's monthly run dump directly into a DeckBox.

    Single-consumer to src/data_refinement/deck_box/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2

    # sts2runs' dump filename is dated and changes monthly (see
    # STS2RunsDownloader's own module docstring) — this default tracks
    # whatever STS2RunsDownloader.DEFAULT_SOURCE_URL currently points
    # at, and goes stale the same way that constant does. A caller
    # working from a newer snapshot should pass raw_path explicitly
    # rather than relying on this default.
    DEFAULT_RAW_PATH: ClassVar[Path] = (
        STS2RunsDownloader.DEFAULT_RAW_DATA_DIR
        / Path(urlparse(STS2RunsDownloader.DEFAULT_SOURCE_URL).path).name
    )

    def extract(
        self, raw_path: Path | None, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Parse sts2runs' gzip-compressed run dump, creating/updating decks on box.

        One _extract_run() call per line of raw_path, extending
        changed_uuids with whatever that call returns — a single run
        can contribute zero, one, or (should sts2runs ever actually
        populate multiple players in one row) more than one changed
        deck.

        Inputs:
            raw_path: path to a gzip-compressed NDJSON run dump (e.g.
                data/raw/sts2runs/runs-all-before-2026-06.json.gz).
                Defaults to DEFAULT_RAW_PATH when None. Always treated
                as gzip-compressed — see module docstring's STREAMING
                section.
            box: the DeckBox to create/update decks on, as a side
                effect. Every deck this call touches is stored under
                self.SOURCE_GAME.
            card_lookup: must already have self.SOURCE_GAME's Unknown
                sentinel card seeded (see
                StsGgDeckExtractionStage's own PRECONDITION) in
                addition to spire_codex's cards.
        Output: nocab_uuid of every (run, player) whose resolved card
            list this call created or changed. A re-seen (run, player)
            whose resolved list is identical to what's already stored
            is NOT included.
        Side effects: reads raw_path, one line at a time, decompressing
            as it goes; creates/updates decks directly on box; emits
            one logging.error() per card that falls back to the
            Unknown sentinel.
        Exceptions: raises if raw_path doesn't exist, isn't a valid
            gzip file, isn't valid NDJSON once decompressed, or a line
            is missing "_serverId" or "players". Raises RuntimeError if
            self.SOURCE_GAME's Unknown sentinel card isn't found on
            card_lookup.

        Example:
            >>> binder = CardBinder.load(
            ...     [Path("data/final/cards/slay_the_spire_2.jsonl")]
            ... )
            >>> box = DeckBox.load(
            ...     [Path("data/final/decks/slay_the_spire_2.jsonl")]
            ... )
            >>> stage = Sts2RunsDeckExtractionStage()
            >>> changed_uuids = stage.extract(None, box, binder)
        """
        path = raw_path or self.DEFAULT_RAW_PATH

        changed_uuids: list[UUID] = []
        # gzip.open in text mode gives us line-by-line iteration
        # straight over the compressed file — no separate decompress
        # step, no full-file read (see module docstring's STREAMING
        # section).
        with gzip.open(path, "rt", encoding="utf-8") as raw_file:
            for line in raw_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                changed_uuids.extend(self._extract_run(row, box, card_lookup))
        return changed_uuids

    def _extract_run(
        self, row: dict, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Create-or-update every player's deck from one sts2runs run.

        Private helper — single consumer is extract(). THIS METHOD IS
        WHERE IDEMPOTENCY HAPPENS: each player's deck_uuid is a pure
        function of (row["_serverId"], player_index) (see module
        docstring's IDEMPOTENT RE-RUNS section), so re-processing the
        same run always targets the same stored deck(s).

        Inputs:
            row: one parsed JSON object from raw_path (one run),
                carrying at least "_serverId" (int) and "players"
                (list of player dicts, each carrying "deck" — list of
                {"id": str, ...} card entries).
            box: the DeckBox to read from and write to.
            card_lookup: used to resolve each card entry's id (see
                _card_uuid()).
        Output: deck_uuid for every player whose create() was new or
            whose update() actually changed stored content; a player
            whose resolved list already matches what's stored
            contributes nothing to this list. Empty list is valid
            (e.g. a "players" list that's empty, though not observed
            in practice).
        Side effects: creates or updates one deck per entry in
            row["players"].
        Exceptions: raises if row is missing "_serverId" or "players",
            or a player entry is missing "deck". Whatever _card_uuid()
            raises propagates.
        """
        run_id = row["_serverId"]
        changed_uuids: list[UUID] = []

        # One deck per player — see module docstring's RAW SHAPE
        # section: always length 1 in the dump checked so far, but
        # treated as a real list rather than assuming players[0].
        for player_index, player in enumerate(row["players"]):
            deck_uuid = self._deck_uuid(run_id, player_index)

            # Resolve every deck entry's card id — never None, falls
            # back to the Unknown sentinel on a miss (see
            # _card_uuid()).
            card_nocab_uuids = [
                self._card_uuid(card_entry["id"], card_lookup)
                for card_entry in player["deck"]
            ]

            deck_name = f"sts2runs run {run_id} player {player_index}"

            existing = box.get_by_uuid(deck_uuid)
            if existing is None:
                # Brand-new (run, player).
                box.create(
                    GenericDeck(
                        nocab_uuid=deck_uuid,
                        source_game=self.SOURCE_GAME,
                        name=deck_name,
                        card_nocab_uuids=card_nocab_uuids,
                    )
                )
                changed_uuids.append(deck_uuid)
                continue

            # card_nocab_uuids is a multiset — compare copy counts via
            # Counter, never list/set equality (same reasoning as
            # StsGgDeckExtractionStage._extract_row).
            if Counter(existing.card_nocab_uuids) != Counter(card_nocab_uuids):
                box.update(deck_uuid, card_nocab_uuids=card_nocab_uuids)
                changed_uuids.append(deck_uuid)
            # else: re-seen (run, player), identical resolved multiset
            # — no-op, nothing appended.

        return changed_uuids

    def _card_uuid(self, raw_card_id: str, card_lookup: CardLookup) -> UUID:
        """Look up the nocab_uuid for one sts2runs deck entry's card id.

        Private helper — single consumer is _extract_run(). See module
        docstring's OPEN DESIGN QUESTION FOR REVIEW note: this body
        would be textually identical to
        StsGgDeckExtractionStage._card_uuid() (same "CARD." prefix
        strip, same DataSource.SPIRE_CODEX alias lookup, same Unknown
        sentinel fallback + logging).

        Inputs:
            raw_card_id: one deck entry's "id" field, e.g.
                "CARD.SETUP_STRIKE".
        Output: the matching nocab_uuid, or (on a miss) the Unknown
            sentinel's nocab_uuid.
        Side effects: emits one logging.error() call on a miss.
        Exceptions: raises RuntimeError if even the Unknown sentinel
            isn't found on card_lookup.
        """
        stripped_id = raw_card_id.removeprefix(_STS2RUNS_CARD_ID_PREFIX)
        card = card_lookup.get_by_alias(
            self.SOURCE_GAME, DataSource.SPIRE_CODEX, stripped_id
        )
        if card is not None:
            return card.nocab_uuid

        _logger.error(
            "Sts2RunsDeckExtractionStage: unresolved card id %r — "
            "substituting the Unknown sentinel card",
            raw_card_id,
        )
        unknown_card = card_lookup.get_by_name_single(
            self.SOURCE_GAME, CardBinder.UNKNOWN_CARD_NAME, strict=False
        )
        if unknown_card is None:
            raise RuntimeError(
                f"Sts2RunsDeckExtractionStage: {self.SOURCE_GAME!r}'s Unknown "
                "sentinel card is not seeded — call "
                "CardBinder.ensure_unknown_card() before extract()"
            )
        return unknown_card.nocab_uuid

    @staticmethod
    def _deck_uuid(run_id: int, player_index: int) -> UUID:
        """Compute the deterministic nocab_uuid for one (run, player) pair.

        Private helper — single consumer is _extract_run(). uuid5 over
        f"{run_id}:{player_index}" (not uuid4): the same pair must
        always produce the same uuid, across every extract() call, so
        a re-seen (run, player) updates rather than duplicates (see
        module docstring's IDEMPOTENT RE-RUNS section).

        Inputs:
            run_id: one run's own "_serverId" field.
            player_index: that player's position in the run's
                "players" list.
        Output: a uuid unique to (this fixed namespace, run_id,
            player_index) — stable forever for a given pair.
        Side effects: none.
        Exceptions: none.
        """
        return uuid5(_DECK_NAMESPACE, f"{run_id}:{player_index}")
