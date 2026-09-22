"""Masking metric: which Dominion expansion a card belongs to.

See ./BRAINSTORM.md item 3 for the full design. Deliberately NOT a
MaskedFieldMetric (../generic/masked_field_metric.py) subclass: that
base's _raw_field_value() walks MASKED_FIELD into an already-ingested
card.raw_content, but cardset_tags was excluded from raw_content at
ingestion time (see
../../card_binder/dominiontabs/ingestion_stage.py's module docstring -
"a later stage that wants 'which expansion(s) is this card in' can
still read it directly from data/raw/dominiontabs/cards_db.json").
This class does exactly that: it reads cards_db.json itself and joins
each entry back to its nocab_uuid via
CardLookup.get_by_alias(GameId.DOMINION, DataSource.DOMINIONTABS,
card_tag) - the same identity dominiontabs/ingestion_stage.py registers
every card under. Still satisfies CorpusScanMetric
(../generic/corpus_scan_metric.py) structurally, via its own scan().

cardset_tags is genuinely messy - confirmed by sampling the full live
corpus, a card's raw cardset_tags list mixes real expansion names
("nocturne", "plunder") with 1st/2nd-edition variant tags, "Removed"/
"Upgrade" edition-transition tags, and "-bigbox2-de" German promo
tags. _CANONICAL_EXPANSION_BY_PREFIX + _canonicalize_tag() collapse
each raw tag down to a canonical expansion name first (19 possible
canonical values; see LABEL_VALUES' own comment for why only 17 of
those 19 are real classification labels - "base" and "guilds" are
canonicalization targets that can never survive as a card's SOLE
canonical label). For 788 of the 817 cards in the binder (819 raw
entries minus the two German big-box duplicates the ingestion stage
skips) this collapses the whole
cardset_tags list to exactly one canonical name - those are this
metric's eligible cards. The other 29 (confirmed: Dominion's
cross-expansion basics - Copper/Silver/Gold/Estate/Duchy/Province/
Curse/Colony/Platinum/Trash/Start Deck - plus a handful of Guilds
cards also reprinted in Cornucopia's 2nd-edition box) canonicalize to
MORE than one expansion and are excluded: they don't have one true
"home expansion," so forcing a single label would be a fabricated
ground truth, not a real one.

Sanity-checked against the live corpus (819 cards): the 17 real
labels split 788 eligible cards fairly evenly (max "allies" 78 down to
min "alchemy" 12, roughly a 6.5x spread) - by far the best-balanced of
this container's three metrics.

Raw cards_db.json rows are parsed into _RawCardEntry right at the file
boundary (_load_db_entries()) rather than threaded through the rest of
this class as untyped dict - the two fields this class actually needs
(card_tag, cardset_tags) become real attributes, not repeated
string-keyed lookups.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd
from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_EDITION_SUFFIX_PATTERN = re.compile(r"(1stEdition|2ndEdition)(Removed|Upgrade)?$")

# A few cards_db.json entries are not in the binder on purpose (the ingestion
# stage skips German big-box duplicates); more than this fraction means the
# binder was built from a different raw directory.
_MAX_MISSING_FROM_BINDER_FRACTION = 0.05


@dataclass(frozen=True)
class _RawCardEntry:
    """One cards_db.json row, parsed down to the two fields this class
    actually reads - the edge-parse boundary _load_db_entries() builds,
    so every other method here works with typed attributes instead of
    string-keyed dict access.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    card_tag: str
    cardset_tags: list[str]


@dataclass(frozen=True)
class _ExpansionMaskRow:
    """One typed output row - mirrors MaskedFieldMetric's own
    _MaskedFieldRow shape (nocab_uuid/masked_field/label) so this
    metric's output is schema-compatible with every other masking
    metric's, even though this class doesn't subclass MaskedFieldMetric
    itself. Not reusing that module's _MaskedFieldRow directly - it's
    named private (leading underscore) to that module, signaling
    single-module use only.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: str
    masked_field: list[str]
    label: str


class SetMaskMetric:
    """Card -> home expansion, masked. Eligibility is real filtering,
    not every card: see this module's docstring for the 788/817 split.

    Satisfies CorpusScanMetric (../generic/corpus_scan_metric.py)
    structurally.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.DOMINION
    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/dominiontabs")
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/dominiontabs/set_mask.parquet"
    )
    CARDS_DB_FILENAME: ClassVar[str] = "cards_db.json"  # matches
    # DominionTabsCardDownloader.CARDS_DB_FILENAME
    # (src/data_retrieval/dominiontabs/downloader.py) - duplicated
    # rather than imported since that class belongs to data_retrieval,
    # a different container this one doesn't otherwise depend on.
    # "base" and "guilds" are deliberately absent, even though the
    # mapping table below still produces them as intermediate canonical
    # values - confirmed against the full live corpus, EVERY base- or
    # guilds-tagged card also canonicalizes to at least one other
    # expansion (base always co-occurs with dominion/intrigue/
    # prosperity/alchemy; guilds always co-occurs with cornucopia or
    # promo, from the 2nd-edition big-box merge), so scan()'s
    # exactly-one-canonical-label eligibility check can never actually
    # emit either as a sole label. Keeping them in LABEL_VALUES would
    # declare two classes with zero possible examples.
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "dominion",
        "intrigue",
        "seaside",
        "alchemy",
        "prosperity",
        "cornucopia",
        "hinterlands",
        "darkAges",
        "adventures",
        "empires",
        "nocturne",
        "renaissance",
        "menagerie",
        "allies",
        "plunder",
        "risingSun",
        "promo",
    )

    # Maps a raw cardset_tags entry, with any "1stEdition"/"2ndEdition"/
    # "Removed"/"Upgrade" suffix already stripped by _canonicalize_tag(),
    # to a canonical expansion name. Includes "base" and "guilds" as
    # targets even though neither is in LABEL_VALUES (see the comment
    # above) - both are still needed here so a card tagged with either
    # canonicalizes to something, letting the multi-label exclusion
    # check in scan() correctly count it as belonging to more than one
    # expansion rather than crashing on an unmapped tag. Confirmed
    # against the full live corpus (data/raw/dominiontabs/cards_db.json):
    # every one of its distinct cardset_tags values maps cleanly
    # through this table - see _canonicalize_tag()'s own docstring for
    # the suffix-stripping rule and what happens if a future raw tag
    # doesn't match any entry here.
    _CANONICAL_EXPANSION_BY_PREFIX: ClassVar[dict[str, str]] = {
        "dominion": "dominion",
        "intrigue": "intrigue",
        "seaside": "seaside",
        "alchemy": "alchemy",
        "prosperity": "prosperity",
        "cornucopiaAndGuilds": "cornucopia",
        "cornucopia": "cornucopia",
        "guilds": "guilds",
        "hinterlands": "hinterlands",
        "darkAges": "darkAges",
        "adventures": "adventures",
        "empires": "empires",
        "nocturne": "nocturne",
        "renaissance": "renaissance",
        "menagerie": "menagerie",
        "allies": "allies",
        "plunder": "plunder",
        "risingSun": "risingSun",
        "base": "base",
        "animals": "menagerie",  # menagerie's "animals" promo sub-tag
    }

    def __init__(
        self,
        card_lookup: CardLookup,
        raw_data_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_lookup: already-populated registry to resolve
                nocab_uuids against - must already have GameId.DOMINION
                cards ingested under DataSource.DOMINIONTABS aliases
                (i.e. DominionTabsCardIngestionStage has already run).
            raw_data_dir: directory containing cards_db.json - the same
                one DominionTabsCardDownloader wrote to (see
                src/data_retrieval/dominiontabs/downloader.py).
                Defaults to DEFAULT_RAW_DATA_DIR.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none - no I/O happens until scan() is called.
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._raw_data_dir = raw_data_dir or self.DEFAULT_RAW_DATA_DIR
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH

    def scan(self) -> Path:
        """Walk every cards_db.json entry, keep the ones whose
        cardset_tags canonicalize to exactly one expansion, and write
        one row per eligible card to self._output_path.

        Inputs: none (uses self._raw_data_dir, self._card_lookup).
        Output: self._output_path.
        Side effects: reads self._raw_data_dir/cards_db.json; creates
            self._output_path's parent directories if missing; writes
            self._output_path (a parquet file with columns
            nocab_uuid: str, masked_field: list[str], label: str - see
            _ExpansionMaskRow). Prints a tqdm progress bar to stderr.
        Exceptions: raises if cards_db.json is missing or isn't valid
            JSON, if an entry is missing "card_tag"/"cardset_tags", if
            a raw tag doesn't canonicalize (see _canonicalize_tag()),
            or if a card_tag has no registered DOMINIONTABS alias on
            self._card_lookup (would mean the binder wasn't built from
            this same raw_data_dir).

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/dominion.jsonl")])
            >>> metric = SetMaskMetric(binder)
            >>> metric.scan()
            PosixPath('data/metrics/dominiontabs/set_mask.parquet')
        """
        entries = self._load_db_entries()

        result: list[_ExpansionMaskRow] = []
        not_in_binder: list[str] = []
        # Walk every entry, keeping only the ones whose cardset_tags
        # canonicalize down to exactly one real expansion.
        for entry in tqdm(entries, desc=type(self).__name__, unit="card"):
            canonical_labels = self._canonical_expansion_labels(entry.cardset_tags)
            if len(canonical_labels) != 1:
                continue
            row = self._mask_row(entry, next(iter(canonical_labels)))
            if row is None:
                not_in_binder.append(entry.card_tag)
            else:
                result.append(row)
        self._check_few_missing_from_binder(not_in_binder, len(entries))

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_rows(result)
        return self._output_path

    def _load_db_entries(self) -> list[_RawCardEntry]:
        """Read, parse, and edge-parse self._raw_data_dir/cards_db.json.

        Private helper - single consumer is scan().

        Inputs: none (uses self._raw_data_dir).
        Output: one _RawCardEntry per cards_db.json row.
        Side effects: reads self._raw_data_dir/cards_db.json.
        Exceptions: raises if the file is missing, isn't valid JSON, or
            a row is missing "card_tag"/"cardset_tags".
        """
        raw_rows = json.loads(
            (self._raw_data_dir / self.CARDS_DB_FILENAME).read_text(encoding="utf-8")
        )
        return [
            _RawCardEntry(
                card_tag=raw_row["card_tag"],
                cardset_tags=raw_row["cardset_tags"],
            )
            for raw_row in raw_rows
        ]

    def _canonical_expansion_labels(self, cardset_tags: list[str]) -> set[str]:
        """Canonicalize every tag in cardset_tags and dedupe the result.

        Private helper - single consumer is scan().

        Inputs:
            cardset_tags: one _RawCardEntry's cardset_tags list (e.g.
                ["dominion1stEdition", "dominion2ndEdition"]).
        Output: the set of distinct canonical expansion names
            cardset_tags maps to via _canonicalize_tag() (e.g.
            {"dominion"} for the example above; {"intrigue", "base",
            "dominion"} for a basic card printed across multiple
            boxes - "base" here is a valid canonicalize_tag() output
            even though it's not in LABEL_VALUES, see that ClassVar's
            own comment).
        Side effects: none.
        Exceptions: raises whatever _canonicalize_tag() raises.
        """
        return {self._canonicalize_tag(raw_tag) for raw_tag in cardset_tags}

    def _canonicalize_tag(self, raw_tag: str) -> str:
        """Map one raw cardset_tags entry to a canonical expansion name.

        Private helper - single consumer is _canonical_expansion_labels().
        Strips a trailing "1stEdition"/"2ndEdition" (optionally followed
        by "Removed"/"Upgrade") suffix, then looks the remainder up in
        _CANONICAL_EXPANSION_BY_PREFIX; any tag equal to "promo",
        starting with "promo-", or containing "bigbox" maps to "promo"
        directly (checked before suffix-stripping).

        Inputs:
            raw_tag: one raw cardset_tags string (e.g.
                "dominion1stEditionRemoved", "guilds-bigbox2-de").
        Output: the matching canonical expansion name - either a
            LABEL_VALUES entry, or "base"/"guilds" (valid
            _CANONICAL_EXPANSION_BY_PREFIX targets that are never a
            card's SOLE canonical label - see that ClassVar's own
            comment).
        Side effects: none.
        Exceptions: raises ValueError if raw_tag doesn't match any
            known prefix after suffix-stripping - fail loudly rather
            than silently mis-mapping a future raw tag this table
            hasn't been taught about yet (confirmed today: every tag
            in the live corpus maps cleanly).
        """
        if raw_tag == "promo" or raw_tag.startswith("promo-") or "bigbox" in raw_tag:
            return "promo"

        stripped = _EDITION_SUFFIX_PATTERN.sub("", raw_tag)
        try:
            return self._CANONICAL_EXPANSION_BY_PREFIX[stripped]
        except KeyError:
            raise ValueError(
                f"{type(self).__name__}: unrecognized cardset_tags entry "
                f"{raw_tag!r} (stripped to {stripped!r}) - "
                "_CANONICAL_EXPANSION_BY_PREFIX needs a new entry for it"
            ) from None

    def _check_few_missing_from_binder(
        self, not_in_binder: list[str], entry_count: int
    ) -> None:
        """Raise if too many raw entries have no binder card.

        A few are expected: the ingestion stage skips cards_db.json entries
        it does not treat as cards (the German big-box duplicates). Many
        would mean the binder was built from a different raw directory.

        Inputs: not_in_binder (list[str]): card_tags with no binder card.
            entry_count (int): number of raw entries scanned.
        Output: none.
        Side effects: none.
        Exceptions: ValueError if more than _MAX_MISSING_FROM_BINDER_FRACTION
            of the entries are missing.
        """
        if len(not_in_binder) > _MAX_MISSING_FROM_BINDER_FRACTION * entry_count:
            raise ValueError(
                f"{type(self).__name__}: {len(not_in_binder)} of {entry_count} "
                f"cards_db.json entries have no registered DOMINIONTABS alias "
                f"(e.g. {not_in_binder[:3]}) - was card_lookup built from this "
                "same raw_data_dir?"
            )

    def _mask_row(self, entry: _RawCardEntry, label: str) -> _ExpansionMaskRow | None:
        """Build one output row for a single already-eligible entry.

        Private helper - single consumer is scan().

        Inputs:
            entry: one _RawCardEntry that scan() has already determined
                is eligible.
            label: entry's single canonical expansion name, from
                _canonical_expansion_labels().
        Output: None if entry.card_tag has no binder card (see
            _check_few_missing_from_binder); otherwise an
            _ExpansionMaskRow - nocab_uuid (resolved via
            self._card_lookup.get_by_alias(self.SOURCE_GAME,
            DataSource.DOMINIONTABS, entry.card_tag)),
            masked_field (["cardset_tags"]), and label.
        Side effects: none.
        Exceptions: none.
        """
        card = self._card_lookup.get_by_alias(
            self.SOURCE_GAME, DataSource.DOMINIONTABS, entry.card_tag
        )
        if card is None:
            return None
        return _ExpansionMaskRow(
            nocab_uuid=str(card.nocab_uuid),
            masked_field=["cardset_tags"],
            label=label,
        )

    def _write_rows(self, rows: list[_ExpansionMaskRow]) -> None:
        """Write rows to self._output_path as a parquet file.

        Private helper - single consumer is scan(). Mirrors
        MaskedFieldMetric.scan()'s own pandas.DataFrame(...).to_parquet(...)
        call, over _ExpansionMaskRow instead of _MaskedFieldRow.

        Inputs:
            rows: every eligible card's output row.
        Output: none.
        Side effects: writes self._output_path, overwriting any
            existing file there.
        Exceptions: whatever pandas.DataFrame.to_parquet raises.
        """
        pd.DataFrame([asdict(row) for row in rows]).to_parquet(
            self._output_path, index=False
        )
