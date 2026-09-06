"""Typed references to 17Lands' public per-set/per-format data files.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

A SeventeenLandsFileRef can be produced three ways, all yielding the
same type and consumed identically by SeventeenLandsDownloader:
    - LandingPageParser.parse() (landing_page_parser.py) — discovery,
      scrapes whatever .csv.gz links are currently listed on a
      fully-rendered copy of the 17lands.com landing page's HTML.
    - SeventeenLandsFileRef.from_known() (this module) — Factory
      Method building a ref directly from a known (data_type,
      expansion, format) triple, since the download URL is a
      deterministic template. No network/HTML needed; this is the
      path when the caller already knows what they want. See
      notes.txt in this directory for a reference list of known
      expansion and format codes.
    - known_files.list_known_refs() — every (data_type, expansion,
      format_code) triple 17Lands is known to have actually published
      a file for, as of a hand-refreshed snapshot (see that module's
      docstring for why this can't be derived from the enums below:
      not every combination is valid, e.g. STX+Sealed only ever got a
      GAME file). This is what SeventeenLandsDownloader.download()
      falls back to when given an empty refs list.

No date-range filtering is offered anywhere in this container: every
17lands file is a static full-history dump per (data_type, expansion,
format) — there is no server-side way to request a time slice. Any
time-windowing is data_refinement's concern, not this one.
"""

from dataclasses import dataclass
from enum import Enum

_URL_TEMPLATE = (
    "https://17lands-public.s3.amazonaws.com/analysis_data/"
    "{data_type}/{data_type}_public.{expansion}.{format_code}.csv.gz"
)


class Expansion(str, Enum):
    HOB = "HOB"
    MSH = "MSH"
    SOS = "SOS"
    TMT = "TMT"
    ECL = "ECL"
    TLA = "TLA"
    # The 17Lands URL slug for this expansion, not a display name —
    # see notes.txt's "Cube - Powered" for the human-readable form.
    Powered_Cube = "Cube_-_Powered"
    OM1 = "OM1"
    EOE = "EOE"
    FIN = "FIN"
    TDM = "TDM"
    DFT = "DFT"
    PIO = "PIO"
    FDN = "FDN"
    DSK = "DSK"
    BLB = "BLB"
    MH3 = "MH3"
    OTJ = "OTJ"
    MKM = "MKM"
    KTK = "KTK"
    LCI = "LCI"
    WOE = "WOE"
    LTR = "LTR"
    MOM = "MOM"
    SIR = "SIR"
    ONE = "ONE"
    BRO = "BRO"
    DMU = "DMU"
    HBG = "HBG"
    SNC = "SNC"
    NEO = "NEO"
    VOW = "VOW"
    MID = "MID"
    AFR = "AFR"
    STX = "STX"
    KHM = "KHM"


class format_code(str, Enum):
    PremierDraft = "PremierDraft"
    TradDraft = "TradDraft"
    PickTwoDraft = "PickTwoDraft"
    Sealed = "Sealed"
    TradSealed = "TradSealed"
    PickTwoTradDraft = "PickTwoTradDraft"
    QuickDraft = "QuickDraft"


class DataType(str, Enum):
    """Which of 17Lands' three per-draft data tables a file holds.

    Inputs: none (enum).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    DRAFT = "draft_data"
    GAME = "game_data"
    REPLAY = "replay_data"


@dataclass(frozen=True)
class SeventeenLandsFileRef:
    """One downloadable 17Lands file: a (data_type, expansion,
    format_code) triple plus the URL it lives at.

    Field named format_code, not format, per PRINCIPLES.md section 7
    (style guide conformance) — `format` shadows the `format()`
    builtin. This is a stated, deliberate deviation from the more
    conventional field name (`format`); the meaning and position are
    otherwise identical.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    data_type: DataType
    expansion: str
    format_code: str
    url: str

    @staticmethod
    def from_known(
        data_type: DataType, expansion: str, format_code: str
    ) -> "SeventeenLandsFileRef":
        """Build a ref directly from a known triple, without parsing HTML.

        The 17Lands download URL is a deterministic template
        (https://17lands-public.s3.amazonaws.com/analysis_data/
        {data_type}/{data_type}_public.{expansion}.{format_code}.csv.gz)
        — this fills it in rather than scraping it, for callers who
        already know the expansion/format codes they want (see
        notes.txt in this directory for a reference list of known
        codes).

        Inputs:
            data_type: which of the three 17Lands data tables.
            expansion: 17Lands expansion code, e.g. "MSH".
            format_code: 17Lands format code, e.g. "PremierDraft".
        Output: a SeventeenLandsFileRef with url filled in from the
            template.
        Side effects: none — no network request, no validation that
            the resulting URL actually exists.
        Exceptions: none.

        Example:
            >>> SeventeenLandsFileRef.from_known(
            ...     DataType.GAME, "MSH", "PremierDraft"
            ... )
            SeventeenLandsFileRef(data_type=DataType.GAME, expansion='MSH', format_code='PremierDraft', url='https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.MSH.PremierDraft.csv.gz')  # noqa: E501
        """
        url = _URL_TEMPLATE.format(
            data_type=data_type.value, expansion=expansion, format_code=format_code
        )
        return SeventeenLandsFileRef(
            data_type=data_type,
            expansion=expansion,
            format_code=format_code,
            url=url,
        )
