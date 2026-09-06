"""Parses SeventeenLandsFileRefs out of the 17lands.com landing page.

Single-consumer to src/data_retrieval/seventeenlands/ — no other source
directory depends on this class. See refs.py for the alternative,
non-HTML way to produce the same SeventeenLandsFileRef type.
"""

import re

from src.data_retrieval.seventeenlands.refs import DataType, SeventeenLandsFileRef

# Matches the .csv.gz download links on the 17lands.com landing page,
# e.g.:
#   https://17lands-public.s3.amazonaws.com/analysis_data/game_data/
#     game_data_public.MSH.PremierDraft.csv.gz
# Captures (data_type_path_segment, expansion, format) — the leading
# "{type}_public." repeat in the filename is redundant with the path
# segment, so it's matched but not captured.
_CSV_GZ_LINK_PATTERN = re.compile(
    r"https://17lands-public\.s3\.amazonaws\.com/analysis_data/"
    r"(draft_data|game_data|replay_data)/"
    r"(?:draft_data|game_data|replay_data)_public\.([^./]+)\.([^./]+)\.csv\.gz"
)


class LandingPageParser:
    """Extracts every 17Lands .csv.gz download link from a landing page.

    Single-consumer to src/data_retrieval/seventeenlands/ — no other source
    directory depends on this class.
    """

    def parse(self, html: str) -> list[SeventeenLandsFileRef]:
        """Extract every 17Lands .csv.gz download link in html.

        Inputs:
            html: raw HTML of the 17lands.com landing page (or any
                page containing 17lands-public.s3.amazonaws.com
                .csv.gz links in the expected form).
        Output: list of SeventeenLandsFileRef, one per matched link,
            in the order they appear in html. Empty if none match —
            never raises on malformed/unexpected input, since this is
            a best-effort scrape of a page this project doesn't
            control (same convention as
            HearthstoneJsonDownloader._parse_build_ids_with_enus_subdirectory).
        Side effects: none — pure parsing, no network or disk I/O.
        Exceptions: none expected from well-formed HTML.

        Example:
            >>> parser = LandingPageParser()
            >>> html = '<a href="https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.MSH.PremierDraft.csv.gz">link</a>'  # noqa: E501
            >>> parser.parse(html)
            [SeventeenLandsFileRef(data_type=DataType.GAME, expansion='MSH', format_code='PremierDraft', url='https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.MSH.PremierDraft.csv.gz')]  # noqa: E501
        """
        refs = []
        for match in _CSV_GZ_LINK_PATTERN.finditer(html):
            data_type_segment, expansion, format_code = match.groups()
            refs.append(
                SeventeenLandsFileRef(
                    data_type=DataType(data_type_segment),
                    expansion=expansion,
                    format_code=format_code,
                    url=match.group(0),
                )
            )
        return refs
