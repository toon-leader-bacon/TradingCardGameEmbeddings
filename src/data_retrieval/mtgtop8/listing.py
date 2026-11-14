"""One MTGTop8 search-results page (/search?format=<f>&current_page=<n>)
read into ListingRows: the unwrap step for phase 1.

A results page is an HTML table of up to SEARCH_PAGE_ROWS decks, newest
first. Each deck row (<tr class="hover_tr">) holds:
    <input name="deck_ref[i]" value="<deck id>">
    <a href="event?e=<event id>&d=<deck id>&f=<format>"><deck name></a>
    <a class="player">...</a>            (not kept: a person's name)
    <td>format label</td>
    <a href="event?e=<event id>&f=<format>"><event name></a>
    <td><img star...></td>               (event level; not kept)
    <td>rank</td>                         ("1", "3-4", "5-8", ...)
    <td>date</td>                         ("09/11/26", dd/mm/yy, kept as text)

Only these fields are kept, as the site writes them; reading them
(dates, ranks) is data_refinement's job.
"""

import json
from dataclasses import asdict, dataclass, fields
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup, Tag

from src.data_retrieval.mtgtop8.formats import MtgTop8Format
from src.schema.json_checks import require_int, require_object, require_str

SEARCH_PAGE_ROWS = 25  # a full results page; a shorter one is the last


@dataclass(frozen=True)
class ListingRow:
    """One deck as a search-results row lists it."""

    deck_id: int
    event_id: int
    format: MtgTop8Format  # from the deck link's f=
    deck_name: str  # usually the archetype ("Elves")
    event_name: str
    rank: str  # the site's text: "1", "3-4", "5-8", ...
    date: str  # the site's text: "dd/mm/yy"

    def to_line(self) -> str:
        """This row as one JSON line, no newline.
        Inputs: none. Output: str. Side effects: none. Exceptions: none.

        Example:
            >>> ListingRow(1, 2, "PAU", "Elves", "League", "1", "09/11/26").to_line()
            '{"deck_id": 1, "event_id": 2, "format": "PAU", ...}'
        """
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> "ListingRow":
        """Parse one to_line() line (a trailing newline is ignored).

        Inputs: line. Output: ListingRow. Side effects: none.
        Exceptions: ValueError if it isn't a JSON object with exactly
            these fields and types, or names an unknown format.

        Example:
            >>> ListingRow.from_line(row.to_line()) == row
            True
        """
        try:
            payload = require_object(json.loads(line))
            if set(payload) != {field.name for field in fields(cls)}:
                raise ValueError(f"fields {sorted(payload)}")
            return cls(
                deck_id=require_int(payload["deck_id"]),
                event_id=require_int(payload["event_id"]),
                format=MtgTop8Format(require_str(payload["format"])),
                deck_name=require_str(payload["deck_name"]),
                event_name=require_str(payload["event_name"]),
                rank=require_str(payload["rank"]),
                date=require_str(payload["date"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed listing line {line!r}: {error}") from error


def parse_search_page(html: str) -> list[ListingRow]:
    """The deck rows of one search-results page, in page order.

    Inputs: html (the page's text, decoded from ISO-8859-1 by requests).
    Output: list of ListingRow; empty past the last page.
    Side effects: none.
    Exceptions: ValueError if html is not a results page (no search
        form: a maintenance or error page), or a deck row lacks a
        field or names a format MtgTop8Format lacks (the site changed).

    Example:
        >>> parse_search_page(page_html)[0].deck_id
        911101
    """
    result: list[ListingRow] = []

    # Validate: a real results page carries the search form
    soup = _results_soup(html)

    # Read each deck row
    for row in soup.select("tr.hover_tr"):
        result.append(_listing_row_from(row))
    return result


def _results_soup(html: str) -> Tag:
    """html parsed, once checked to be a results page.
    Inputs: html. Output: the parsed document. Side effects: none.
    Exceptions: ValueError if it has no <form action="search">."""
    result = BeautifulSoup(html, "html.parser")
    if result.find("form", action="search") is None:
        raise ValueError("not an MTGTop8 search-results page (no search form)")
    return result


def _listing_row_from(row: Tag) -> ListingRow:
    """One <tr class="hover_tr"> -> its ListingRow (the deck link gives
    deck id, event id and format; the event link the event name; the
    last two cells rank and date).
    Inputs: row. Output: ListingRow. Side effects: none.
    Exceptions: ValueError naming the missing field, or an unknown
        format code."""
    deck_link: Tag | None = None
    event_link: Tag | None = None
    for link in row.find_all("a", href=True):
        parameters = _link_parameters(str(link["href"]))
        if "e" in parameters and "d" in parameters:
            deck_link = link
        elif "e" in parameters:
            event_link = link
    cells = row.find_all("td", recursive=False)
    if deck_link is None or event_link is None or len(cells) < 2:
        raise ValueError(f"deck row lacks its deck link, event link or cells: {row}")

    deck_parameters = _link_parameters(str(deck_link["href"]))
    try:
        return ListingRow(
            deck_id=int(deck_parameters["d"]),
            event_id=int(deck_parameters["e"]),
            format=MtgTop8Format(deck_parameters["f"]),
            deck_name=deck_link.get_text(strip=True),
            event_name=event_link.get_text(strip=True),
            rank=cells[-2].get_text(strip=True),
            date=cells[-1].get_text(strip=True),
        )
    except KeyError as error:
        raise ValueError(f"deck link {deck_link['href']!r} lacks {error}") from error


def _link_parameters(href: str) -> dict[str, str]:
    """The query parameters of an "event?e=..&d=..&f=.." link (the first
    value of each); {} for a link to another page.
    Inputs: href. Output: parameter name -> value. Side effects: none.
    Exceptions: none."""
    parts = urlsplit(href)
    if not parts.path.endswith("event"):
        return {}
    return {name: values[0] for name, values in parse_qs(parts.query).items()}
