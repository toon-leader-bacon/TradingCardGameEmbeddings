"""Shared builders for the mtgtop8 retrieval tests: search pages and
deck exports in the site's shape (probed 2026-11-11)."""

from dataclasses import dataclass

from src.data_retrieval.mtgtop8.formats import MtgTop8Format
from src.data_retrieval.mtgtop8.listing import ListingRow


@dataclass(frozen=True)
class PageDeck:
    """One deck row of a fake search page."""

    deck_id: int
    event_id: int = 93623
    format_code: str = "PAU"
    deck_name: str = "Elves"
    event_name: str = "Bournemouth League @ The Trove (Bournemouth, UK)"
    rank: str = "2"
    date: str = "09/11/26"


def listing_row(
    deck_id: int, deck_format: MtgTop8Format = MtgTop8Format.PAUPER
) -> ListingRow:
    """A ListingRow with fixed text fields."""
    return ListingRow(deck_id, 93623, deck_format, "Elves", "League", "1", "09/11/26")


def search_page_html(decks: list[PageDeck]) -> str:
    """A search-results page holding decks, with the search form, the
    compare form's checkbox cells, a player link and a star image, as
    the site writes them."""
    rows = "".join(f"""<tr class=hover_tr>
<td><input type=checkbox name=deck_check[{index}] value=1 checked>
<input type=hidden name=deck_ref[{index}] value={deck.deck_id}></td>
<td class=S12><a href={_deck_href(deck)}>{deck.deck_name}</a></td>
<td class=G12><a class=player href=search?player=Some+One>Some One</a></td>
<td class=S12>Pauper</td>
<td class=S11><a href=event?e={deck.event_id}&f={deck.format_code}>{deck.event_name}</a></td>
<td align=center><img src=/graph/star.png></td>
<td align=center class=S12>{deck.rank}</td>
<td class=S11>{deck.date}</td>
</tr>""" for index, deck in enumerate(decks, start=1))
    return f"""<html><body>
<form action=search method=POST><input type=hidden name=current_page value=""></form>
<div>{len(decks)} decks matching</div>
<form action=compare method=POST><table>{rows}</table></form>
</body></html>"""


def _deck_href(deck: PageDeck) -> str:
    """The deck row's link to its event page, with the deck selected."""
    return f"event?e={deck.event_id}&d={deck.deck_id}&f={deck.format_code}"


EXPORT = (
    "4 Lightning Bolt\r\n4 Fire/Ice\r\n52 Mountain\r\n"
    "Sideboard\r\n1 Lurrus of the Dream-Den\r\n3 Pyroblast\r\n"
)
