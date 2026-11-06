"""One crawl GET and the reading of its answer: crawl_get (paced, a
plain honest User-Agent, permanent statuses returned rather than raised)
and read_outcome (the answer as a FetchOutcome variant).

Anything raised is transient: network errors, 5xx, a 429 past
paced_get's waits, and (from read_outcome) a 2xx whose body the
caller's parse_body can't read, as a challenge or maintenance page is.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable

import requests

from src.data_retrieval.download_utils import paced_get
from src.data_retrieval.rate_limiter import RateLimiter

# Every 4xx is a permanent answer for one id, except 429 (waited out)
PERMANENT_STATUSES = frozenset(range(400, 500)) - {429}
# The redirects a no-follow GET hands back to its caller
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_CRAWL_USER_AGENT = "TradingCardGameEmbeddings-research/0.1 (non-commercial)"
_REQUEST_TIMEOUT_SECONDS = 60
_BODY_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class ItemFound:
    """An item, pruned, ready to land as one shard row: in the crawl's
    directory, or (subdirectory set) in a subdirectory of it, apart from
    the other rows."""

    row: dict[str, Any]
    subdirectory: str | None = None


@dataclass(frozen=True)
class ItemNotFound:
    """A 404: deleted, private or never used."""


@dataclass(frozen=True)
class ItemRejected:
    """A permanent failure for one id; recorded, and the walk moves on."""

    reason: str


FetchOutcome = ItemFound | ItemNotFound | ItemRejected


def crawl_get(
    url: str,
    rate_limiter: RateLimiter,
    tolerated_statuses: frozenset[int] = frozenset(),
    follow_redirects: bool = True,
) -> requests.Response:
    """paced_get with the crawls' User-Agent (plain and honest; no
    contact details) and timeout.

    Inputs: url, rate_limiter, tolerated_statuses (non-2xx returned, not
        raised), follow_redirects (False: a 3xx comes back as it is, so
        the caller paces the redirect's GET). Output: requests.Response.
    Side effects: one paced GET (more on a retry or a 429).
    Exceptions: as paced_get.

    Example:
        >>> crawl_get("https://moxfield.com/sitemap.xml", RateLimiter()).status_code
        200
    """
    return paced_get(
        url,
        rate_limiter=rate_limiter,
        headers={"User-Agent": _CRAWL_USER_AGENT},
        timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
        tolerated_statuses=tolerated_statuses,
        follow_redirects=follow_redirects,
    )


def read_outcome(
    response: requests.Response,
    outcome_of_body: Callable[[Any], FetchOutcome],
    item_label: str,
    parse_body: Callable[[str], Any] = json.loads,
) -> FetchOutcome:
    """Read one GET's answer (fetched with
    tolerated_statuses=PERMANENT_STATUSES, so anything raised was
    transient) into a FetchOutcome: 404 -> ItemNotFound; another
    tolerated 4xx -> ItemRejected("HTTP <n>"); a 3xx (from a GET that
    didn't follow redirects, and whose caller didn't follow this one) ->
    ItemRejected("HTTP <n> to <Location>"); a parsed body
    outcome_of_body can't read (KeyError or TypeError) ->
    ItemRejected("unreadable deck: ...", Archidekt's wording); else
    outcome_of_body(parse_body(text)).

    Inputs: response, outcome_of_body (parsed body -> outcome; for a
        plain prune, wrap_as_found(prune)), item_label (names the item in
        an error, "deck 4242"), parse_body (text -> body; JSON by
        default, Deckstats reads its page JSON out of HTML).
    Output: FetchOutcome. Side effects: none.
    Exceptions: ValueError for a 2xx body parse_body can't read (a
        challenge or maintenance page: transient, so the cursor stays).

    Example:
        >>> read_outcome(response_404, wrap_as_found(prune), "deck 4242")
        ItemNotFound()
    """
    if response.status_code == 404:
        return ItemNotFound()
    if response.status_code in PERMANENT_STATUSES:
        return ItemRejected(f"HTTP {response.status_code}")
    # A redirect a no-follow caller didn't take: permanent for this item
    if response.status_code in REDIRECT_STATUSES:
        location = response.headers.get("Location", "")
        return ItemRejected(f"HTTP {response.status_code} to {location!r}")
    try:
        payload = parse_body(response.text)
    except ValueError as error:
        raise ValueError(
            f"{item_label}: HTTP {response.status_code} body is unreadable: "
            f"{response.text[:_BODY_PREVIEW_CHARS]!r}"
        ) from error
    try:
        return outcome_of_body(payload)
    except (KeyError, TypeError) as error:
        return ItemRejected(f"unreadable deck: {error!r}")


def wrap_as_found(
    prune: Callable[[Any], dict[str, Any]],
) -> Callable[[Any], FetchOutcome]:
    """read_outcome's outcome_of_body for a source whose every readable
    body is one item: body -> ItemFound(prune(body)).

    Inputs: prune (body -> the row to land). Output: a callable.
    Side effects: none. Exceptions: none (the callable raises as prune).

    Example:
        >>> wrap_as_found(lambda body: {"id": body["id"]})({"id": 4})
        ItemFound(row={'id': 4}, subdirectory=None)
    """
    return lambda body: ItemFound(prune(body))
