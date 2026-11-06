import json
from typing import Any
from unittest.mock import patch

import pytest
import requests

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.resumable_crawl.fetch import (
    PERMANENT_STATUSES,
    ItemFound,
    ItemNotFound,
    ItemRejected,
    crawl_get,
    read_outcome,
    wrap_as_found,
)


def _response(status_code: int, text: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode("utf-8")
    response.encoding = "utf-8"
    return response


def _prune(payload: Any) -> dict[str, Any]:
    return {"id": payload["id"]}


class TestReadOutcome:
    def test_a_deck_is_found_and_pruned(self) -> None:
        response = _response(200, json.dumps({"id": 4, "owner": "x"}))
        assert read_outcome(response, wrap_as_found(_prune), "deck 4") == ItemFound(
            {"id": 4}
        )

    def test_404_is_not_found(self) -> None:
        assert (
            read_outcome(_response(404, "{}"), wrap_as_found(_prune), "deck 4")
            == ItemNotFound()
        )

    @pytest.mark.parametrize("status", [400, 403, 410, 451])
    def test_another_4xx_is_rejected(self, status: int) -> None:
        assert read_outcome(
            _response(status, "x"), wrap_as_found(_prune), "deck 4"
        ) == ItemRejected(f"HTTP {status}")

    @pytest.mark.parametrize("body", ["{}", "[]"])
    def test_an_unreadable_body_is_rejected(self, body: str) -> None:
        outcome = read_outcome(_response(200, body), wrap_as_found(_prune), "deck 4")
        assert isinstance(outcome, ItemRejected)
        assert outcome.reason.startswith("unreadable deck: ")

    def test_a_non_json_200_raises_naming_the_item(self) -> None:
        with pytest.raises(ValueError, match="deck 4: HTTP 200 body is unreadable"):
            read_outcome(
                _response(200, "<html>challenge</html>"),
                wrap_as_found(_prune),
                "deck 4",
            )

    def test_429_is_not_permanent(self) -> None:
        assert 429 not in PERMANENT_STATUSES and 404 in PERMANENT_STATUSES


class TestCrawlGet:
    def test_sends_the_plain_user_agent_and_tolerated_statuses(self) -> None:
        limiter = RateLimiter(requests_per_minute=1_000_000_000)
        with patch("src.data_retrieval.resumable_crawl.fetch.paced_get") as get:
            crawl_get("https://x.test", limiter, frozenset({404}))
        kwargs = get.call_args.kwargs
        assert "non-commercial" in kwargs["headers"]["User-Agent"]
        assert "@" not in kwargs["headers"]["User-Agent"]
        assert kwargs["tolerated_statuses"] == frozenset({404})
        assert kwargs["rate_limiter"] is limiter


def test_an_unfollowed_redirect_is_rejected_with_its_location() -> None:
    response = _response(301, "")
    response.headers["Location"] = "https://x.test/"
    assert read_outcome(response, wrap_as_found(_prune), "deck 4") == ItemRejected(
        "HTTP 301 to 'https://x.test/'"
    )
