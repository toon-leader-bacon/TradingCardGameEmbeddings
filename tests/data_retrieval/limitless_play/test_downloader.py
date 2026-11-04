import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.limitless_play.downloader import (
    _MAX_RATE_LIMITED_WAITS,
    LimitlessPlayDownloader,
)
from src.data_retrieval.rate_limiter import RateLimiter

_API_URL = "https://example.test/api"
_GET = "src.data_retrieval.limitless_play.downloader.requests.get"
_SLEEP = "src.data_retrieval.limitless_play.downloader.time.sleep"


def _make_downloader(raw_data_dir: Path) -> LimitlessPlayDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return LimitlessPlayDownloader(
        RateLimiter(requests_per_minute=1_000_000_000), raw_data_dir, _API_URL
    )


def _response(payload: list[dict], status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(str(status_code))
    return response


def _rate_limited(headers: dict[str, str]) -> MagicMock:
    response = _response([], status_code=429)
    response.headers = headers
    return response


def _tournament(tournament_id: str, players: int = 8) -> dict:
    return {"id": tournament_id, "game": "PTCG", "players": players}


def _write_tournaments(raw_data_dir: Path, tournaments: list[dict]) -> None:
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    (raw_data_dir / "tournaments.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in tournaments), encoding="utf-8"
    )


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestPhase1:
    def test_pages_until_empty_and_keeps_walk_order(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        pages = [
            [_tournament("c"), _tournament("b")],
            [_tournament("b"), _tournament("a")],  # "b" shifted by a new one
            [],
        ]

        with patch(_GET, side_effect=[_response(page) for page in pages]) as get:
            result = downloader.phase_1()

        assert result == tmp_path / "tournaments.jsonl"
        assert [t["id"] for t in _jsonl(result)] == ["c", "b", "a"]
        assert [call.kwargs["params"]["page"] for call in get.call_args_list] == [
            1,
            2,
            3,
        ]
        assert get.call_args_list[0].kwargs["params"]["game"] == "PTCG"

    def test_raises_when_no_tournaments_listed(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch(_GET, return_value=_response([])):
            with pytest.raises(ValueError):
                downloader.phase_1()


class TestPhase2:
    def test_appends_standings_tagged_with_tournament_id(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        _write_tournaments(tmp_path, [_tournament("t1"), _tournament("t2")])
        standings = {"t1": [{"placing": 1}], "t2": [{"placing": 2}]}

        def _get(url: str, **_: object) -> MagicMock:
            return _response(standings[url.split("/")[-2]])

        with patch(_GET, side_effect=_get):
            result = downloader.phase_2()

        assert _jsonl(result) == [
            {"tournament_id": "t1", "standings": [{"placing": 1}]},
            {"tournament_id": "t2", "standings": [{"placing": 2}]},
        ]
        manifest = (tmp_path / "standings_manifest.txt").read_text(encoding="utf-8")
        assert manifest.split() == ["t1", "t2"]

    def test_skips_manifested_and_empty_tournaments(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        _write_tournaments(
            tmp_path,
            [_tournament("done"), _tournament("empty", players=0), _tournament("new")],
        )
        (tmp_path / "standings_manifest.txt").write_text("done\n", encoding="utf-8")

        with patch(_GET, return_value=_response([])) as get:
            downloader.phase_2()

        assert [call.args[0] for call in get.call_args_list] == [
            f"{_API_URL}/tournaments/new/standings"
        ]

    def test_skips_a_failing_tournament_and_continues(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        _write_tournaments(tmp_path, [_tournament("bad"), _tournament("good")])

        def _get(url: str, **_: object) -> MagicMock:
            if "/bad/" in url:
                return _response([], status_code=500)
            return _response([{"placing": 1}])

        with patch(_GET, side_effect=_get), patch(
            "src.data_retrieval.download_utils.time.sleep"
        ):
            result = downloader.phase_2()

        assert [row["tournament_id"] for row in _jsonl(result)] == ["good"]
        manifest = (tmp_path / "standings_manifest.txt").read_text(encoding="utf-8")
        assert manifest.split() == ["good"]


class TestRateLimiting:
    def test_waits_out_a_429_then_retries(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        responses = [
            _rate_limited({"RateLimit": '"50-in-5min"; r=0; t=42'}),
            _response([{"placing": 1}]),
        ]

        with patch(_GET, side_effect=responses), patch(_SLEEP) as sleep:
            result = downloader._fetch_standings("t1")

        assert result == [{"placing": 1}]
        sleep.assert_called_once_with(43.0)

    @pytest.mark.parametrize(
        "headers, expected_seconds",
        [
            ({"Retry-After": "7"}, 7.0),
            ({"RateLimit": '"50-in-5min"; r=0; t=212'}, 213.0),
            ({}, 300.0),
        ],
    )
    def test_wait_seconds_from_headers(
        self, tmp_path: Path, headers: dict[str, str], expected_seconds: float
    ) -> None:
        downloader = _make_downloader(tmp_path)

        assert (
            downloader._rate_limited_wait_seconds(_rate_limited(headers))
            == expected_seconds
        )

    def test_gives_up_after_repeated_429s(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch(
            _GET, return_value=_rate_limited({"Retry-After": "1"})
        ) as get, patch(_SLEEP) as sleep:
            with pytest.raises(requests.HTTPError, match="still rate limited"):
                downloader._fetch_standings("t1")

        # Every wait is followed by one more request; no sleep after the last
        assert get.call_count == _MAX_RATE_LIMITED_WAITS + 1
        assert sleep.call_count == _MAX_RATE_LIMITED_WAITS

    def test_server_errors_retry_without_a_rate_limit_wait(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        responses = [_response([], status_code=500), _response([{"placing": 1}])]

        with patch(_GET, side_effect=responses), patch(_SLEEP) as sleep, patch(
            "src.data_retrieval.download_utils.time.sleep"
        ):
            assert downloader._fetch_standings("t1") == [{"placing": 1}]

        sleep.assert_not_called()
