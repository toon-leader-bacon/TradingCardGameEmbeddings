import gzip
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.tcgcsv.downloader import (
    CATEGORY_IDS,
    TcgcsvCatalogDownloader,
    parse_results,
    snapshot_name,
)

_MODULE = "src.data_retrieval.tcgcsv.downloader"
_BASE = "https://t.test"
_STAMP = "2026-11-06T20:05:28+0000"
_SNAPSHOT = "20261106T200528Z"


def _answer(results: list[Any]) -> str:
    return json.dumps({"success": True, "errors": [], "results": results})


# A tiny tcgcsv site: Gundam (86) with two groups
_SITE = {
    f"{_BASE}/last-updated.txt": _STAMP,
    f"{_BASE}/tcgplayer/86/groups": _answer(
        [{"groupId": 1, "name": "Newtype Rising"}, {"groupId": 2, "name": "Promos"}]
    ),
    f"{_BASE}/tcgplayer/86/1/products": _answer([{"productId": 10, "groupId": 1}]),
    f"{_BASE}/tcgplayer/86/2/products": _answer([{"productId": 20, "groupId": 2}]),
}


def _make_downloader(raw_dir: Path, **kwargs: Any) -> TcgcsvCatalogDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    kwargs.setdefault("slugs", ("gundam",))
    return TcgcsvCatalogDownloader(
        RateLimiter(requests_per_minute=1_000_000_000),
        raw_dir,
        base_url=_BASE,
        **kwargs,
    )


def _run(raw_dir: Path, site: dict[str, str] = _SITE) -> tuple[Path, list[str]]:
    calls: list[str] = []

    def _get(url: str, **kwargs: Any) -> str:
        calls.append(url)
        assert "TradingCardGameEmbeddings" in kwargs["headers"]["User-Agent"]
        return site[url]

    with patch(f"{_MODULE}.download_to_string", side_effect=_get):
        result = _make_downloader(raw_dir).phase_1()
    return result, calls


class TestPhase1:
    def test_lands_groups_and_products_then_the_stamp(self, tmp_path: Path) -> None:
        result, calls = _run(tmp_path)

        assert result == tmp_path / _SNAPSHOT
        assert len(calls) == 4
        with gzip.open(result / "gundam.json.gz", "rt", encoding="utf-8") as file:
            catalog = json.load(file)
        assert catalog["category_id"] == 86
        assert [g["groupId"] for g in catalog["groups"]] == [1, 2]
        assert [p["productId"] for p in catalog["products"]] == [10, 20]
        assert (result / "last-updated.txt").read_text() == _STAMP
        assert not list(result.glob("*.tmp"))

    def test_a_landed_game_is_not_fetched_again(self, tmp_path: Path) -> None:
        (tmp_path / _SNAPSHOT).mkdir()
        (tmp_path / _SNAPSHOT / "gundam.json.gz").write_bytes(b"landed")
        _, calls = _run(tmp_path)
        assert calls == [f"{_BASE}/last-updated.txt"]

    def test_a_failed_answer_writes_no_catalog_and_no_stamp(
        self, tmp_path: Path
    ) -> None:
        site = {**_SITE, f"{_BASE}/tcgplayer/86/2/products": '{"success": false}'}
        with pytest.raises(ValueError, match="success"):
            _run(tmp_path, site)
        assert list((tmp_path / _SNAPSHOT).iterdir()) == []

    def test_a_group_without_an_int_id_raises(self, tmp_path: Path) -> None:
        site = {**_SITE, f"{_BASE}/tcgplayer/86/groups": _answer([{"groupId": "1"}])}
        with pytest.raises(TypeError):
            _run(tmp_path, site)


def test_snapshot_name() -> None:
    assert snapshot_name(_STAMP + "\n") == _SNAPSHOT


@pytest.mark.parametrize("stamp", ["2026-11-06", "2026-11-06T20:05:28Z", "../../x", ""])
def test_a_malformed_stamp_is_a_value_error(stamp: str) -> None:
    with pytest.raises(ValueError):
        snapshot_name(stamp)


def test_parse_results() -> None:
    assert parse_results(_answer([1, 2]), "u") == [1, 2]


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"success": false, "results": []}',
        '{"success": "true", "results": []}',
        '{"success": true}',
        '{"success": true, "results": {}}',
    ],
)
def test_a_malformed_answer_is_a_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        parse_results(text, "u")


class TestLatestSnapshotDir:
    def test_newest_complete_snapshot(self, tmp_path: Path) -> None:
        for name in ("20261105T200000Z", "20261106T200528Z", "20261107T200000Z"):
            (tmp_path / name).mkdir()
        for name in ("20261105T200000Z", "20261106T200528Z"):
            (tmp_path / name / "last-updated.txt").write_text(_STAMP)
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "last-updated.txt").write_text(_STAMP)

        assert TcgcsvCatalogDownloader.latest_snapshot_dir(tmp_path) == (
            tmp_path / "20261106T200528Z"
        )

    def test_none_complete_raises(self, tmp_path: Path) -> None:
        (tmp_path / "20261107T200000Z").mkdir()
        with pytest.raises(FileNotFoundError):
            TcgcsvCatalogDownloader.latest_snapshot_dir(tmp_path)

    def test_a_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TcgcsvCatalogDownloader.latest_snapshot_dir(tmp_path / "absent")


@pytest.mark.parametrize("slugs", [(), ("gundam", "pokemon")])
def test_slugs_must_be_known_games(tmp_path: Path, slugs: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        _make_downloader(tmp_path, slugs=slugs)


def test_every_category_id_is_distinct() -> None:
    assert len(set(CATEGORY_IDS.values())) == len(CATEGORY_IDS)
