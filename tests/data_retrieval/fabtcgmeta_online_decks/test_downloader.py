from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
import requests

from src.data_retrieval.fabtcgmeta_online_decks.downloader import (
    FabtcgmetaOnlineDecksDownloader,
    all_shard_prefixes,
)
from src.data_retrieval.rate_limiter import RateLimiter

_DOWNLOAD = "src.data_retrieval.fabtcgmeta_online_decks.downloader.download_to_file"


def _make_downloader(
    raw_data_dir: Path, prefixes: tuple[str, ...]
) -> FabtcgmetaOnlineDecksDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return FabtcgmetaOnlineDecksDownloader(
        RateLimiter(requests_per_minute=1_000_000_000),
        raw_data_dir,
        "https://f.test/online",
        prefixes,
    )


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(response=response)


def _fake_download(
    missing_urls: set[str], status_code: int = 404
) -> Callable[..., None]:
    def _download(url: str, destination_path: Path, **_: object) -> None:
        if url in missing_urls:
            raise _http_error(status_code)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(f'{{"url": "{url}"}}', encoding="utf-8")

    return _download


class TestPhase1:
    def test_downloads_every_prefix_in_both_tiers(self, tmp_path: Path) -> None:
        with patch(_DOWNLOAD, side_effect=_fake_download(set())) as download:
            result = _make_downloader(tmp_path, ("000", "abc")).phase_1()

        assert result == tmp_path
        assert [call.args[0] for call in download.call_args_list] == [
            "https://f.test/online/archive/000.json",
            "https://f.test/online/archive/abc.json",
            "https://f.test/online/recent/000.json",
            "https://f.test/online/recent/abc.json",
        ]
        assert (tmp_path / "recent" / "abc.json").is_file()

    def test_skips_a_shard_already_on_disk(self, tmp_path: Path) -> None:
        (tmp_path / "archive").mkdir()
        (tmp_path / "archive" / "000.json").write_text("{}", encoding="utf-8")

        with patch(_DOWNLOAD, side_effect=_fake_download(set())) as download:
            _make_downloader(tmp_path, ("000",)).phase_1()

        assert [call.args[0] for call in download.call_args_list] == [
            "https://f.test/online/recent/000.json"
        ]
        assert (tmp_path / "archive" / "000.json").read_text(encoding="utf-8") == "{}"

    def test_a_missing_shard_is_skipped_not_saved(self, tmp_path: Path) -> None:
        missing = {"https://f.test/online/recent/000.json"}

        with patch(_DOWNLOAD, side_effect=_fake_download(missing)):
            _make_downloader(tmp_path, ("000",)).phase_1()

        assert (tmp_path / "archive" / "000.json").is_file()
        assert not (tmp_path / "recent" / "000.json").exists()

    def test_any_other_http_error_raises(self, tmp_path: Path) -> None:
        failing = {"https://f.test/online/archive/000.json"}

        with patch(_DOWNLOAD, side_effect=_fake_download(failing, 500)):
            with pytest.raises(requests.HTTPError):
                _make_downloader(tmp_path, ("000",)).phase_1()

    def test_a_part_file_is_not_taken_for_a_shard(self, tmp_path: Path) -> None:
        (tmp_path / "archive").mkdir()
        (tmp_path / "archive" / "000.json.part").write_text('{"trunc', encoding="utf-8")

        with patch(_DOWNLOAD, side_effect=_fake_download(set())) as download:
            _make_downloader(tmp_path, ("000",)).phase_1()

        assert download.call_count == 2
        shard = (tmp_path / "archive" / "000.json").read_text(encoding="utf-8")
        assert "archive/000.json" in shard
        assert not (tmp_path / "archive" / "000.json.part").exists()


def test_all_shard_prefixes_cover_three_hex_digits() -> None:
    prefixes = all_shard_prefixes()

    assert len(prefixes) == 4096
    assert (prefixes[0], prefixes[255], prefixes[-1]) == ("000", "0ff", "fff")
