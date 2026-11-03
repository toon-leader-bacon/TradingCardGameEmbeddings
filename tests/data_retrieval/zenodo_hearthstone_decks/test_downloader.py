import hashlib
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.zenodo_hearthstone_decks.downloader import (
    ZenodoFile,
    ZenodoHearthstoneDecksDownloader,
)

_DOWNLOAD = "src.data_retrieval.zenodo_hearthstone_decks.downloader.download_to_file"
_DECKS_BODY = b"idDeck;...\n"
_CARDS_BODY = b"idDeck;idCard\n1;Fireball\n"


def _md5(body: bytes) -> str:
    return hashlib.md5(body).hexdigest()


class _FakeDownloader(ZenodoHearthstoneDecksDownloader):
    DECKS_FILE = ZenodoFile("decks.csv", _md5(_DECKS_BODY))
    CARDS_FILE = ZenodoFile("cards.csv", _md5(_CARDS_BODY))


def _make_downloader(raw_data_dir: Path) -> _FakeDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return _FakeDownloader(
        RateLimiter(requests_per_minute=1_000_000_000), raw_data_dir, "https://z.test"
    )


def _fake_download(bodies: dict[str, bytes]) -> Callable[..., None]:
    def _download(url: str, destination_path: Path, **_: object) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(bodies[url.rsplit("/", 1)[-1]])

    return _download


class TestPhase1:
    def test_downloads_both_files(self, tmp_path: Path) -> None:
        bodies = {"decks.csv": _DECKS_BODY, "cards.csv": _CARDS_BODY}

        with patch(_DOWNLOAD, side_effect=_fake_download(bodies)) as download:
            result = _make_downloader(tmp_path).phase_1()

        assert result == tmp_path
        assert [call.args[0] for call in download.call_args_list] == [
            "https://z.test/decks.csv",
            "https://z.test/cards.csv",
        ]
        assert (tmp_path / "cards.csv").read_bytes() == _CARDS_BODY

    def test_skips_a_file_already_on_disk_with_the_right_md5(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "decks.csv").write_bytes(_DECKS_BODY)
        bodies = {"cards.csv": _CARDS_BODY}

        with patch(_DOWNLOAD, side_effect=_fake_download(bodies)) as download:
            _make_downloader(tmp_path).phase_1()

        assert [call.args[0] for call in download.call_args_list] == [
            "https://z.test/cards.csv"
        ]

    def test_redownloads_a_file_with_the_wrong_md5(self, tmp_path: Path) -> None:
        (tmp_path / "decks.csv").write_bytes(b"truncated")
        (tmp_path / "cards.csv").write_bytes(_CARDS_BODY)
        bodies = {"decks.csv": _DECKS_BODY}

        with patch(_DOWNLOAD, side_effect=_fake_download(bodies)):
            _make_downloader(tmp_path).phase_1()

        assert (tmp_path / "decks.csv").read_bytes() == _DECKS_BODY

    def test_a_corrupt_download_raises_and_is_deleted(self, tmp_path: Path) -> None:
        bodies = {"decks.csv": b"corrupt"}

        with patch(_DOWNLOAD, side_effect=_fake_download(bodies)):
            with pytest.raises(ValueError, match="MD5"):
                _make_downloader(tmp_path).phase_1()

        assert not (tmp_path / "decks.csv").exists()
