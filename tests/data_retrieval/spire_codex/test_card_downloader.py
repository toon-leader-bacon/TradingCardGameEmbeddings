from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.spire_codex.card_downloader import SpireCodexCardDownloader


def _make_downloader(raw_data_dir: Path) -> SpireCodexCardDownloader:
    return SpireCodexCardDownloader(raw_data_dir=raw_data_dir)


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class TestInit:
    def test_defaults_cards_url_and_raw_data_dir_when_omitted(self) -> None:
        downloader = SpireCodexCardDownloader()

        assert downloader.cards_url == SpireCodexCardDownloader.DEFAULT_CARDS_URL
        assert downloader.raw_data_dir == SpireCodexCardDownloader.DEFAULT_RAW_DATA_DIR

    def test_honors_explicit_overrides(self, tmp_path: Path) -> None:
        downloader = SpireCodexCardDownloader("https://example.test/cards.json", tmp_path)

        assert downloader.cards_url == "https://example.test/cards.json"
        assert downloader.raw_data_dir == tmp_path


class TestFetch:
    def test_writes_response_to_cards_json(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([b'[{"name": "Strike"}]'])

        with patch("requests.get", return_value=response) as mock_get:
            result_path = downloader.fetch()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == SpireCodexCardDownloader.DEFAULT_CARDS_URL
        assert result_path == tmp_path / "cards.json"
        assert result_path.read_bytes() == b'[{"name": "Strike"}]'

    def test_creates_raw_data_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        response = _mock_streaming_response([b"[]"])

        with patch("requests.get", return_value=response):
            downloader.fetch()

        assert nested_dir.is_dir()

    def test_always_overwrites_existing_cards_json(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "cards.json").write_bytes(b"stale")
        response = _mock_streaming_response([b"fresh"])

        with patch("requests.get", return_value=response):
            downloader.fetch()

        assert (tmp_path / "cards.json").read_bytes() == b"fresh"

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                downloader.fetch()
