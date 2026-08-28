import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.scryfall.downloader import (
    ScryfallDownloadResult,
    ScryfallOracleDownloader,
)

_SOURCE_URL = (
    "https://data.scryfall.io/oracle-cards/oracle-cards-20260820090157.jsonl.gz"
)


def _make_downloader(raw_data_dir: Path) -> ScryfallOracleDownloader:
    return ScryfallOracleDownloader(source_url=_SOURCE_URL, raw_data_dir=raw_data_dir)


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class TestDownload:
    def test_writes_streamed_response_to_raw_data_dir(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([b"chunk-one-", b"chunk-two"])

        with patch("requests.get", return_value=response) as mock_get:
            result_path = downloader.download()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == _SOURCE_URL
        assert result_path == tmp_path / "oracle-cards-20260820090157.jsonl.gz"
        assert result_path.read_bytes() == b"chunk-one-chunk-two"

    def test_creates_raw_data_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response):
            downloader.download()

        assert nested_dir.is_dir()

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                downloader.download()


class TestExtract:
    def test_decompresses_gzip_to_sibling_jsonl(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        compressed_path = tmp_path / "oracle-cards-20260820090157.jsonl.gz"
        content = b'{"name": "Static Orb"}\n{"name": "Mine Security"}\n'
        compressed_path.write_bytes(gzip.compress(content))

        jsonl_path = downloader.extract(compressed_path)

        assert jsonl_path == tmp_path / "oracle-cards-20260820090157.jsonl"
        assert jsonl_path.read_bytes() == content

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        missing_path = tmp_path / "does-not-exist.jsonl.gz"

        with pytest.raises(OSError):
            downloader.extract(missing_path)

    def test_raises_on_invalid_gzip(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        bad_path = tmp_path / "not-actually-gzip.jsonl.gz"
        bad_path.write_bytes(b"this is plain text, not gzip data")

        with pytest.raises(gzip.BadGzipFile):
            downloader.extract(bad_path)


class TestFetch:
    def test_composes_download_and_extract(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        compressed_path = tmp_path / "oracle-cards-20260820090157.jsonl.gz"
        jsonl_path = tmp_path / "oracle-cards-20260820090157.jsonl"

        with patch.object(
            downloader, "download", return_value=compressed_path
        ) as mock_download:
            with patch.object(
                downloader, "extract", return_value=jsonl_path
            ) as mock_extract:
                result = downloader.fetch()

        mock_download.assert_called_once_with()
        mock_extract.assert_called_once_with(compressed_path)
        assert result == ScryfallDownloadResult(
            compressed_path=compressed_path, jsonl_path=jsonl_path
        )
