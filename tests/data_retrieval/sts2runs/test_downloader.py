import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.sts2runs.downloader import (
    STS2RunsDownloader,
    STS2RunsDownloadResult,
)

_SOURCE_URL = "https://sts2runs.com/downloads/runs-all-before-2026-06.json.gz"


def _make_downloader(raw_data_dir: Path) -> STS2RunsDownloader:
    return STS2RunsDownloader(source_url=_SOURCE_URL, raw_data_dir=raw_data_dir)


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class TestInit:
    def test_defaults_raw_data_dir_when_omitted(self) -> None:
        downloader = STS2RunsDownloader(source_url=_SOURCE_URL)

        assert downloader.source_url == _SOURCE_URL
        assert downloader.raw_data_dir == STS2RunsDownloader.DEFAULT_RAW_DATA_DIR

    def test_defaults_source_url_when_omitted(self) -> None:
        downloader = STS2RunsDownloader()

        assert downloader.source_url == STS2RunsDownloader.DEFAULT_SOURCE_URL

    def test_honors_explicit_raw_data_dir(self, tmp_path: Path) -> None:
        downloader = STS2RunsDownloader(source_url=_SOURCE_URL, raw_data_dir=tmp_path)

        assert downloader.raw_data_dir == tmp_path


class TestDownload:
    def test_writes_streamed_response_to_raw_data_dir(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([b"chunk-one-", b"chunk-two"])

        with patch("requests.get", return_value=response) as mock_get:
            result_path = downloader.download()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == _SOURCE_URL
        assert result_path == tmp_path / "runs-all-before-2026-06.json.gz"
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
    def test_decompresses_gzip_to_sibling_ndjson(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        compressed_path = tmp_path / "runs-all-before-2026-06.json.gz"
        content = b'{"build_id": "v0.98.0"}\n{"build_id": "v0.97.0"}\n'
        compressed_path.write_bytes(gzip.compress(content))

        ndjson_path = downloader.extract(compressed_path)

        assert ndjson_path == tmp_path / "runs-all-before-2026-06.json"
        assert ndjson_path.read_bytes() == content

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        missing_path = tmp_path / "does-not-exist.json.gz"

        with pytest.raises(OSError):
            downloader.extract(missing_path)

    def test_raises_on_invalid_gzip(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        bad_path = tmp_path / "not-actually-gzip.json.gz"
        bad_path.write_bytes(b"this is plain text, not gzip data")

        with pytest.raises(gzip.BadGzipFile):
            downloader.extract(bad_path)


class TestFetch:
    def test_composes_download_and_extract(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        compressed_path = tmp_path / "runs-all-before-2026-06.json.gz"
        ndjson_path = tmp_path / "runs-all-before-2026-06.json"

        with patch.object(
            downloader, "download", return_value=compressed_path
        ) as mock_download:
            with patch.object(
                downloader, "extract", return_value=ndjson_path
            ) as mock_extract:
                result = downloader.fetch()

        mock_download.assert_called_once_with()
        mock_extract.assert_called_once_with(compressed_path)
        assert result == STS2RunsDownloadResult(
            compressed_path=compressed_path, ndjson_path=ndjson_path
        )
