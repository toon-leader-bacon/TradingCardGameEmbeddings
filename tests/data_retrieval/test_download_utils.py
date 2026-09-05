from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.download_utils import download_to_file, download_to_string


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = text
    return response


class TestDownloadToFile:
    def test_writes_streamed_response_to_destination_path(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"chunk-one-", b"chunk-two"])

        with patch("requests.get", return_value=response) as mock_get:
            download_to_file("https://example.com/file.bin", destination_path)

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == "https://example.com/file.bin"
        assert mock_get.call_args.kwargs["stream"] is True
        assert destination_path.read_bytes() == b"chunk-one-chunk-two"

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "does" / "not" / "exist" / "file.bin"
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response):
            download_to_file("https://example.com/file.bin", destination_path)

        assert destination_path.exists()

    def test_uses_given_timeout_and_chunk_size(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response) as mock_get:
            download_to_file(
                "https://example.com/file.bin",
                destination_path,
                timeout_seconds=5,
                chunk_size_bytes=64,
            )

        assert mock_get.call_args.kwargs["timeout"] == 5
        response.iter_content.assert_called_once_with(chunk_size=64)

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                download_to_file("https://example.com/file.bin", destination_path)

    def test_removes_partial_file_on_write_failure(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"first-chunk"])
        response.iter_content.side_effect = requests.ConnectionError("dropped")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.ConnectionError):
                download_to_file("https://example.com/file.bin", destination_path)

        assert not destination_path.exists()


class TestDownloadToString:
    def test_returns_response_text(self) -> None:
        response = _mock_text_response('{"hello": "world"}')

        with patch("requests.get", return_value=response) as mock_get:
            body = download_to_string("https://example.com/page.json")

        mock_get.assert_called_once_with(
            "https://example.com/page.json", timeout=60
        )
        assert body == '{"hello": "world"}'

    def test_uses_given_timeout(self) -> None:
        response = _mock_text_response("data")

        with patch("requests.get", return_value=response) as mock_get:
            download_to_string("https://example.com/page.json", timeout_seconds=5)

        assert mock_get.call_args.kwargs["timeout"] == 5

    def test_retries_then_succeeds(self) -> None:
        failing_response = _mock_text_response("")
        failing_response.raise_for_status.side_effect = requests.ConnectionError(
            "dropped"
        )
        succeeding_response = _mock_text_response("data")

        with patch(
            "requests.get", side_effect=[failing_response, succeeding_response]
        ) as mock_get:
            body = download_to_string("https://example.com/page.json")

        assert mock_get.call_count == 2
        assert body == "data"

    def test_raises_after_max_attempts_exhausted(self) -> None:
        response = _mock_text_response("")
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with patch("requests.get", return_value=response) as mock_get:
            with pytest.raises(requests.HTTPError):
                download_to_string("https://example.com/page.json", max_attempts=2)

        assert mock_get.call_count == 2
