import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.pokemon_tcg.downloader import (
    PokemonTcgDataDownloadResult,
    PokemonTcgDataDownloader,
)

_REPO_ZIP_URL = "https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/zipball"
_TOP_LEVEL_FOLDER = "PokemonTCG-pokemon-tcg-data-abc1234"


def _make_downloader(raw_data_dir: Path) -> PokemonTcgDataDownloader:
    return PokemonTcgDataDownloader(
        repo_zip_url=_REPO_ZIP_URL, raw_data_dir=raw_data_dir
    )


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def _write_fixture_zip(zip_path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for arcname, content in members.items():
            zip_file.writestr(f"{_TOP_LEVEL_FOLDER}/{arcname}", content)
    return zip_path


class TestDownload:
    def test_writes_streamed_response_to_raw_data_dir(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([b"zip-bytes-one-", b"zip-bytes-two"])

        with patch("requests.get", return_value=response) as mock_get:
            result_path = downloader.download()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == _REPO_ZIP_URL
        assert result_path == tmp_path / "pokemon-tcg-data.zip"
        assert result_path.read_bytes() == b"zip-bytes-one-zip-bytes-two"

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

    def test_removes_partial_file_on_write_failure(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([b"first-chunk"])
        response.iter_content.side_effect = requests.ConnectionError("dropped")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.ConnectionError):
                downloader.download()

        assert not (tmp_path / "pokemon-tcg-data.zip").exists()


class TestExtract:
    def test_extracts_cards_and_decks_json_flattened(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        zip_path = _write_fixture_zip(
            tmp_path / "pokemon-tcg-data.zip",
            {
                "cards/en/base1.json": b'[{"id": "base1-1"}]',
                "cards/en/base2.json": b'[{"id": "base2-1"}]',
                "cards/en/README.md": b"not json, should be skipped",
                "decks/en/starter1.json": b'{"cards": []}',
                "other/unrelated.json": b'{"irrelevant": true}',
            },
        )

        result = downloader.extract(zip_path)

        assert result == PokemonTcgDataDownloadResult(
            cards_dir=tmp_path / "cards", decks_dir=tmp_path / "decks"
        )
        assert (
            tmp_path / "cards" / "base1.json"
        ).read_bytes() == b'[{"id": "base1-1"}]'
        assert (
            tmp_path / "cards" / "base2.json"
        ).read_bytes() == b'[{"id": "base2-1"}]'
        assert not (tmp_path / "cards" / "README.md").exists()
        assert (tmp_path / "decks" / "starter1.json").read_bytes() == b'{"cards": []}'
        assert not (tmp_path / "decks" / "unrelated.json").exists()

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        missing_path = tmp_path / "does-not-exist.zip"

        with pytest.raises(OSError):
            downloader.extract(missing_path)

    def test_raises_on_invalid_zip(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        bad_path = tmp_path / "not-actually-a-zip.zip"
        bad_path.write_bytes(b"this is plain text, not zip data")

        with pytest.raises(zipfile.BadZipFile):
            downloader.extract(bad_path)

    def test_raises_when_cards_directory_missing_from_archive(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        zip_path = _write_fixture_zip(
            tmp_path / "pokemon-tcg-data.zip",
            {"decks/en/starter1.json": b'{"cards": []}'},
        )

        with pytest.raises(ValueError, match="cards/en"):
            downloader.extract(zip_path)

    def test_removes_only_the_failing_file_on_mid_extraction_failure(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        zip_path = _write_fixture_zip(
            tmp_path / "pokemon-tcg-data.zip",
            {
                "cards/en/base1.json": b'[{"id": "base1-1"}]',
                "cards/en/base2.json": b'[{"id": "base2-1"}]',
            },
        )

        real_copyfileobj = shutil.copyfileobj
        call_count = 0

        def flaky_copyfileobj(source, destination):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("disk full")
            real_copyfileobj(source, destination)

        with patch(
            "src.data_retrieval.pokemon_tcg.downloader.shutil.copyfileobj",
            side_effect=flaky_copyfileobj,
        ):
            with pytest.raises(OSError):
                downloader.extract(zip_path)

        cards_dir = tmp_path / "cards"
        written_files = sorted(p.name for p in cards_dir.iterdir())
        # base1.json (written before the failure) survives; base2.json
        # (failing mid-copy) was cleaned up rather than left truncated.
        assert written_files == ["base1.json"]


class TestPhase1:
    def test_composes_download_and_extract_and_returns_raw_data_dir(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        zip_path = tmp_path / "pokemon-tcg-data.zip"
        extract_result = PokemonTcgDataDownloadResult(
            cards_dir=tmp_path / "cards", decks_dir=tmp_path / "decks"
        )

        with patch.object(
            downloader, "download", return_value=zip_path
        ) as mock_download:
            with patch.object(
                downloader, "extract", return_value=extract_result
            ) as mock_extract:
                result = downloader.phase_1()

        mock_download.assert_called_once_with()
        mock_extract.assert_called_once_with(zip_path)
        assert result == tmp_path
