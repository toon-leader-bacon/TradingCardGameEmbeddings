import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from src.data_retrieval.mtgjson_decks.downloader import MtgjsonDeckFilesDownloader
from src.data_retrieval.rate_limiter import RateLimiter

_MODULE = "src.data_retrieval.mtgjson_decks.downloader"
_API = "https://m.test"
_META = json.dumps({"meta": {"date": "2026-11-05", "version": "5.3.0+20261105"}})
_BUILD = "5.3.0+20261105"


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for name, body in members.items():
            zip_file.writestr(name, body)
    return buffer.getvalue()


_ZIP = _zip_bytes(
    {
        "AllDeckFiles/ThemeDeck_ABC.json": b'{"data": {"name": "A"}}',
        "AllDeckFiles/CommanderDeck_XYZ.json": b'{"data": {"name": "B"}}',
        "AllDeckFiles/README.txt": b"not a deck",
    }
)


def _make_downloader(raw_data_dir: Path) -> MtgjsonDeckFilesDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return MtgjsonDeckFilesDownloader(
        RateLimiter(requests_per_minute=1_000_000_000), raw_data_dir, _API
    )


def _fake_string(sha256: str, meta: str = _META) -> Callable[..., str]:
    bodies = {f"{_API}/Meta.json": meta, f"{_API}/AllDeckFiles.zip.sha256": sha256}

    def _download(url: str, **_: object) -> str:
        return bodies[url]

    return _download


def _fake_file(body: bytes) -> Callable[..., None]:
    def _download(url: str, destination_path: Path, **_: object) -> None:
        assert url == f"{_API}/AllDeckFiles.zip"
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(body)

    return _download


def _good_sha256() -> str:
    return f"{hashlib.sha256(_ZIP).hexdigest()}  AllDeckFiles.zip\n"


class TestPhase1:
    def test_lands_the_build_with_its_json_files_and_meta(self, tmp_path: Path) -> None:
        with (
            patch(
                f"{_MODULE}.download_to_string",
                side_effect=_fake_string(_good_sha256()),
            ),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(_ZIP)),
        ):
            result = _make_downloader(tmp_path).phase_1()

        assert result == tmp_path / _BUILD
        assert sorted(path.name for path in (result / "decks").iterdir()) == [
            "CommanderDeck_XYZ.json",
            "ThemeDeck_ABC.json",
        ]
        assert (result / "Meta.json").read_text(encoding="utf-8") == _META
        assert not (result / "AllDeckFiles.zip").exists()

    def test_a_landed_build_downloads_nothing_more(self, tmp_path: Path) -> None:
        (tmp_path / _BUILD).mkdir()
        (tmp_path / _BUILD / "Meta.json").write_text(_META, encoding="utf-8")

        with (
            patch(
                f"{_MODULE}.download_to_string", side_effect=_fake_string("")
            ) as text,
            patch(f"{_MODULE}.download_to_file") as file,
        ):
            result = _make_downloader(tmp_path).phase_1()

        assert result == tmp_path / _BUILD
        assert [call.args[0] for call in text.call_args_list] == [f"{_API}/Meta.json"]
        file.assert_not_called()

    def test_a_wrong_sha256_raises_and_deletes_the_zip(self, tmp_path: Path) -> None:
        with (
            patch(f"{_MODULE}.download_to_string", side_effect=_fake_string("0" * 64)),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(_ZIP)),
            pytest.raises(ValueError, match="SHA-256"),
        ):
            _make_downloader(tmp_path).phase_1()

        assert not (tmp_path / _BUILD / "AllDeckFiles.zip").exists()
        assert not (tmp_path / _BUILD / "Meta.json").exists()

    def test_a_zip_without_json_raises_and_leaves_the_build_incomplete(
        self, tmp_path: Path
    ) -> None:
        empty_zip = _zip_bytes({"README.txt": b"x"})
        sha256 = hashlib.sha256(empty_zip).hexdigest()
        with (
            patch(f"{_MODULE}.download_to_string", side_effect=_fake_string(sha256)),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(empty_zip)),
            pytest.raises(ValueError, match="no JSON"),
        ):
            _make_downloader(tmp_path).phase_1()

        assert not (tmp_path / _BUILD / "Meta.json").exists()

    @pytest.mark.parametrize(
        "meta",
        [
            "not json",
            json.dumps({"meta": {}}),
            json.dumps({"meta": {"version": ""}}),
            json.dumps({"meta": {"version": "../escape"}}),
        ],
    )
    def test_a_bad_meta_raises(self, tmp_path: Path, meta: str) -> None:
        with (
            patch(f"{_MODULE}.download_to_string", side_effect=_fake_string("", meta)),
            patch(f"{_MODULE}.download_to_file") as file,
            pytest.raises(ValueError, match="meta.version"),
        ):
            _make_downloader(tmp_path).phase_1()

        file.assert_not_called()

    def test_phase_2_returns_the_build_dir(self, tmp_path: Path) -> None:
        (tmp_path / _BUILD).mkdir()
        (tmp_path / _BUILD / "Meta.json").write_text(_META, encoding="utf-8")
        downloader = _make_downloader(tmp_path)

        with patch(f"{_MODULE}.download_to_string", side_effect=_fake_string("")):
            downloader.phase_1()

        assert downloader.phase_2() == tmp_path / _BUILD


class TestPhase1Guards:
    @pytest.mark.parametrize("checksum", ["", "   \n", "<html>error</html>", "abc"])
    def test_a_bad_checksum_file_raises_before_the_zip_download(
        self, tmp_path: Path, checksum: str
    ) -> None:
        with (
            patch(f"{_MODULE}.download_to_string", side_effect=_fake_string(checksum)),
            patch(f"{_MODULE}.download_to_file") as file,
            pytest.raises(ValueError, match="no SHA-256 digest"),
        ):
            _make_downloader(tmp_path).phase_1()

        file.assert_not_called()

    def test_an_uppercase_checksum_verifies(self, tmp_path: Path) -> None:
        with (
            patch(
                f"{_MODULE}.download_to_string",
                side_effect=_fake_string(_good_sha256().upper()),
            ),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(_ZIP)),
        ):
            result = _make_downloader(tmp_path).phase_1()

        assert (result / "Meta.json").is_file()

    def test_a_rebuild_mid_download_raises_and_leaves_the_build_incomplete(
        self, tmp_path: Path
    ) -> None:
        newer = json.dumps({"meta": {"version": "5.3.0+20261106"}})
        bodies = iter([_META, _good_sha256(), newer])
        with (
            patch(
                f"{_MODULE}.download_to_string",
                side_effect=lambda url, **_: next(bodies),
            ),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(_ZIP)),
            pytest.raises(ValueError, match="rebuilt"),
        ):
            _make_downloader(tmp_path).phase_1()

        assert not (tmp_path / _BUILD / "Meta.json").exists()

    def test_a_member_path_cannot_escape_the_decks_dir(self, tmp_path: Path) -> None:
        escaping_zip = _zip_bytes({"AllDeckFiles/../../escape.json": b"{}"})
        sha256 = hashlib.sha256(escaping_zip).hexdigest()
        raw_dir = tmp_path / "raw"
        with (
            patch(f"{_MODULE}.download_to_string", side_effect=_fake_string(sha256)),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(escaping_zip)),
        ):
            result = _make_downloader(raw_dir).phase_1()

        assert (result / "decks" / "escape.json").is_file()
        assert sorted(path.name for path in tmp_path.iterdir()) == ["raw"]
        assert sorted(path.name for path in result.iterdir()) == ["Meta.json", "decks"]

    def test_two_members_with_one_base_name_raise(self, tmp_path: Path) -> None:
        clashing_zip = _zip_bytes({"A/x.json": b"{}", "B/x.json": b"{}"})
        sha256 = hashlib.sha256(clashing_zip).hexdigest()
        with (
            patch(f"{_MODULE}.download_to_string", side_effect=_fake_string(sha256)),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(clashing_zip)),
            pytest.raises(ValueError, match="two members named x.json"),
        ):
            _make_downloader(tmp_path).phase_1()


def _land_build(root: Path, name: str, complete: bool = True) -> Path:
    (root / name / "decks").mkdir(parents=True)
    if complete:
        (root / name / "Meta.json").write_text(_META, encoding="utf-8")
    return root / name / "decks"


class TestLatestBuildDecksDir:
    def test_the_only_build(self, tmp_path: Path) -> None:
        decks = _land_build(tmp_path, _BUILD)
        assert MtgjsonDeckFilesDownloader.latest_build_decks_dir(tmp_path) == decks

    def test_newest_by_date_not_by_name(self, tmp_path: Path) -> None:
        _land_build(tmp_path, "5.10.0+20261101")
        newest = _land_build(tmp_path, "5.3.0+20261105")
        assert MtgjsonDeckFilesDownloader.latest_build_decks_dir(tmp_path) == newest

    def test_one_date_is_ordered_by_name(self, tmp_path: Path) -> None:
        _land_build(tmp_path, "5.3.0+20261105")
        later = _land_build(tmp_path, "5.3.1+20261105")
        assert MtgjsonDeckFilesDownloader.latest_build_decks_dir(tmp_path) == later

    def test_an_incomplete_build_is_skipped(self, tmp_path: Path) -> None:
        complete = _land_build(tmp_path, "5.3.0+20261104")
        _land_build(tmp_path, "5.3.0+20261105", complete=False)
        assert MtgjsonDeckFilesDownloader.latest_build_decks_dir(tmp_path) == complete

    def test_an_undated_directory_is_skipped(self, tmp_path: Path) -> None:
        dated = _land_build(tmp_path, "5.3.0+20261104")
        _land_build(tmp_path, "scratch")
        assert MtgjsonDeckFilesDownloader.latest_build_decks_dir(tmp_path) == dated

    @pytest.mark.parametrize("make_root", [False, True])
    def test_no_complete_build_raises(self, tmp_path: Path, make_root: bool) -> None:
        root = tmp_path / "raw"
        if make_root:
            _land_build(root, _BUILD, complete=False)
        with pytest.raises(FileNotFoundError):
            MtgjsonDeckFilesDownloader.latest_build_decks_dir(root)
