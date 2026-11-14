import hashlib
import json
import logging
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

from src.data_retrieval.download_utils import RESEARCH_USER_AGENT
from src.data_retrieval.huggingface_datasets.downloader import (
    DigestKind,
    FileDigest,
    HubFile,
    HuggingFaceDatasetDownloader,
)
from src.data_retrieval.rate_limiter import RateLimiter

_MODULE = "src.data_retrieval.huggingface_datasets.downloader"
_HUB = "https://hub.test"
_REPO = "owner/cards"
_SHA = "a" * 40
_INFO_URL = f"{_HUB}/api/datasets/{_REPO}"
_TREE_URL = f"{_HUB}/api/datasets/{_REPO}/tree/{_SHA}"


def _blob_oid(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


def _file_entry(name: str, body: bytes) -> dict:
    return {"type": "file", "path": name, "size": len(body), "oid": _blob_oid(body)}


def _response(payload: object, link: str = "") -> MagicMock:
    response = MagicMock()
    response.text = json.dumps(payload)
    response.headers = {"Link": link} if link else {}
    return response


def _fake_hub(pages: dict[str, MagicMock]) -> Callable[..., MagicMock]:
    def _get(url: str, **_: object) -> MagicMock:
        return pages[url]

    return _get


def _fake_download(bodies: dict[str, bytes]) -> Callable[..., None]:
    def _download(url: str, destination_path: Path, **_: object) -> None:
        assert f"/datasets/{_REPO}/resolve/{_SHA}/" in url
        destination_path.write_bytes(bodies[url.rsplit("/", 1)[-1]])

    return _download


def _downloader(raw_data_dir: Path) -> HuggingFaceDatasetDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return HuggingFaceDatasetDownloader(
        _REPO, RateLimiter(requests_per_minute=1_000_000_000), raw_data_dir, _HUB
    )


def _mirror(
    raw_data_dir: Path, entries: list[dict], bodies: dict[str, bytes]
) -> tuple[Path, MagicMock]:
    pages = {_INFO_URL: _response({"sha": _SHA}), _TREE_URL: _response(entries)}
    with (
        patch(f"{_MODULE}.paced_get", side_effect=_fake_hub(pages)),
        patch(
            f"{_MODULE}.download_to_file", side_effect=_fake_download(bodies)
        ) as download,
    ):
        result = _downloader(raw_data_dir).phase_1()
    return result, download


class TestPhase1:
    def test_mirrors_files_and_writes_the_revision_last(self, tmp_path: Path) -> None:
        bodies = {"a.csv": b"x,y\n1,2\n", "b.csv": b"z\n"}
        entries = [_file_entry(name, body) for name, body in bodies.items()]

        result, download = _mirror(tmp_path, entries, bodies)

        assert result == tmp_path
        assert (tmp_path / "a.csv").read_bytes() == bodies["a.csv"]
        assert download.call_count == 2
        assert download.call_args.kwargs["headers"] == {
            "User-Agent": RESEARCH_USER_AGENT
        }
        revision = json.loads((tmp_path / "revision.json").read_text(encoding="utf-8"))
        assert revision == {"repo_id": _REPO, "sha": _SHA, "files": ["a.csv", "b.csv"]}
        assert HuggingFaceDatasetDownloader.is_complete_mirror(tmp_path)

    def test_a_rerun_downloads_only_changed_files_and_drops_unlisted(
        self, tmp_path: Path
    ) -> None:
        old = {"a.csv": b"old\n", "b.csv": b"same\n", "gone.csv": b"bye\n"}
        _mirror(tmp_path, [_file_entry(n, b) for n, b in old.items()], old)
        new = {"a.csv": b"new body\n", "b.csv": b"same\n"}

        _, download = _mirror(
            tmp_path, [_file_entry(n, b) for n, b in new.items()], new
        )

        assert [call.args[1].name for call in download.call_args_list] == ["a.csv"]
        assert (tmp_path / "a.csv").read_bytes() == b"new body\n"
        assert not (tmp_path / "gone.csv").exists()

    def test_skips_directories_gitattributes_and_unsafe_names(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        bodies = {"a.csv": b"1\n"}
        entries = [
            _file_entry("a.csv", b"1\n"),
            {"type": "directory", "path": "data", "size": 0, "oid": "0" * 40},
            _file_entry(".gitattributes", b"*.csv text\n"),
            _file_entry("../escape.csv", b"no\n"),
            _file_entry("revision.json", b"{}"),
            _file_entry("_last_deck_num.txt", b"743050"),
        ]
        bodies["_last_deck_num.txt"] = b"743050"

        with caplog.at_level(logging.WARNING, logger=_MODULE):
            _mirror(tmp_path, entries, bodies)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "_last_deck_num.txt",
            "a.csv",
            "revision.json",
        ]
        warned = " ".join(record.getMessage() for record in caplog.records)
        assert "data" in warned and "escape" in warned and "gitattributes" not in warned

    def test_follows_next_page_links(self, tmp_path: Path) -> None:
        bodies = {"a.csv": b"1\n", "b.csv": b"2\n"}
        page_2 = f"{_TREE_URL}?cursor=abc"
        pages = {
            _INFO_URL: _response({"sha": _SHA}),
            _TREE_URL: _response(
                [_file_entry("a.csv", b"1\n")], link=f'<{page_2}>; rel="next"'
            ),
            page_2: _response([_file_entry("b.csv", b"2\n")]),
        }
        with (
            patch(f"{_MODULE}.paced_get", side_effect=_fake_hub(pages)),
            patch(f"{_MODULE}.download_to_file", side_effect=_fake_download(bodies)),
        ):
            _downloader(tmp_path).phase_1()

        assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["a.csv", "b.csv"]

    def test_an_lfs_file_is_checked_by_sha256(self, tmp_path: Path) -> None:
        body = b"large\n"
        entry = {
            "type": "file",
            "path": "big.csv",
            "size": len(body),
            "oid": "f" * 40,  # the pointer's blob oid, not the content's
            "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)},
        }

        _mirror(tmp_path, [entry], {"big.csv": body})

        assert (tmp_path / "big.csv").read_bytes() == body

    def test_a_wrong_download_is_deleted_and_the_mirror_left_incomplete(
        self, tmp_path: Path
    ) -> None:
        entry = _file_entry("a.csv", b"expected\n")

        with pytest.raises(ValueError, match="a.csv"):
            _mirror(tmp_path, [entry], {"a.csv": b"tampered\n"})

        assert not (tmp_path / "a.csv").exists()
        assert not HuggingFaceDatasetDownloader.is_complete_mirror(tmp_path)

    def test_raises_before_touching_disk_when_nothing_is_kept(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "revision.json").write_text("{}", encoding="utf-8")
        entries = [{"type": "directory", "path": "data", "size": 0, "oid": "0" * 40}]

        with pytest.raises(ValueError, match="no file"):
            _mirror(tmp_path, entries, {})

        assert HuggingFaceDatasetDownloader.is_complete_mirror(tmp_path)

    def test_rejects_a_response_without_a_commit_sha(self, tmp_path: Path) -> None:
        pages = {_INFO_URL: _response({"id": _REPO})}
        with patch(f"{_MODULE}.paced_get", side_effect=_fake_hub(pages)):
            with pytest.raises(ValueError, match="commit sha"):
                _downloader(tmp_path).phase_1()


class TestRawDataDir:
    def test_default_directory_per_repo(self) -> None:
        assert HuggingFaceDatasetDownloader("feregrino/yugioh-decks").raw_data_dir == (
            Path("data/raw/huggingface/feregrino__yugioh-decks")
        )

    @pytest.mark.parametrize("repo_id", ["no-slash", "a/b/c", "../x", "a/.."])
    def test_rejects_a_repo_id_that_is_not_owner_name(self, repo_id: str) -> None:
        with pytest.raises(ValueError, match="repo id"):
            HuggingFaceDatasetDownloader(repo_id)

    def test_require_complete_mirror(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="revision.json"):
            HuggingFaceDatasetDownloader.require_complete_mirror(tmp_path)
        (tmp_path / "revision.json").write_text("{}", encoding="utf-8")
        HuggingFaceDatasetDownloader.require_complete_mirror(tmp_path)


class TestHubFile:
    def test_git_blob_sha1_matches_git(self, tmp_path: Path) -> None:
        path = tmp_path / "hello.txt"
        path.write_bytes(b"hello\n")
        # `git hash-object hello.txt`
        assert DigestKind.GIT_BLOB_SHA1.hex_of(path) == (
            "ce013625030ba8dba906f756967f9e9ca394464a"
        )

    def test_local_mismatch_reasons(self, tmp_path: Path) -> None:
        path = tmp_path / "a.csv"
        hub_file = HubFile(
            "a.csv", 3, FileDigest(DigestKind.GIT_BLOB_SHA1, _blob_oid(b"ab\n"))
        )
        assert hub_file.local_mismatch(path) == "missing"
        path.write_bytes(b"abcd\n")
        assert hub_file.local_mismatch(path) == "size 5, expected 3"
        path.write_bytes(b"xy\n")
        assert "git_blob_sha1" in str(hub_file.local_mismatch(path))
        path.write_bytes(b"ab\n")
        assert hub_file.local_mismatch(path) is None
