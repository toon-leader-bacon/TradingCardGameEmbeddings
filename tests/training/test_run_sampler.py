import csv
from pathlib import Path

import pytest

from src.encoder_model.feature_cache.feature_store import TierStats
from src.encoder_model.feature_cache.pooled_feature_cache import FeatureCacheStats
from src.training.recording.run_sampler import RunSampler, SampledSources


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stats(texts: int) -> FeatureCacheStats:
    return FeatureCacheStats(
        texts=texts,
        encoded=2,
        admitted=2,
        tiers=(TierStats("vram", 5, 10, 100, 1, 0, 3),),
        uncacheable=1,
    )


class TestRunSampler:
    def test_writes_a_first_and_a_last_row(self, tmp_path: Path) -> None:
        asked = iter([10, 20, 30])
        sampler = RunSampler(
            tmp_path / "samples.csv",
            SampledSources(
                seconds_since_open=lambda: 1.0,
                cache_stats=lambda: _stats(next(asked)),
                steps_attempted=lambda: 4,
            ),
            interval_seconds=3600,
        )
        sampler.start()
        sampler.stop()
        sampler.stop()  # idempotent

        rows = _rows(tmp_path / "samples.csv")
        assert [row["cache_texts"] for row in rows] == ["20", "30"]
        assert rows[0]["vram_hits"] == "5" and rows[0]["vram_evicted"] == "3"
        assert rows[0]["steps_attempted"] == "4"

    def test_a_cacheless_run_has_no_tier_columns(self, tmp_path: Path) -> None:
        sampler = RunSampler(
            tmp_path / "samples.csv",
            SampledSources(lambda: 0.0, lambda: None, lambda: 0),
            interval_seconds=3600,
        )
        sampler.start()
        sampler.stop()
        rows = _rows(tmp_path / "samples.csv")
        assert rows and rows[0]["cache_texts"] == ""
        assert not any(key.endswith("_hits") for key in rows[0])

    def test_a_failing_source_skips_the_row(self, tmp_path: Path) -> None:
        def broken() -> int:
            raise RuntimeError("source down")

        sampler = RunSampler(
            tmp_path / "samples.csv",
            SampledSources(lambda: 0.0, lambda: None, broken),
            interval_seconds=3600,
        )
        sampler.start()
        sampler.stop()
        assert _rows(tmp_path / "samples.csv") == []

    def test_rejects_a_non_positive_interval(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            RunSampler(
                tmp_path / "s.csv",
                SampledSources(lambda: 0.0, lambda: None, lambda: 0),
                interval_seconds=0,
            )

    def test_cannot_start_twice(self, tmp_path: Path) -> None:
        sampler = RunSampler(
            tmp_path / "s.csv",
            SampledSources(lambda: 0.0, lambda: None, lambda: 0),
            interval_seconds=3600,
        )
        sampler.start()
        with pytest.raises(RuntimeError):
            sampler.start()
        sampler.stop()
