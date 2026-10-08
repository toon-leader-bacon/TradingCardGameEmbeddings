import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.run_record import CardSlot, Sts2Run
from src.data_refinement.metrics.sts2_runs.scanner import (
    RunExclusion,
    RunSource,
    default_run_sources,
    scan_sts2_runs,
    scored_run,
)
from tests.data_refinement.metrics.sts2_runs._runs import binder, raw_run


@pytest.fixture
def parser(tmp_path: Path) -> Sts2RunParser:
    return Sts2RunParser(binder(tmp_path), SpireCodexRunsDeckExtractionStage())


class _RecordingMetric:
    def __init__(self, fail: bool = False) -> None:
        self.run_ids: list[str] = []
        self.finalized = False
        self._fail = fail

    def accumulate(self, run: Sts2Run) -> None:
        if self._fail:
            raise RuntimeError("boom")
        self.run_ids.append(run.run_id)

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused")


class TestScoredRun:
    def test_a_standard_finished_run_is_scored(self, parser: Sts2RunParser) -> None:
        run = parser.parse(raw_run())
        assert scored_run(run) == run

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"game_mode": "daily"}, RunExclusion.NOT_STANDARD),
            ({"_isCheated": True}, RunExclusion.CHEATED),
            ({"was_abandoned": True}, RunExclusion.ABANDONED),
        ],
    )
    def test_conditioning_drops_a_run(
        self, parser: Sts2RunParser, overrides: dict, expected: RunExclusion
    ) -> None:
        assert scored_run(parser.parse(raw_run(**overrides))) is expected

    def test_a_player_with_only_unknown_cards_is_dropped(
        self, parser: Sts2RunParser
    ) -> None:
        run = parser.parse(raw_run())
        unknown_only = dataclasses.replace(
            run.players[0], deck=(CardSlot(None, 1, False),)
        )
        two_players = dataclasses.replace(run, players=(run.players[0], unknown_only))

        scored = scored_run(two_players)

        assert isinstance(scored, Sts2Run)
        assert scored.players == (run.players[0],)
        assert len(two_players.players) == 2  # not mutated

    def test_a_run_with_no_known_cards_is_not_scored(
        self, parser: Sts2RunParser
    ) -> None:
        run = parser.parse(raw_run())
        unknown_only = dataclasses.replace(
            run.players[0], deck=(CardSlot(None, 1, False),)
        )
        no_known = dataclasses.replace(run, players=(unknown_only,))
        assert scored_run(no_known) is RunExclusion.NO_KNOWN_CARDS


def _write_runs(path: Path, runs: list[dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as raw_file:
        for run in runs:
            raw_file.write(json.dumps(run) + "\n")
    return path


class TestScanSts2Runs:
    def test_feeds_scored_runs_to_every_metric_and_tallies_the_rest(
        self, tmp_path: Path
    ) -> None:
        malformed = raw_run(run_hash="3")
        del malformed["players"][0]["deck"][0]["floor_added_to_deck"]
        raw_path = _write_runs(
            tmp_path / "runs.json.gz",
            [
                raw_run(run_hash="1"),
                raw_run(run_hash="2", was_abandoned=True),
                malformed,
            ],
        )
        healthy, failing = _RecordingMetric(), _RecordingMetric(fail=True)

        tally = scan_sts2_runs(
            [RunSource(SpireCodexRunsDeckExtractionStage(), raw_path)],
            binder(tmp_path),
            [failing, healthy],
        )

        assert healthy.run_ids == ["1"]
        assert healthy.finalized and failing.finalized
        assert tally.scored == 1
        assert tally.excluded == {
            RunExclusion.ABANDONED: 1,
            RunExclusion.MALFORMED: 1,
        }

    def test_a_non_dict_deck_entry_is_malformed_not_fatal(self, tmp_path: Path) -> None:
        broken = raw_run(run_hash="1")
        broken["players"][0]["deck"].append("CARD.NOT_A_DICT")
        raw_path = _write_runs(tmp_path / "runs.json.gz", [broken])

        tally = scan_sts2_runs(
            [RunSource(SpireCodexRunsDeckExtractionStage(), raw_path)],
            binder(tmp_path),
            [],
        )

        assert tally.excluded == {RunExclusion.MALFORMED: 1}

    def test_default_source_is_spire_codex(self) -> None:
        stages = [type(source.stage) for source in default_run_sources()]
        assert stages == [SpireCodexRunsDeckExtractionStage]
