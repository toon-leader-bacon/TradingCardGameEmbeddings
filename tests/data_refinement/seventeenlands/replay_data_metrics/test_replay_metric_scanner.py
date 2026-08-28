from src.data_refinement.seventeenlands.replay_data_metrics.replay_metric_scanner import (
    ReplayMetricScanner,
)

# Only default_output_path() is tested here — ReplayMetricScanner's
# scan()/checkpoint mechanics have no dedicated test suite yet
# (out of scope for this session's DEFAULT_OUTPUT_DIR/NAME addition).


class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = ReplayMetricScanner.DEFAULT_OUTPUT_DIR / "MSH.PremierDraft.parquet"
        assert ReplayMetricScanner.default_output_path("MSH", "PremierDraft") == expected

    def test_varies_by_expansion_and_format(self) -> None:
        assert ReplayMetricScanner.default_output_path(
            "MSH", "PremierDraft"
        ) != ReplayMetricScanner.default_output_path("MSH", "TradDraft")
