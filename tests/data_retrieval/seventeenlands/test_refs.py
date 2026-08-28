from src.data_retrieval.seventeenlands.refs import DataType, SeventeenLandsFileRef


class TestFromKnown:
    def test_builds_url_from_template(self) -> None:
        ref = SeventeenLandsFileRef.from_known(DataType.GAME, "MSH", "PremierDraft")

        assert ref == SeventeenLandsFileRef(
            data_type=DataType.GAME,
            expansion="MSH",
            format_code="PremierDraft",
            url=(
                "https://17lands-public.s3.amazonaws.com/analysis_data/"
                "game_data/game_data_public.MSH.PremierDraft.csv.gz"
            ),
        )

    def test_uses_correct_path_segment_per_data_type(self) -> None:
        draft_ref = SeventeenLandsFileRef.from_known(DataType.DRAFT, "MSH", "Sealed")
        replay_ref = SeventeenLandsFileRef.from_known(DataType.REPLAY, "MSH", "Sealed")

        assert "draft_data/draft_data_public" in draft_ref.url
        assert "replay_data/replay_data_public" in replay_ref.url

    def test_preserves_expansion_and_format_code_verbatim(self) -> None:
        # Real 17Lands expansion codes can contain spaces/hyphens, e.g.
        # "Cube - Powered" — from_known should not sanitize these.
        ref = SeventeenLandsFileRef.from_known(
            DataType.DRAFT, "Cube_-_Powered", "PremierDraft"
        )

        assert ref.expansion == "Cube_-_Powered"
        assert "Cube_-_Powered" in ref.url
