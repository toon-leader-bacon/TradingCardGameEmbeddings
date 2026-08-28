from src.data_retrieval.seventeenlands.landing_page_parser import LandingPageParser
from src.data_retrieval.seventeenlands.refs import DataType, SeventeenLandsFileRef

# A real excerpt of matching link syntax, pulled from the actual
# 17lands.com landing page (data/tmp/17Lands.com.html) — confirmed by
# hand to be representative of the live markup, not invented.
_REAL_LINK_EXCERPT = (
    '<a target="_blank" href="https://17lands-public.s3.amazonaws.com/'
    'analysis_data/draft_data/draft_data_public.MSH.PremierDraft.csv.gz" '
    'rel="noreferrer">link</a>'
    '<a target="_blank" href="https://17lands-public.s3.amazonaws.com/'
    'analysis_data/game_data/game_data_public.MSH.PremierDraft.csv.gz" '
    'rel="noreferrer">link</a>'
    '<a target="_blank" href="https://17lands-public.s3.amazonaws.com/'
    'analysis_data/replay_data/replay_data_public.MSH.PremierDraft.csv.gz" '
    'rel="noreferrer">link</a>'
)


class TestParse:
    def test_parses_real_link_excerpt(self) -> None:
        refs = LandingPageParser().parse(_REAL_LINK_EXCERPT)

        assert refs == [
            SeventeenLandsFileRef(
                data_type=DataType.DRAFT,
                expansion="MSH",
                format_code="PremierDraft",
                url=(
                    "https://17lands-public.s3.amazonaws.com/analysis_data/"
                    "draft_data/draft_data_public.MSH.PremierDraft.csv.gz"
                ),
            ),
            SeventeenLandsFileRef(
                data_type=DataType.GAME,
                expansion="MSH",
                format_code="PremierDraft",
                url=(
                    "https://17lands-public.s3.amazonaws.com/analysis_data/"
                    "game_data/game_data_public.MSH.PremierDraft.csv.gz"
                ),
            ),
            SeventeenLandsFileRef(
                data_type=DataType.REPLAY,
                expansion="MSH",
                format_code="PremierDraft",
                url=(
                    "https://17lands-public.s3.amazonaws.com/analysis_data/"
                    "replay_data/replay_data_public.MSH.PremierDraft.csv.gz"
                ),
            ),
        ]

    def test_parses_synthetic_multi_link_snippet_preserving_order(self) -> None:
        html = (
            '<a href="https://17lands-public.s3.amazonaws.com/analysis_data/'
            'game_data/game_data_public.WOE.TradDraft.csv.gz">link</a>'
            "<p>some unrelated text</p>"
            '<a href="https://17lands-public.s3.amazonaws.com/analysis_data/'
            'draft_data/draft_data_public.BLB.PremierDraft.csv.gz">link</a>'
        )

        refs = LandingPageParser().parse(html)

        assert [r.data_type for r in refs] == [DataType.GAME, DataType.DRAFT]
        assert [r.expansion for r in refs] == ["WOE", "BLB"]
        assert [r.format_code for r in refs] == ["TradDraft", "PremierDraft"]

    def test_expansion_code_with_spaces_and_hyphens(self) -> None:
        html = (
            '<a href="https://17lands-public.s3.amazonaws.com/analysis_data/'
            'draft_data/draft_data_public.Cube_-_Powered.PremierDraft.csv.gz">'
            "link</a>"
        )

        refs = LandingPageParser().parse(html)

        assert refs == [
            SeventeenLandsFileRef(
                data_type=DataType.DRAFT,
                expansion="Cube_-_Powered",
                format_code="PremierDraft",
                url=(
                    "https://17lands-public.s3.amazonaws.com/analysis_data/"
                    "draft_data/draft_data_public.Cube_-_Powered.PremierDraft.csv.gz"
                ),
            )
        ]

    def test_returns_empty_list_for_html_with_no_matches(self) -> None:
        refs = LandingPageParser().parse("<html><body>nothing here</body></html>")

        assert refs == []

    def test_returns_empty_list_for_malformed_input(self) -> None:
        refs = LandingPageParser().parse("<<<not html>>>")

        assert refs == []

    def test_ignores_non_csv_gz_links(self) -> None:
        html = (
            '<a href="https://17lands-public.s3.amazonaws.com/analysis_data/'
            'game_data/game_data_public.MSH.PremierDraft.csv">not gzipped</a>'
            '<a href="https://example.com/unrelated.csv.gz">unrelated host</a>'
        )

        refs = LandingPageParser().parse(html)

        assert refs == []
