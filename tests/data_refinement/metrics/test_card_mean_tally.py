from uuid import uuid4

from src.data_refinement.metrics.card_mean_tally import CardMeanTally


class TestCardMeanTally:
    def test_writes_each_cards_mean_and_sample_count_in_first_added_order(
        self,
    ) -> None:
        first, second = uuid4(), uuid4()
        tally = CardMeanTally()
        tally.add(first, 1.0)
        tally.add(second, 3)
        tally.add(first, 0.0)

        frame = tally.mean_frame("win_rate")

        assert frame.to_dict("records") == [
            {"nocab_uuid": str(first), "win_rate": 0.5, "sample_count": 2},
            {"nocab_uuid": str(second), "win_rate": 3.0, "sample_count": 1},
        ]

    def test_empty_tally_still_has_the_columns(self) -> None:
        frame = CardMeanTally().mean_frame("veto_rate")

        assert frame.empty
        assert list(frame.columns) == ["nocab_uuid", "veto_rate", "sample_count"]
        assert frame["veto_rate"].dtype == "float64"
        assert frame["sample_count"].dtype == "int64"
