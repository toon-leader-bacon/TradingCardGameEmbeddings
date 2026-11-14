from src.data_refinement.deck_box.card_matching import CardMatchKind
from src.data_refinement.deck_box.mtg_crawls.crawled_deck import MtgCardRef
from src.data_refinement.deck_box.mtg_crawls.mtg_card_matcher import MtgCardMatcher
from tests.data_refinement.deck_box.mtg_crawls.fixtures import (
    card_uuid,
    make_mtg_binder,
)

_BINDER_CARDS = [
    ("Lightning Bolt", "o-bolt", ("p-bolt-1", "p-bolt-2")),
    ("Bruce Banner // The Hulk", "o-banner", ()),
    ("Shock", "o-shock", ()),
]


class TestMtgCardMatcher:
    def test_oracle_id_first(self) -> None:
        matcher = MtgCardMatcher(make_mtg_binder(_BINDER_CARDS))

        # The oracle id wins over a printing id and a name naming other cards
        matched = matcher.match_card_uuid(MtgCardRef("Shock", "o-bolt", "p-bolt-9"))

        assert matched == card_uuid("o-bolt")
        assert matcher.match_counts == {CardMatchKind.CARD_ID: 1}

    def test_printing_id_then_name(self) -> None:
        matcher = MtgCardMatcher(make_mtg_binder(_BINDER_CARDS))

        by_printing = matcher.match_card_uuid(MtgCardRef("x", None, "p-bolt-2"))
        by_face_name = matcher.match_card_uuid(MtgCardRef("The Hulk", None, "p-new"))
        by_name = matcher.match_card_uuid(MtgCardRef("Shock", "o-missing", None))

        assert (by_printing, by_face_name, by_name) == (
            card_uuid("o-bolt"),
            card_uuid("o-banner"),
            card_uuid("o-shock"),
        )
        assert matcher.match_counts == {
            CardMatchKind.PRINTING: 1,
            CardMatchKind.NAME: 2,
        }

    def test_nothing_matches(self) -> None:
        matcher = MtgCardMatcher(make_mtg_binder(_BINDER_CARDS))
        missing = MtgCardRef("Not A Card", None, "p-none")

        assert matcher.match_card_uuid(missing) is None
        assert matcher.unmatched_counts == {missing: 1}
