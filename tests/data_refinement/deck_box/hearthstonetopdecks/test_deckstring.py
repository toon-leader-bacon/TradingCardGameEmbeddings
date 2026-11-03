import pytest

from src.data_refinement.deck_box.hearthstonetopdecks.deckstring import (
    decode_deckstring,
)
from tests.data_refinement.deck_box.hearthstonetopdecks.fixtures import (
    encode_deckstring,
)

# A real HearthstoneTopDecks code: Roger's Odd Rogue, HCT 2018
_REAL_CODE = (
    "AAECAaIHCLICyAPn+gKFigOSoASf1ATm5gbH6AYLjAKBwgKfwgLR4QKL5QKSnwT1nwSVoAS/"
    "+AXt5gbJ9AYAAA=="
)


class TestDecodeDeckstring:
    def test_decodes_a_real_code(self) -> None:
        deckstring = decode_deckstring(_REAL_CODE)

        assert deckstring.format_id == 2
        assert deckstring.hero_dbf_ids == (930,)
        assert deckstring.card_count == 30
        assert len(deckstring.card_counts) == 19  # 8 singles + 11 pairs
        assert dict(deckstring.sideboard_counts) == {}

    def test_round_trips_every_copy_block(self) -> None:
        cards = {1: 1, 300: 2, 70000: 3, 5: 1}

        deckstring = decode_deckstring(encode_deckstring(cards, (7, 8), 1))

        assert dict(deckstring.card_counts) == cards
        assert deckstring.hero_dbf_ids == (7, 8)
        assert deckstring.format_id == 1

    def test_keeps_sideboards_out_of_the_deck(self) -> None:
        sideboard = {(90749, 1): 1, (90749, 2): 2, (102983, 3): 3}

        deckstring = decode_deckstring(
            encode_deckstring({90749: 1, 10: 2}, sideboard_counts=sideboard)
        )

        assert dict(deckstring.card_counts) == {90749: 1, 10: 2}
        assert dict(deckstring.sideboard_counts) == sideboard
        assert deckstring.card_count == 3

    def test_tolerates_missing_padding(self) -> None:
        assert decode_deckstring(_REAL_CODE.rstrip("=")).card_count == 30

    def test_is_read_only(self) -> None:
        deckstring = decode_deckstring(_REAL_CODE)

        with pytest.raises(TypeError):
            deckstring.card_counts[1] = 2  # type: ignore[index]

    @pytest.mark.parametrize(
        "deck_code",
        [
            "not base64!!",
            "AAEC",  # header only: no hero count
            "AAICAQ==",  # version 2
            "AQECAQ==",  # nonzero reserved byte
            "AAECgQ==",  # ends mid-varint
        ],
    )
    def test_rejects_malformed_codes(self, deck_code: str) -> None:
        with pytest.raises(ValueError):
            decode_deckstring(deck_code)
