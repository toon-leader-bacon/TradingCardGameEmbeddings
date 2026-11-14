"""The two feregrino Hugging Face datasets built from YGOPRODeck's public
API (github.com/fferegrino/yu-gi-oh, .../yu-gi-oh-decks), CC-BY-SA-3.0.

Any published dataset derived from them (the Yu-Gi-Oh card binder and
deck box) must credit YGOPRODeck and these datasets
and carry the same licence.

    yugioh-cards (weekly): cards.csv (id = passcode, name, type,
        formats, ...), cards_variants.csv (original,variant:
        alternate-artwork passcodes), cardsets.csv, cards_cardsets.csv.
    yugioh-decks (daily): NNNNNNNN.csv shards of 10,000 deck_num each
        (deck_num, deck_name, format, main_deck, extra_deck, side_deck
        as JSON lists of passcode strings, dates, user, description).

Defined here, on the retrieval side, so the run script and both
data_refinement stages import one spelling of each id.
"""

YGOPRODECK_CARDS_REPO_ID = "feregrino/yugioh-cards"
YGOPRODECK_DECKS_REPO_ID = "feregrino/yugioh-decks"
