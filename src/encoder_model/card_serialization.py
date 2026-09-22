"""Turns a GenericCard into the text string encoder_model's text encoders
consume - generic across any onboarded TCG, no per-game logic.
"""

import json

from src.schema.card import GenericCard


def serialize_card_to_json(card: GenericCard) -> str:
    """
    Inputs: card, any GenericCard from any onboarded game.
    Output: the card's raw_content as compact JSON: no spaces after "," or
        ":" and non-ASCII characters kept as themselves rather than escaped
        (e.g. an em dash, not \u2014). Both cost fewer tokens than
        json.dumps' defaults (~17% fewer on MTG cards) and change nothing
        the encoder reads.
    Side effects: none.
    Exceptions: none beyond whatever json.dumps raises for a
        non-JSON-serializable raw_content value.

    Example:
        >>> serialize_card_to_json(card)
        '{"name":"Lightning Bolt","mana_cost":"{R}","type_line":"Instant"}'
    """
    # payload = {
    #     "name": card.name,
    #     "source_game": card.source_game.value,
    #     "raw_content": card.raw_content,
    # }
    payload = card.raw_content
    # sort_keys intentionally left False (the default): a dojo Mod may
    # deliberately shuffle raw_content's key order as an augmentation, and
    # sorting here would silently undo that.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
