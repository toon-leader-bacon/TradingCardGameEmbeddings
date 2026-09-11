"""Turns a GenericCard into the text string encoder_model's text encoders
consume - generic across any onboarded TCG, no per-game logic.
"""

import json

from src.schema.card import GenericCard


def serialize_card_to_json(card: GenericCard) -> str:
    """
    Inputs: card, any GenericCard from any onboarded game.
    Output: a JSON string of the card's name, source game, and raw_content.
    Side effects: none.
    Exceptions: none beyond whatever json.dumps raises for a
        non-JSON-serializable raw_content value.

    Example:
        >>> serialize_card_to_json(card)
        '{"name": "Lightning Bolt", "source_game": "mtg", "raw_content": {...}}'
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
    return json.dumps(payload)
