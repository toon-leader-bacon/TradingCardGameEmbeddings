import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.encoder_model.card_serialization import serialize_card_to_json
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(raw_content: dict[str, Any]) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="Test",
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id="x",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def test_output_has_no_padding_spaces_after_separators() -> None:
    text = serialize_card_to_json(_card({"name": "Bolt", "colors": ["R", "U"]}))

    assert text == '{"name":"Bolt","colors":["R","U"]}'


def test_non_ascii_characters_are_kept_not_escaped() -> None:
    text = serialize_card_to_json(_card({"type_line": "Creature — Human"}))

    assert "—" in text
    assert "\\u2014" not in text


def test_round_trips_to_the_same_content() -> None:
    content = {"name": "Bolt", "cmc": 1, "faces": [{"text": "Deal 3. • Done"}]}

    assert json.loads(serialize_card_to_json(_card(content))) == content


def test_key_order_is_preserved_not_sorted() -> None:
    text = serialize_card_to_json(_card({"z": 1, "a": 2}))

    assert list(json.loads(text)) == ["z", "a"]


def test_empty_content_serializes_to_empty_object() -> None:
    assert serialize_card_to_json(_card({})) == "{}"
