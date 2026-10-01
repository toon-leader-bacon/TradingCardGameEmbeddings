import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.pokemon_tcg.extraction_stage import (
    PokemonTcgDeckExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _binder(card_ids: list[str]) -> CardBinder:
    binder = CardBinder()
    for card_id in card_ids:
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.POKEMON,
            name=card_id,
            raw_content={},
            provenance=Provenance(
                data_source=DataSource.POKEMON_TCG,
                source_id=card_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder.create(card)
        binder.register_alias(
            GameId.POKEMON, DataSource.POKEMON_TCG, card_id, card.nocab_uuid
        )
    binder.ensure_unknown_card(GameId.POKEMON)
    return binder


def _deck(deck_id: str, counts: dict[str, int]) -> dict:
    return {
        "id": deck_id,
        "name": f"deck {deck_id}",
        "types": ["Fire"],
        "cards": [{"id": card_id, "count": n} for card_id, n in counts.items()],
    }


def _write(directory: Path, name: str, decks: list[dict]) -> None:
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(json.dumps(decks), encoding="utf-8")


def _uuid(binder: CardBinder, card_id: str) -> UUID:
    card = binder.get_by_alias(GameId.POKEMON, DataSource.POKEMON_TCG, card_id)
    assert card is not None
    return card.nocab_uuid


def test_expands_copy_counts_into_a_multiset(tmp_path: Path) -> None:
    binder = _binder(["base1-1", "base1-2"])
    _write(tmp_path, "base1.json", [_deck("d-1", {"base1-1": 3, "base1-2": 1})])
    box = DeckBox()

    changed = PokemonTcgDeckExtractionStage().extract(tmp_path, box, binder)

    deck = box.get_by_uuid(changed[0])
    assert deck is not None and deck.source_game == GameId.POKEMON
    assert Counter(deck.card_nocab_uuids) == {
        _uuid(binder, "base1-1"): 3,
        _uuid(binder, "base1-2"): 1,
    }


def test_reads_every_file_and_every_deck(tmp_path: Path) -> None:
    binder = _binder(["a-1"])
    _write(tmp_path, "a.json", [_deck("d-1", {"a-1": 1}), _deck("d-2", {"a-1": 2})])
    _write(tmp_path, "b.json", [_deck("d-3", {"a-1": 1})])
    box = DeckBox()
    assert len(PokemonTcgDeckExtractionStage().extract(tmp_path, box, binder)) == 3


def test_a_rerun_is_idempotent_and_a_change_updates(tmp_path: Path) -> None:
    binder = _binder(["a-1", "a-2"])
    _write(tmp_path, "a.json", [_deck("d-1", {"a-1": 1})])
    box = DeckBox()
    stage = PokemonTcgDeckExtractionStage()
    first = stage.extract(tmp_path, box, binder)
    assert stage.extract(tmp_path, box, binder) == []

    _write(tmp_path, "a.json", [_deck("d-1", {"a-1": 1, "a-2": 1})])
    assert stage.extract(tmp_path, box, binder) == first
    assert len(list(box.all_uuids(GameId.POKEMON))) == 1


def test_an_unknown_card_becomes_the_unknown_sentinel(tmp_path: Path) -> None:
    binder = _binder(["a-1"])
    _write(tmp_path, "a.json", [_deck("d-1", {"a-1": 1, "missing-0": 2})])
    box = DeckBox()
    changed = PokemonTcgDeckExtractionStage().extract(tmp_path, box, binder)
    deck = box.get_by_uuid(changed[0])
    unknown = binder.get_by_name_single(GameId.POKEMON, CardBinder.UNKNOWN_CARD_NAME)
    assert deck is not None and unknown is not None
    assert Counter(deck.card_nocab_uuids)[unknown.nocab_uuid] == 2


def test_a_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PokemonTcgDeckExtractionStage().extract(
            tmp_path / "nope", DeckBox(), _binder([])
        )
