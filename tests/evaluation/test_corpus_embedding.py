import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from src.encoder_model.embedding_head import LinearEmbeddingHead
from src.encoder_model.multi_card_model import MultiCardModel
from src.encoder_model.single_card_model import SingleCardModel
from src.evaluation.corpus_embedding import CardEmbedder, embed_corpus
from src.evaluation.embedding_table import (
    CardRow,
    EmbeddingTable,
    EmbeddingTableMetadata,
)
from src.schema.game_id import GameId
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder
from tests.evaluation.fakes import WIDTH, FakeEmbedder, make_card, vector_for


def _table(tmp_path: Path, width: int = WIDTH, name: str = "a.db") -> EmbeddingTable:
    metadata = EmbeddingTableMetadata(
        encoder_label="fake",
        checkpoint_dir=None,
        embedding_dim=width,
        binder_versions={GameId.MTG: "mtg-v1"},
        created_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )
    return EmbeddingTable.create(tmp_path / name, metadata)


_CARDS = [make_card(f"card{i}") for i in range(7)]


def test_embeds_every_card_in_batches(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    with _table(tmp_path) as table:
        added = embed_corpus(embedder, _CARDS, table, batch_size=3)
        stored = table.vectors_for(
            [CardRow(c.nocab_uuid, c.source_game) for c in _CARDS]
        )
    assert added == 7
    assert [len(call) for call in embedder.calls] == [3, 3, 1]
    assert np.array_equal(stored, np.array([vector_for(c.name) for c in _CARDS]))


def test_a_rerun_embeds_nothing_and_a_new_card_only_itself(tmp_path: Path) -> None:
    extra = make_card("extra")
    with _table(tmp_path) as table:
        embed_corpus(FakeEmbedder(), _CARDS, table, batch_size=3)
        rerun = FakeEmbedder()
        assert embed_corpus(rerun, _CARDS, table, batch_size=3) == 0
        assert rerun.calls == []
        extending = FakeEmbedder()
        assert embed_corpus(extending, _CARDS + [extra], table, batch_size=3) == 1
        assert extending.calls == [["extra"]]


def test_a_repeated_card_is_embedded_once(tmp_path: Path) -> None:
    card = _CARDS[0]
    embedder = FakeEmbedder()
    with _table(tmp_path) as table:
        assert embed_corpus(embedder, [card, card, card], table, batch_size=2) == 1
    assert embedder.calls == [[card.name]]


def test_an_empty_corpus_adds_nothing(tmp_path: Path) -> None:
    with _table(tmp_path) as table:
        assert embed_corpus(FakeEmbedder(), [], table, batch_size=4) == 0


@pytest.mark.parametrize(
    ("fail_calls", "nan_calls"),
    [(frozenset({2}), frozenset()), (frozenset(), frozenset({2}))],
)
def test_a_failed_batch_is_logged_and_skipped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    fail_calls: frozenset[int],
    nan_calls: frozenset[int],
) -> None:
    embedder = FakeEmbedder(fail_calls=fail_calls, nan_calls=nan_calls)
    with caplog.at_level(logging.WARNING):
        with _table(tmp_path) as table:
            added = embed_corpus(embedder, _CARDS, table, batch_size=3)
            skipped = _CARDS[3:6]
            stored = {row.nocab_uuid for row in table.rows()}
    assert added == 4
    assert stored == {c.nocab_uuid for c in _CARDS} - {c.nocab_uuid for c in skipped}
    assert "skipped a batch of 3 cards" in caplog.text


def test_a_rerun_retries_the_skipped_cards(tmp_path: Path) -> None:
    with _table(tmp_path) as table:
        embed_corpus(FakeEmbedder(fail_calls=frozenset({1})), _CARDS, table, 3)
        retry = FakeEmbedder()
        assert embed_corpus(retry, _CARDS, table, 3) == 3
        assert len(table.rows()) == 7


def test_consecutive_failures_stop_the_run(tmp_path: Path) -> None:
    embedder = FakeEmbedder(fail_calls=frozenset({1, 2}))
    with _table(tmp_path) as table:
        with pytest.raises(RuntimeError, match="2 consecutive") as caught:
            embed_corpus(embedder, _CARDS, table, 3, max_consecutive_failures=2)
        assert table.rows() == []
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_a_success_resets_the_failure_count(tmp_path: Path) -> None:
    cards = [make_card(f"c{i}") for i in range(5)]
    embedder = FakeEmbedder(fail_calls=frozenset({1, 3, 5}))
    with _table(tmp_path) as table:
        added = embed_corpus(embedder, cards, table, 1, max_consecutive_failures=2)
    assert added == 2


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"batch_size": 2, "max_consecutive_failures": 0}]
)
def test_invalid_counts_are_rejected_up_front(tmp_path: Path, kwargs: dict) -> None:
    embedder = FakeEmbedder()
    with _table(tmp_path) as table:
        with pytest.raises(ValueError):
            embed_corpus(embedder, _CARDS, table, **kwargs)
    assert embedder.calls == []


def test_a_width_mismatch_is_rejected_up_front(tmp_path: Path) -> None:
    embedder = FakeEmbedder(width=WIDTH + 1)
    with _table(tmp_path) as table:
        with pytest.raises(ValueError, match="width"):
            embed_corpus(embedder, _CARDS, table, 3)
    assert embedder.calls == []


def test_a_card_of_another_game_propagates_and_keeps_earlier_batches(
    tmp_path: Path,
) -> None:
    foreign = make_card("foreign", GameId.GWENT)
    with _table(tmp_path) as table:
        with pytest.raises(ValueError, match="gwent"):
            embed_corpus(FakeEmbedder(), _CARDS[:3] + [foreign], table, 3)
        assert len(table.rows()) == 3


@pytest.mark.parametrize("model_class", [SingleCardModel, MultiCardModel])
def test_both_encoder_models_embed_a_corpus(tmp_path: Path, model_class: type) -> None:
    torch.manual_seed(0)
    head = LinearEmbeddingHead(HIDDEN, 8)
    if model_class is MultiCardModel:
        model = MultiCardModel(FakeTextEncoder(), head, num_heads=2, num_layers=1)
    else:
        model = SingleCardModel(FakeTextEncoder(), head)
    embedder: CardEmbedder = model
    with _table(tmp_path, width=8) as table:
        assert embed_corpus(embedder, _CARDS, table, batch_size=4) == 7
        rows = [CardRow(c.nocab_uuid, c.source_game) for c in _CARDS]
        stored = table.vectors_for(rows)
    assert np.allclose(stored, model.isolated_embeddings(_CARDS).numpy(), atol=1e-5)


def test_float64_output_beyond_float32_range_is_skipped_not_fatal(
    tmp_path: Path,
) -> None:
    class HugeEmbedder(FakeEmbedder):
        def isolated_embeddings(self, cards, precision="fp32"):  # type: ignore[no-untyped-def]
            result = super().isolated_embeddings(cards, precision).double()
            if len(self.calls) == 1:
                result[0, 0] = 1e39
            return result

    with _table(tmp_path) as table:
        assert embed_corpus(HugeEmbedder(), _CARDS, table, batch_size=3) == 4
