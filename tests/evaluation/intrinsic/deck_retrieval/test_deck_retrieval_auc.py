import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.evaluation.intrinsic.deck_retrieval.deck_retrieval_auc import (
    DeckAucConfig,
    DeckRetrievalAuc,
)
from src.evaluation.intrinsic.card_row import CardRow
from src.evaluation.intrinsic.deck_retrieval.evaluation_decks import EvaluationDeck
from src.evaluation.intrinsic.embedding.embedding_table import EmbeddingTable
from src.evaluation.intrinsic.embedding.embedding_table_metadata import (
    EmbeddingTableMetadata,
)
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

_WIDTH = 8


def _holdout(tier_ratios: tuple[float, float, float] = (1, 1, 1)) -> HoldoutSpec:
    return HoldoutSpec(seed=0, tier_ratios=tier_ratios, held_out_games=frozenset())


def _table(path: Path, cards: dict[UUID, np.ndarray]) -> EmbeddingTable:
    """An MTG table holding cards (uuid -> vector)."""
    metadata = EmbeddingTableMetadata(
        encoder_label="synthetic",
        checkpoint_dir=None,
        embedding_dim=_WIDTH,
        binder_versions={GameId.MTG: "mtg-v1"},
        created_at=datetime(2026, 10, 16, tzinfo=timezone.utc),
    )
    table = EmbeddingTable.create(path, metadata)
    rows = [CardRow(uuid, GameId.MTG) for uuid in cards]
    table.add(rows, np.array(list(cards.values()), dtype=np.float32))
    return table


def _clustered(
    deck_count: int, deck_size: int = 4, noise: float = 0.02, seed: int = 0
) -> tuple[dict[UUID, np.ndarray], list[EvaluationDeck]]:
    """deck_count decks whose cards cluster around orthogonal directions."""
    rng = np.random.default_rng(seed)
    cards: dict[UUID, np.ndarray] = {}
    decks = []
    for index in range(deck_count):
        center = np.zeros(_WIDTH)
        center[index] = 1.0
        uuids = tuple(uuid4() for _ in range(deck_size))
        for uuid in uuids:
            cards[uuid] = center + noise * rng.standard_normal(_WIDTH)
        decks.append(EvaluationDeck(uuid4(), GameId.MTG, uuids))
    return cards, decks


def _per_deck_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


class TestConfig:
    def test_rejects_zero_bootstrap_draws(self) -> None:
        with pytest.raises(ValueError, match="bootstrap_draws"):
            DeckAucConfig(bootstrap_draws=0)

    def test_rejects_zero_negative_cap(self) -> None:
        with pytest.raises(ValueError, match="max_negatives_per_game"):
            DeckAucConfig(max_negatives_per_game=0)


class TestDeckRetrievalAuc:
    def test_rejects_no_decks(self) -> None:
        with pytest.raises(ValueError, match="at least one deck"):
            DeckRetrievalAuc([], _holdout())

    def test_docstring_example_separated_decks_score_one(self, tmp_path: Path) -> None:
        cards, decks = _clustered(4)
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks, _holdout()).run(table, tmp_path / "out")
        assert result.scalars["deck_auc"] == pytest.approx(1.0)
        assert result.scalars["deck_auc.mtg"] == pytest.approx(1.0)
        assert result.scalars["n_decks"] == 4
        assert result.scalars["n_anchors"] == 16
        assert result.scalars["n_missing_cards"] == 0
        assert result.scalars["n_dropped_decks"] == 0
        assert result.scalars["n_dropped_games"] == 0
        assert {path.name for path in result.files} == {"per_deck.csv", "scalars.json"}

    def test_random_vectors_score_near_chance(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        cards = {uuid4(): rng.standard_normal(_WIDTH) for _ in range(400)}
        uuids = list(cards)
        decks = [
            EvaluationDeck(uuid4(), GameId.MTG, tuple(uuids[i : i + 8]))
            for i in range(0, 400, 8)
        ]
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks, _holdout()).run(table, tmp_path / "out")
        assert result.scalars["deck_auc"] == pytest.approx(0.5, abs=0.05)
        lower, upper = result.scalars["deck_auc_lo95"], result.scalars["deck_auc_hi95"]
        assert lower < result.scalars["deck_auc"] < upper

    def test_per_deck_csv_has_one_row_per_scored_deck(self, tmp_path: Path) -> None:
        cards, decks = _clustered(3)
        with _table(tmp_path / "t.db", cards) as table:
            DeckRetrievalAuc(decks, _holdout()).run(table, tmp_path / "out")
        rows = _per_deck_rows(tmp_path / "out" / "per_deck.csv")
        assert [row["deck_uuid"] for row in rows] == [str(d.deck_uuid) for d in decks]
        assert list(rows[0]) == [
            "deck_uuid",
            "source_game",
            "n_cards",
            "n_train_anchors",
            "n_heldout_anchors",
            "auc",
            "auc_train_anchor",
            "auc_heldout_anchor",
        ]
        for row in rows:
            assert row["source_game"] == "mtg"
            assert int(row["n_train_anchors"]) + int(row["n_heldout_anchors"]) == 4

    def test_all_train_tier_leaves_heldout_out(self, tmp_path: Path) -> None:
        cards, decks = _clustered(3)
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks, _holdout((1, 0, 0))).run(
                table, tmp_path / "out"
            )
        assert "deck_auc.heldout_anchor" not in result.scalars
        assert result.scalars["deck_auc.train_anchor"] == pytest.approx(1.0)
        rows = _per_deck_rows(tmp_path / "out" / "per_deck.csv")
        assert all(row["auc_heldout_anchor"] == "" for row in rows)

    def test_no_train_tier_leaves_train_out(self, tmp_path: Path) -> None:
        cards, decks = _clustered(3)
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks, _holdout((0, 1, 1))).run(
                table, tmp_path / "out"
            )
        assert "deck_auc.train_anchor" not in result.scalars
        assert result.scalars["deck_auc.heldout_anchor"] == pytest.approx(1.0)

    def test_missing_cards_are_skipped_and_counted(self, tmp_path: Path) -> None:
        cards, decks = _clustered(3)
        absent = uuid4()
        decks[0] = EvaluationDeck(
            decks[0].deck_uuid, GameId.MTG, decks[0].card_uuids + (absent,)
        )
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks, _holdout()).run(table, tmp_path / "out")
        assert result.scalars["n_missing_cards"] == 1
        assert result.scalars["n_decks"] == 3
        assert result.scalars["deck_auc"] == pytest.approx(1.0)

    def test_deck_with_one_card_in_table_is_dropped(self, tmp_path: Path) -> None:
        cards, decks = _clustered(3)
        lonely = EvaluationDeck(uuid4(), GameId.MTG, (decks[0].card_uuids[0], uuid4()))
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks + [lonely], _holdout()).run(
                table, tmp_path / "out"
            )
        assert result.scalars["n_decks"] == 3
        assert result.scalars["n_dropped_decks"] == 1
        assert result.scalars["n_missing_cards"] == 1

    def test_game_absent_from_table_is_dropped_and_counted(
        self, tmp_path: Path
    ) -> None:
        cards, decks = _clustered(3)
        gwent = EvaluationDeck(uuid4(), GameId.GWENT, (uuid4(), uuid4()))
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks + [gwent], _holdout()).run(
                table, tmp_path / "out"
            )
        assert result.scalars["n_dropped_games"] == 1
        assert result.scalars["n_dropped_decks"] == 1
        # A dropped deck's cards still count as missing
        assert result.scalars["n_missing_cards"] == 2
        assert "deck_auc.gwent" not in result.scalars

    def test_deck_holding_every_card_has_no_negatives_and_drops(
        self, tmp_path: Path
    ) -> None:
        cards, decks = _clustered(2)
        everything = EvaluationDeck(uuid4(), GameId.MTG, tuple(cards))
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks + [everything], _holdout()).run(
                table, tmp_path / "out"
            )
        assert result.scalars["n_dropped_decks"] == 1

    def test_no_scorable_deck_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        cards, _ = _clustered(2)
        decks = [EvaluationDeck(uuid4(), GameId.MTG, (uuid4(), uuid4()))]
        out = tmp_path / "out"
        with _table(tmp_path / "t.db", cards) as table:
            with pytest.raises(ValueError, match="could be scored"):
                DeckRetrievalAuc(decks, _holdout()).run(table, out)
        assert not out.exists()

    def test_refuses_a_non_empty_output_dir(self, tmp_path: Path) -> None:
        cards, decks = _clustered(2)
        out = tmp_path / "out"
        out.mkdir()
        (out / "old.txt").write_text("x")
        with _table(tmp_path / "t.db", cards) as table:
            with pytest.raises(FileExistsError):
                DeckRetrievalAuc(decks, _holdout()).run(table, out)

    def test_unknown_placeholder_is_never_a_negative(self, tmp_path: Path) -> None:
        cards, decks = _clustered(3)
        # The placeholder sits on deck 0's direction: as a negative it would
        # tie or beat deck 0's mates
        placeholder = CardBinder.unknown_card_uuid(GameId.MTG)
        cards[placeholder] = np.eye(_WIDTH)[0]
        with _table(tmp_path / "t.db", cards) as table:
            result = DeckRetrievalAuc(decks, _holdout()).run(table, tmp_path / "out")
        assert result.scalars["deck_auc"] == pytest.approx(1.0)

    def test_negative_cap_is_seeded(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2)
        cards = {uuid4(): rng.standard_normal(_WIDTH) for _ in range(120)}
        uuids = list(cards)
        decks = [
            EvaluationDeck(uuid4(), GameId.MTG, tuple(uuids[i : i + 6]))
            for i in range(0, 60, 6)
        ]
        config = DeckAucConfig(bootstrap_draws=50, max_negatives_per_game=10)
        results = []
        with _table(tmp_path / "t.db", cards) as table:
            for seed, name in ((0, "a"), (0, "b"), (1, "c")):
                analysis = DeckRetrievalAuc(decks, _holdout(), config, seed=seed)
                results.append(analysis.run(table, tmp_path / name).scalars)
        assert results[0] == results[1]
        assert results[0]["deck_auc"] != results[2]["deck_auc"]

    def test_cap_above_the_game_size_changes_nothing(self, tmp_path: Path) -> None:
        cards, decks = _clustered(4, noise=0.5)
        capped = DeckAucConfig(max_negatives_per_game=10_000)
        with _table(tmp_path / "t.db", cards) as table:
            full = DeckRetrievalAuc(decks, _holdout()).run(table, tmp_path / "a")
            same = DeckRetrievalAuc(decks, _holdout(), capped).run(
                table, tmp_path / "b"
            )
        assert full.scalars == same.scalars
