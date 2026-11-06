"""DeckRetrievalAuc: how well an embedding keeps a deck's cards together -
the contrastive dojos' task, measured on a frozen encoder with no head.

For an anchor card in a deck: the probability that a random deck-mate is
closer (cosine) to it than a random same-game card outside the deck; 0.5
is chance. Scoring is in deck_score.py, the scalars in deck_auc_summary.py.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence
from uuid import UUID

import numpy as np

from src.data_refinement.card_binder.card_binder import CardBinder
from src.evaluation.intrinsic.deck_retrieval.deck_auc_summary import (
    summarize_deck_scores,
)
from src.evaluation.intrinsic.deck_retrieval.deck_score import DeckScore, anchor_aucs
from src.evaluation.intrinsic.embedding_analysis import (
    AnalysisResult,
    require_fresh_output_dir,
    write_analysis_result,
)
from src.evaluation.intrinsic.unit_length import scale_to_unit_length
from src.evaluation.intrinsic.card_row import CardRow
from src.evaluation.intrinsic.deck_retrieval.evaluation_decks import EvaluationDeck
from src.evaluation.intrinsic.embedding.embedding_table import EmbeddingTable
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec

_PER_DECK_FILE = "per_deck.csv"
_PER_DECK_COLUMNS = (
    "deck_uuid",
    "source_game",
    "n_cards",
    "n_train_anchors",
    "n_heldout_anchors",
    "auc",
    "auc_train_anchor",
    "auc_heldout_anchor",
)


@dataclass(frozen=True)
class DeckAucConfig:
    """Tunables for DeckRetrievalAuc.

    bootstrap_draws: draws for every 95% interval.
    max_negatives_per_game: None scores every other same-game card in the
        table as a negative; a number draws a seeded sample of that many
        per game once, shared by all of that game's decks (an
        approximation, for very large games).

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: ValueError if bootstrap_draws < 1 or
        max_negatives_per_game < 1.
    """

    bootstrap_draws: int = 1000
    max_negatives_per_game: int | None = None

    def __post_init__(self) -> None:
        if self.bootstrap_draws < 1:
            raise ValueError(
                f"bootstrap_draws must be >= 1, got {self.bootstrap_draws}"
            )
        cap = self.max_negatives_per_game
        if cap is not None and cap < 1:
            raise ValueError(f"max_negatives_per_game must be >= 1, got {cap}")


# eq=False: numpy fields have no single-bool equality
@dataclass(frozen=True, eq=False)
class _GameCards:
    """One game's cards in the table, ready for scoring.

    unit_vectors: float64 (n, d), row i is uuids[i] at unit length (float64
        so every cosine is computed at one precision, from one array).
    negative_rows: int rows eligible as negatives (the Unknown placeholder
        excluded; capped by max_negatives_per_game).
    row_of (derived, not a field): uuid -> row in unit_vectors.

    Side effects: none.
    Exceptions: ValueError if unit_vectors has a row count other than
        len(uuids).
    """

    uuids: tuple[UUID, ...]
    unit_vectors: np.ndarray
    negative_rows: np.ndarray

    def __post_init__(self) -> None:
        if len(self.unit_vectors) != len(self.uuids):
            raise ValueError(
                f"{len(self.unit_vectors)} vectors for {len(self.uuids)} uuids"
            )

    @cached_property
    def row_of(self) -> dict[UUID, int]:
        """Inputs: none. Output: uuid -> row. Side effects: none (cached).
        Exceptions: none."""
        return {uuid: row for row, uuid in enumerate(self.uuids)}


class DeckRetrievalAuc:
    """Same-deck retrieval AUC over fixed decks (satisfies EmbeddingAnalysis).

    Each anchor's AUC is a Mann-Whitney statistic (deck_score.anchor_aucs);
    anchors average into a deck AUC, decks into a game AUC, games into
    deck_auc. Anchors also split by holdout tier: a non-TRAIN anchor was
    never in a training example, though its deck-mates may have been.

    A deck card missing from the table is skipped and counted. A deck is
    dropped (counted, not an error) when fewer than two of its cards are in
    the table or its game has no negative left.

    Scalars: see summarize_deck_scores, plus "n_decks", "n_anchors",
    "n_missing_cards", "n_dropped_decks", "n_dropped_games" (deck games
    with no scored deck; left out of the per-game scalars). Also writes
    per_deck.csv (one row per scored deck), for paired per-deck comparison
    across encoders scored on the same decks. name = "deck_retrieval_auc".

    Inputs (constructor): decks (from sample_evaluation_decks); holdout
        (the one the decks were sampled with, for anchor tiers - nothing
        checks this); config; seed (split into independent streams for
        negative sampling and the bootstrap).
    """

    def __init__(
        self,
        decks: Sequence[EvaluationDeck],
        holdout: HoldoutSpec,
        config: DeckAucConfig = DeckAucConfig(),
        seed: int = 0,
    ) -> None:
        """Side effects: none. Exceptions: ValueError if decks is empty."""
        if not decks:
            raise ValueError("DeckRetrievalAuc needs at least one deck")
        self.name = "deck_retrieval_auc"
        self._decks = tuple(decks)
        self._holdout = holdout
        self._config = config
        self._seed = seed

    def run(self, table: EmbeddingTable, output_dir: Path) -> AnalysisResult:
        """See EmbeddingAnalysis.run. Writes per_deck.csv and scalars.json.
        Raises ValueError (nothing written) if no deck can be scored.

        Example:
            >>> DeckRetrievalAuc(decks, holdout).run(
            ...     table, out / "deck_retrieval_auc").scalars["deck_auc"]
            0.83
        """
        scores: list[DeckScore] = []
        missing_cards = 0
        require_fresh_output_dir(output_dir)
        negative_seed, bootstrap_seed = np.random.SeedSequence(self._seed).spawn(2)
        cards_by_game = self._game_cards(table, negative_seed)

        # Score each deck on the cards the table holds
        for deck in self._decks:
            cards = cards_by_game[deck.source_game]
            member_rows = _member_rows(deck, cards)
            missing_cards += len(deck.card_uuids) - len(member_rows)
            score = self._score_deck(deck, member_rows, cards)
            if score is not None:
                scores.append(score)
        if not scores:
            raise ValueError(f"none of {len(self._decks)} decks could be scored")

        # Summarize, then write the per-deck rows and the scalars
        scalars = summarize_deck_scores(
            scores, self._config.bootstrap_draws, bootstrap_seed
        )
        scalars.update(_count_scalars(scores, missing_cards, self._decks))
        output_dir.mkdir(parents=True, exist_ok=True)
        per_deck = _write_per_deck_csv(output_dir / _PER_DECK_FILE, scores)
        return write_analysis_result(output_dir, scalars, extra_files=(per_deck,))

    def _game_cards(
        self, table: EmbeddingTable, seed: np.random.SeedSequence
    ) -> dict[GameId, _GameCards]:
        """Every deck game's cards in the table (a game with none gets an
        empty _GameCards, so its decks drop rather than fail).

        Inputs: table; seed (for the negative cap).
        Output: GameId -> _GameCards for each deck game.
        Side effects: reads table's rows and vectors.
        Exceptions: none of its own (sqlite3.Error from the table propagates).
        """
        result: dict[GameId, _GameCards] = {}
        rows_by_game: dict[GameId, list[CardRow]] = defaultdict(list)
        rng = np.random.default_rng(seed)

        # Group the table's rows by the games the decks need
        deck_games = {deck.source_game for deck in self._decks}
        for row in table.rows():
            if row.source_game in deck_games:
                rows_by_game[row.source_game].append(row)

        # Scale each game's vectors once; choose its negatives once
        for game in sorted(deck_games):
            rows = rows_by_game[game]
            vectors = table.vectors_for(rows).astype(np.float64)
            unit_vectors = scale_to_unit_length(vectors)
            result[game] = self._game_cards_of(game, rows, unit_vectors, rng)
        return result

    def _game_cards_of(
        self,
        game: GameId,
        rows: list[CardRow],
        unit_vectors: np.ndarray,
        rng: np.random.Generator,
    ) -> _GameCards:
        """Assemble one game's _GameCards, the placeholder
        (CardBinder.unknown_card_uuid(game)) left out of the negatives and
        the negatives capped per config.

        The only place a _GameCards is built.

        Inputs: game, rows (that game's table rows), unit_vectors (same
            order), rng (advanced only when capping).
        Output: _GameCards. Side effects: may advance rng. Exceptions: none.
        """
        uuids = tuple(row.nocab_uuid for row in rows)
        placeholder = CardBinder.unknown_card_uuid(game)
        negative_rows = np.array(
            [row for row, uuid in enumerate(uuids) if uuid != placeholder],
            dtype=np.int64,
        )

        # Cap the negatives with one seeded draw, shared by the game's decks
        cap = self._config.max_negatives_per_game
        if cap is not None and len(negative_rows) > cap:
            negative_rows = np.sort(rng.choice(negative_rows, size=cap, replace=False))
        return _GameCards(uuids, unit_vectors, negative_rows)

    def _score_deck(
        self, deck: EvaluationDeck, member_rows: np.ndarray, cards: _GameCards
    ) -> DeckScore | None:
        """Score one deck, or None when it cannot be measured (fewer than
        two members in the table, or no negative outside the deck).

        Inputs: deck; member_rows (its cards' rows in cards, deck order);
            cards (its game's).
        Output: DeckScore or None. Side effects: none. Exceptions: none.
        """
        # Negatives: the game's negative rows minus this deck's members
        negative_rows = np.setdiff1d(cards.negative_rows, member_rows)
        if len(member_rows) < 2 or len(negative_rows) == 0:
            return None

        # Score each member as an anchor against the game's cards; record
        # its tier
        anchor_cosines = cards.unit_vectors[member_rows] @ cards.unit_vectors.T
        aucs = anchor_aucs(anchor_cosines, member_rows, negative_rows)
        is_train = np.array(
            [
                self._is_train_anchor(cards.uuids[row], deck.source_game)
                for row in member_rows
            ]
        )
        return DeckScore(deck.deck_uuid, deck.source_game, aucs, is_train)

    def _is_train_anchor(self, card_uuid: UUID, game: GameId) -> bool:
        """Inputs: card_uuid, game. Output: holdout tier is TRAIN.
        Side effects: none. Exceptions: none."""
        return self._holdout.tier_of(card_uuid, game) == CardTier.TRAIN


def _member_rows(deck: EvaluationDeck, cards: _GameCards) -> np.ndarray:
    """deck's cards' rows in cards, in deck order, missing cards skipped.

    Inputs: deck, cards. Output: int array. Side effects: none.
    Exceptions: none.
    """
    rows = [cards.row_of[uuid] for uuid in deck.card_uuids if uuid in cards.row_of]
    return np.array(rows, dtype=np.int64)


def _count_scalars(
    scores: Sequence[DeckScore],
    missing_cards: int,
    decks: Sequence[EvaluationDeck],
) -> dict[str, float]:
    """{"n_decks", "n_anchors", "n_missing_cards", "n_dropped_decks",
    "n_dropped_games"}.

    Inputs: scores (the scored decks), missing_cards, decks (all given).
    Output: dict. Side effects: none. Exceptions: none.
    """
    deck_games = {deck.source_game for deck in decks}
    scored_games = {score.source_game for score in scores}
    return {
        "n_decks": float(len(scores)),
        "n_anchors": float(sum(score.n_cards for score in scores)),
        "n_missing_cards": float(missing_cards),
        "n_dropped_decks": float(len(decks) - len(scores)),
        "n_dropped_games": float(len(deck_games - scored_games)),
    }


def _write_per_deck_csv(path: Path, scores: Sequence[DeckScore]) -> Path:
    """One row per scored deck (_PER_DECK_COLUMNS); a tier AUC is blank when
    the deck has no anchor of that tier.

    Inputs: path (its directory exists), scores. Output: path.
    Side effects: writes path. Exceptions: OSError on a write failure.
    """
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_PER_DECK_COLUMNS)
        for score in scores:
            writer.writerow(
                [
                    score.deck_uuid,
                    score.source_game.value,
                    score.n_cards,
                    score.n_train_anchors,
                    score.n_heldout_anchors,
                    score.auc,
                    _blank_if_none(score.tier_auc(train=True)),
                    _blank_if_none(score.tier_auc(train=False)),
                ]
            )
    return path


def _blank_if_none(value: float | None) -> float | str:
    """Inputs: value. Output: value, or "" for None (a blank CSV cell).
    Side effects: none. Exceptions: none."""
    return "" if value is None else value
