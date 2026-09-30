from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping
from uuid import UUID, uuid4

import pandas as pd
import torch
from torch import nn

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.dojo import BatchBudget, DojoBatch
from src.dojos.mods.mod import ModTally
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import TrainingDatum
from src.training.preflight import _random_embeddings_like, preflight_dojo

_BUDGET = BatchBudget(max_cost=8, cost_of=lambda card: 1)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(DataSource.SCRYFALL, name, datetime.now(timezone.utc)),
    )


@dataclass
class _CardBatch:
    """A batch shaped like a real Dojo's: inputs is a List[GenericCard],
    so preflight's embeddings mirror that with a List[Tensor]."""

    inputs: Any

    def __len__(self) -> int:
        return len(self.inputs)


class _ListCardFakeDojo:
    """A minimal Dojo whose batch.inputs is GenericCard-shaped (unlike
    tests/training/fakes.py's FakeDojo, whose inputs are a bare tensor
    standing in for an already-encoded batch) - what preflight_dojo
    actually runs against.

    fail: 'train' raises when TRAIN batches are requested. head_size
    fixes the decoder head's expected embedding width, independent of
    whatever card_embedding_size a test passes to preflight_dojo.
    """

    def __init__(
        self,
        name: str,
        train_count: int = 10,
        n_batches: int = 3,
        fail: str | None = None,
        nan_loss: bool = False,
        count_fails: bool = False,
        head_size: int = 2,
    ) -> None:
        self.name = name
        self.holdout = HoldoutSpec.no_holdout()
        self._train_count = train_count
        self._n_batches = n_batches
        self._fail = fail
        self._nan_loss = nan_loss
        self._count_fails = count_fails
        self._head = nn.Linear(head_size, 1)

    def batches(
        self, split: Split, budget: BatchBudget, max_examples: int | None = None
    ) -> Iterator[DojoBatch]:
        if self._fail == split.value:
            raise RuntimeError(f"{self.name} cannot serve {split}")
        for _ in range(self._n_batches):
            yield _CardBatch([_card("A"), _card("B")])

    def example_count(self, split: Split) -> int:
        if self._count_fails:
            raise OSError("cannot read split file")
        return self._train_count

    def compute_loss(self, embeddings: Any, batch: Any) -> torch.Tensor:
        loss = torch.stack([self._head(embedding) for embedding in embeddings]).mean()
        return loss * float("nan") if self._nan_loss else loss

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self._head.parameters()

    def move_head_to(self, device: torch.device) -> None:
        self._head.to(device)

    def mod_tallies(self) -> Mapping[str, ModTally]:
        return {}

    def reset_head(self) -> None:
        pass


class TestPreflightAgainstAFakeDojo:
    def test_a_healthy_dojo_reports_counts_and_a_finite_loss(self) -> None:
        dojo = _ListCardFakeDojo("healthy", train_count=50, head_size=2)
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=2)
        assert result.ok
        assert result.dojo_name == "healthy"
        assert result.train_count == 50
        assert result.test_count == 50
        assert result.sample_loss is not None
        assert result.error is None

    def test_zero_train_examples_is_reported_not_raised(self) -> None:
        dojo = _ListCardFakeDojo("empty", train_count=0)
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=2)
        assert not result.ok
        assert result.sample_loss is None
        assert "0 TRAIN examples" in (result.error or "")

    def test_example_count_raising_is_caught(self) -> None:
        dojo = _ListCardFakeDojo("uncountable", count_fails=True)
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=2)
        assert not result.ok
        assert result.train_count == -1
        assert "OSError" in (result.error or "")

    def test_batches_raising_is_caught(self) -> None:
        dojo = _ListCardFakeDojo("unreadable", train_count=10, fail="train")
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=2)
        assert not result.ok
        # example_count succeeded before batches() failed
        assert result.train_count == 10
        assert "RuntimeError" in (result.error or "")

    def test_a_dojo_with_no_train_batches_is_reported_not_raised(self) -> None:
        dojo = _ListCardFakeDojo("miscounted", train_count=10, n_batches=0)
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=2)
        assert not result.ok
        assert "yielded none" in (result.error or "")

    def test_nan_loss_is_reported_not_raised(self) -> None:
        dojo = _ListCardFakeDojo("nan", train_count=10, nan_loss=True, head_size=2)
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=2)
        assert not result.ok
        assert "non-finite" in (result.error or "")

    def test_a_head_expecting_the_wrong_embedding_size_is_reported_not_raised(
        self,
    ) -> None:
        # The head is nn.Linear(2, 1); asking preflight for size-8
        # embeddings mismatches it, the same way a real dojo built for
        # one card_embedding_size would fail against another.
        dojo = _ListCardFakeDojo("mismatched", train_count=10, head_size=2)
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=8)
        assert not result.ok
        assert result.sample_loss is None


class TestRandomEmbeddingsMirrorInputNesting:
    def test_a_single_card_becomes_one_tensor(self) -> None:
        embeddings = _random_embeddings_like(_card("A"), card_embedding_size=5)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (5,)

    def test_a_list_of_cards_becomes_a_list_of_tensors(self) -> None:
        cards = [_card("A"), _card("B"), _card("C")]
        embeddings = _random_embeddings_like(cards, card_embedding_size=3)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(tensor.shape == (3,) for tensor in embeddings)

    def test_a_list_of_decks_becomes_a_matching_nested_list(self) -> None:
        decks = [[_card("A"), _card("B")], [_card("C")]]
        embeddings = _random_embeddings_like(decks, card_embedding_size=4)
        assert [len(group) for group in embeddings] == [2, 1]
        assert embeddings[0][0].shape == (4,)
        assert embeddings[1][0].shape == (4,)


class _CardPerRowConstructor:
    """Row i -> the card whose uuid the row names, label = i."""

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        data: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            card = lookup.get_by_uuid(UUID(row["nocab_uuid"]))
            if card is not None:
                data.append((card, float(row["label"])))
        return data


class TestPreflightAgainstARealDojo:
    """Confirms preflight_dojo runs GenericDojo's real batches()/compute_loss
    path, not just FakeDojo's simplified stand-in."""

    def _dojo(
        self, tmp_path: Path, card_embedding_size: int = 4
    ) -> SingleCardRegressionDojo:
        cards = [_card("A"), _card("B")]
        binder = CardBinder()
        for card in cards:
            binder.create(card)
        source = tmp_path / "source.parquet"
        pd.DataFrame(
            {
                "nocab_uuid": [str(cards[i % 2].nocab_uuid) for i in range(20)],
                "label": [float(i) for i in range(20)],
            }
        ).to_parquet(source, index=False)
        return SingleCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_CardPerRowConstructor(),
            card_lookup=binder,
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=0,
                strict_version_check=False,
                output_directory=tmp_path / "splits",
            ),
        )

    def test_a_real_dojo_built_for_the_right_size_passes(self, tmp_path: Path) -> None:
        dojo = self._dojo(tmp_path, card_embedding_size=4)
        result = preflight_dojo(dojo, _BUDGET, 4)
        assert result.ok, result.error
        assert result.train_count > 0
        assert result.sample_loss is not None

    def test_a_real_dojo_checked_against_the_wrong_size_fails(
        self, tmp_path: Path
    ) -> None:
        dojo = self._dojo(tmp_path, card_embedding_size=4)
        result = preflight_dojo(dojo, _BUDGET, 6)
        assert not result.ok
        assert result.sample_loss is None


class _TalliedDojo(_ListCardFakeDojo):
    """A fake dojo with one live mod tally."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self.tally = ModTally(cards_seen=4, cards_changed=3)

    def mod_tallies(self) -> Mapping[str, ModTally]:
        return {"0:FakeMod": self.tally}


class TestModTallies:
    def test_a_snapshot_is_reported(self) -> None:
        dojo = _TalliedDojo("tallied")
        result = preflight_dojo(dojo, _BUDGET, card_embedding_size=2)

        assert result.mod_tallies["0:FakeMod"] == dojo.tally
        assert result.mod_tallies["0:FakeMod"] is not dojo.tally
        dojo.tally.cards_changed = 0  # later training must not rewrite the report
        assert result.mod_tallies["0:FakeMod"].cards_changed == 3

    def test_tallies_are_kept_on_failure(self) -> None:
        result = preflight_dojo(_TalliedDojo("broken", fail="train"), _BUDGET, 2)
        assert not result.ok
        assert "0:FakeMod" in result.mod_tallies
