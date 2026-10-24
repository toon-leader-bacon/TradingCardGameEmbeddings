import math
import dataclasses
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.styles.missing_card import MissingCardInfoNCELoss
from src.dojos.contrastive.dojo import ContrastiveDojo
from src.dojos.dojo import BatchBudget, Dojo
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.dojos.mods.card_field_mods import RandomKeyMaskMod
from src.dojos.mods.mod import MASK_TOKEN
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec, UnknownDeckSplitError
from src.schema.splits import Split
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.dojos.padding import padded


def _card(name: str = "card") -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _deck() -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="deck",
        card_nocab_uuids=[uuid4()],
        provenance=None,
    )


class _ScriptedPairConstructor:
    """Test double: yields the given ContrastiveBatches in order,
    ignoring the actual decks/card_lookup it's called with."""

    cards_per_deck = 2

    def __init__(self, batches: list[ContrastiveBatch]) -> None:
        self._batches = iter(batches)

    def build(self, decks: list[GenericDeck], card_lookup: object) -> ContrastiveBatch:
        return next(self._batches)


class _DeckCountingPairConstructor:
    """Test double: records each deck sample's size and builds a small
    usable batch (two decks of one card each) whatever it is given."""

    cards_per_deck = 2

    def __init__(self) -> None:
        self.deck_counts: list[int] = []

    def build(self, decks: list[GenericDeck], card_lookup: object) -> ContrastiveBatch:
        self.deck_counts.append(len(decks))
        return ContrastiveBatch(
            inputs=[_card() for _ in range(4)],
            identities=[(uuid4(),) for _ in range(4)],
            positive_cliques=[[0, 1], [2, 3]],
        )


def _counting_dojo(
    constructor: _DeckCountingPairConstructor, deck_count: int
) -> ContrastiveDojo:
    return ContrastiveDojo(
        dealer=_dealer_with_decks(deck_count),
        pair_constructor=constructor,
        card_lookup=CardBinder(),
        holdout=_deck_holdout(),
        strict_version_check=False,
    )


class _RecordingLoss:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def calculate(self, item_embeddings, identities, positive_cliques) -> torch.Tensor:
        self.calls.append((item_embeddings, identities, positive_cliques))
        return torch.tensor(0.0)


def _deck_holdout(
    deck_split_ratios: tuple[float, float, float] = (1, 0, 0),
) -> HoldoutSpec:
    """Every card visible; decks split by deck_split_ratios."""
    return HoldoutSpec(
        seed=1,
        tier_ratios=(1, 0, 0),
        held_out_games=frozenset(),
        deck_split_ratios=deck_split_ratios,
    )


def _dealer_with_decks(
    count: int, deck_split_ratios: tuple[float, float, float] = (1, 0, 0)
) -> DeckBoxDealer:
    box = DeckBox()
    for _ in range(count):
        box.create(_deck())
    # A throwaway index file - these tests don't care about a dealer's
    # persisted index location, only its dealt-decks behavior, so a
    # fresh tempdir per call avoids threading a tmp_path fixture
    # through every one of this module's test methods.
    index_path = Path(tempfile.mkdtemp()) / "dealer.db"
    return DeckBoxDealer(box, GameId.MTG, index_path, _deck_holdout(deck_split_ratios))


# Two decks per batch (the test pair constructor costs 2 cards a deck)
_BUDGET = BatchBudget(max_cost=4, cost_of=lambda card: 1)


class TestBatches:
    def test_skips_a_degenerate_batch_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        good_batch = ContrastiveBatch(
            inputs=[_card(name) for name in "abcd"],
            identities=[(uuid4(),) for _ in range(4)],
            positive_cliques=[[0, 1], [2, 3]],
        )
        # No positive pair at all
        no_positive_batch = ContrastiveBatch(
            inputs=[_card("e")], identities=[(uuid4(),)], positive_cliques=[[0]]
        )
        # One surviving deck: a positive but no negative
        no_negative_batch = ContrastiveBatch(
            inputs=[_card("f"), _card("g")],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )
        pair_constructor = _ScriptedPairConstructor(
            [good_batch, no_positive_batch, no_negative_batch, good_batch]
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(8),
            pair_constructor=pair_constructor,
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )

        with caplog.at_level(logging.WARNING):
            batches = list(dojo.batches(Split.TRAIN, _BUDGET))

        assert batches == [good_batch, good_batch]
        assert "degenerate" in caplog.text


class TestMalformedBatch:
    def test_a_malformed_batch_raises_instead_of_being_skipped(self) -> None:
        # A pair-constructor bug (a clique that isn't a [context, card]
        # pair) must surface, not pass as a degenerate batch
        malformed = ContrastiveBatch(
            inputs=[[_card("a"), _card("b")], [_card("c")], [_card("d")]],
            identities=[(uuid4(), uuid4()), (uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1, 2]],
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(4),
            pair_constructor=_ScriptedPairConstructor([malformed]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
            contrastive_loss=MissingCardInfoNCELoss(),
        )

        with pytest.raises(ValueError, match="pair"):
            list(dojo.batches(Split.TRAIN, _BUDGET))


class TestComputeLoss:
    def test_raises_on_length_mismatch(self) -> None:
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(1),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )
        batch = ContrastiveBatch(
            inputs=[_card(), _card()],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )

        with pytest.raises(ValueError):
            dojo.compute_loss(padded([torch.randn(4)]), batch)

    def test_delegates_to_the_injected_loss(self) -> None:
        recording_loss = _RecordingLoss()
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(1),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
            contrastive_loss=recording_loss,
        )
        batch = ContrastiveBatch(
            inputs=[_card(), _card()],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )
        embeddings = padded([torch.randn(4), torch.randn(4)])

        dojo.compute_loss(embeddings, batch)

        called_embeddings, called_identities, called_positive_cliques = (
            recording_loss.calls[0]
        )
        assert called_embeddings is embeddings
        assert called_identities == batch.identities
        assert called_positive_cliques == batch.positive_cliques


class TestBudget:
    def test_raises_when_the_budget_cannot_fit_two_decks(self) -> None:
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )

        with pytest.raises(ValueError, match="fits only 0 deck"):
            list(dojo.batches(Split.TRAIN, BatchBudget(1, lambda card: 1)))

    def test_raises_when_a_built_batch_exceeds_the_budget(self) -> None:
        cards = [_card(), _card(), _card()]
        heavy = ContrastiveBatch(
            inputs=cards,  # type: ignore[arg-type]
            identities=[(c.nocab_uuid,) for c in cards],
            positive_cliques=[[0, 1], [2]],  # card 2 is 0 and 1's negative
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([heavy]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )

        # Two decks fit by count (4 // 2), but these 3 cards cost 2 each
        with pytest.raises(ValueError, match="costs 6, over budget 4"):
            list(dojo.batches(Split.TRAIN, BatchBudget(4, lambda card: 2)))

    def test_max_examples_stops_after_that_many_decks(self) -> None:
        good = ContrastiveBatch(
            inputs=[_card() for _ in range(4)],
            identities=[(uuid4(),) for _ in range(4)],
            positive_cliques=[[0, 1], [2, 3]],
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([good, good, good]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )

        batches = list(dojo.batches(Split.TRAIN, _BUDGET, max_examples=4))

        assert len(batches) == 2

    def test_max_examples_trims_the_last_batch_instead_of_dropping_it(
        self,
    ) -> None:
        # 3 decks per batch; the first 5 decks are 3 + 2
        constructor = _DeckCountingPairConstructor()
        dojo = _counting_dojo(constructor, deck_count=9)

        list(dojo.batches(Split.TRAIN, BatchBudget(6, lambda card: 1), 5))

        assert constructor.deck_counts == [3, 2]

    def test_a_budget_larger_than_the_split_yields_one_batch_of_it(self) -> None:
        # e.g. a small deck box under a large budget, or a capped TEST pass
        constructor = _DeckCountingPairConstructor()
        dojo = _counting_dojo(constructor, deck_count=5)

        list(dojo.batches(Split.TRAIN, BatchBudget(200, lambda card: 1), 64))

        assert constructor.deck_counts == [5]

    def test_the_budget_alone_sets_the_decks_per_batch(self) -> None:
        constructor = _DeckCountingPairConstructor()
        dojo = _counting_dojo(constructor, deck_count=50)

        list(dojo.batches(Split.TRAIN, BatchBudget(80, lambda card: 1)))

        assert constructor.deck_counts == [40, 10]

    def test_a_lone_trailing_deck_is_dropped(self) -> None:
        constructor = _DeckCountingPairConstructor()
        dojo = _counting_dojo(constructor, deck_count=5)

        list(dojo.batches(Split.TRAIN, _BUDGET))

        assert constructor.deck_counts == [2, 2]

    def test_satisfies_the_dojo_protocol_and_counts_decks(self) -> None:
        dojo: Dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )

        assert dojo.example_count(Split.TRAIN) == 6
        assert list(dojo.trainable_parameters()) == []


class TestVersionCheck:
    def test_raises_when_dealer_has_no_recorded_card_binder_version(self) -> None:
        with pytest.raises(ValueError, match="CardBinder"):
            ContrastiveDojo(
                dealer=_dealer_with_decks(2),
                pair_constructor=_ScriptedPairConstructor([]),
                card_lookup=CardBinder(),
                holdout=_deck_holdout(),
            )

    def test_passes_when_dealers_recorded_version_matches(self, tmp_path: Path) -> None:
        binder = CardBinder()
        box = DeckBox()
        box.create(_deck())
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, binder.version_for(GameId.MTG))
        dealer = DeckBoxDealer(
            DeckBox.load([path]),
            GameId.MTG,
            tmp_path / "dealer.db",
            _deck_holdout(),
        )

        dojo = ContrastiveDojo(
            dealer=dealer,
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=binder,
            holdout=_deck_holdout(),
        )

        assert dojo.example_count(Split.TRAIN) == 1

    def test_raises_when_dealers_recorded_version_is_stale(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        box = DeckBox()
        box.create(_deck())
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, "stale-version")
        dealer = DeckBoxDealer(
            DeckBox.load([path]),
            GameId.MTG,
            tmp_path / "dealer.db",
            _deck_holdout(),
        )

        with pytest.raises(ValueError, match="CardBinder"):
            ContrastiveDojo(
                dealer=dealer,
                pair_constructor=_ScriptedPairConstructor([]),
                card_lookup=binder,
                holdout=_deck_holdout(),
            )

    def test_a_dealer_under_another_deck_split_is_refused(self) -> None:
        with pytest.raises(ValueError, match="splits decks by"):
            ContrastiveDojo(
                dealer=_dealer_with_decks(2, deck_split_ratios=(1, 0, 0)),
                pair_constructor=_ScriptedPairConstructor([]),
                card_lookup=CardBinder(),
                holdout=_deck_holdout((0, 1, 0)),
                strict_version_check=False,
            )

    def test_a_holdout_with_no_deck_split_is_refused(self) -> None:
        unknown = HoldoutSpec(1, (1, 0, 0), frozenset(), deck_split_ratios=None)
        with pytest.raises(UnknownDeckSplitError):
            ContrastiveDojo(
                dealer=_dealer_with_decks(2),
                pair_constructor=_ScriptedPairConstructor([]),
                card_lookup=CardBinder(),
                holdout=unknown,
                strict_version_check=False,
            )

    def test_strict_version_check_false_skips_a_real_mismatch(self) -> None:
        dojo: Dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(2),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )

        assert dojo.example_count(Split.TRAIN) == 2


class TestMods:
    def _batch(self) -> ContrastiveBatch:
        cards = [
            dataclasses.replace(
                _card(f"c{i}"), raw_content={"name": f"c{i}", "cost": i}
            )
            for i in range(4)
        ]
        return ContrastiveBatch(
            inputs=cards,
            identities=[(card.nocab_uuid,) for card in cards],
            positive_cliques=[[0, 1], [2, 3]],
        )

    def _dojo(self, batch: ContrastiveBatch, mods: list) -> ContrastiveDojo:
        return ContrastiveDojo(
            # Hash-split, so enough decks that both splits fill two samples
            dealer=_dealer_with_decks(40, deck_split_ratios=(1, 1, 0)),
            pair_constructor=_ScriptedPairConstructor([batch] * 3),
            card_lookup=CardBinder(),
            holdout=_deck_holdout((1, 1, 0)),
            strict_version_check=False,
            mod_pipeline=ModPipeline(mods),
        )

    def test_train_batches_are_modded_with_structure_kept(self) -> None:
        original = self._batch()
        dojo = self._dojo(original, [RandomKeyMaskMod(rng_seed=0)])

        modded = next(dojo.batches(Split.TRAIN, _BUDGET))

        assert modded.identities == original.identities
        assert modded.positive_cliques == original.positive_cliques
        assert all(MASK_TOKEN in card.raw_content.values() for card in modded.inputs)
        assert all(MASK_TOKEN not in c.raw_content.values() for c in original.inputs)
        assert dojo.mod_tallies()["0:RandomKeyMaskMod"].cards_changed == 4

    def test_train_only_mods_skip_test_batches(self) -> None:
        original = self._batch()
        dojo = self._dojo(original, [RandomKeyMaskMod(rng_seed=0)])
        batch = next(dojo.batches(Split.TEST, _BUDGET))
        assert batch.inputs == original.inputs

    def test_no_mods_yields_the_batch_itself(self) -> None:
        original = self._batch()
        dojo = self._dojo(original, [])
        assert next(dojo.batches(Split.TRAIN, _BUDGET)) is original
        assert dojo.mod_tallies() == {}


class TestBaselineLoss:
    def test_is_the_losss_constant_logit_loss_for_the_batch(self) -> None:
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(1),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )
        batch = ContrastiveBatch(
            inputs=[_card() for _ in range(6)],
            identities=[(uuid4(),) for _ in range(6)],
            positive_cliques=[[0, 1], [2, 3], [4, 5]],
        )
        assert dojo.baseline_loss(batch) == pytest.approx(math.log(5))

    def test_rejects_a_foreign_batch(self) -> None:
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(1),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=_deck_holdout(),
            strict_version_check=False,
        )
        with pytest.raises(TypeError):
            dojo.baseline_loss(object())  # type: ignore[arg-type]
