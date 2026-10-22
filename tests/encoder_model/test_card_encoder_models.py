"""CardEncoderModel shape handling for SingleCardModel and MultiCardModel.

The key regression: forward() always reads its input as a batch. So a
MultiCardModel given a batch of single cards embeds each card on its own,
and no card attends to an unrelated batch-mate.

A tiny deterministic fake text encoder stands in for a real LM. Each
text maps to a fixed vector, independent of the rest of the batch.
"""

from datetime import datetime, timezone
from typing import Any, List, Sequence
from uuid import uuid4

import pytest
import torch

from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.embedding_head import (
    EmbeddingHead,
    LinearEmbeddingHead,
    ResidualMlpEmbeddingHead,
)
from src.encoder_model.multi_card_model import MultiCardModel
from src.encoder_model.single_card_model import SingleCardModel
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape
from tests.encoder_model.fakes import HIDDEN, fake_card_pipeline

_EMBED = 8


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={"name": name, "text": f"rules for {name}"},
        provenance=Provenance(
            DataSource.SCRYFALL, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def _single_model() -> SingleCardModel:
    torch.manual_seed(0)
    return SingleCardModel(fake_card_pipeline(_EMBED)).eval()


def _multi_model() -> MultiCardModel:
    torch.manual_seed(0)
    model = MultiCardModel(
        fake_card_pipeline(_EMBED),
        num_heads=2,
        num_layers=1,
    )
    return model.eval()  # no dropout, so outputs are deterministic


MODELS = [_single_model, _multi_model]
A, B, C, D = (_card(n) for n in ("Alpha", "Bravo", "Charlie", "Delta"))


def _close(x: torch.Tensor, y: torch.Tensor) -> bool:
    return torch.allclose(x, y, atol=1e-5)


def _padded(output: Any) -> PaddedEmbeddings:
    assert isinstance(output, PaddedEmbeddings)
    return output


def _deck(model: CardEncoderModel, cards: List[GenericCard]) -> List[torch.Tensor]:
    """One multi-card example's embeddings, through forward()."""
    return list(_padded(model([cards])).to_nested()[0])


def _alone(model: CardEncoderModel, card: GenericCard) -> torch.Tensor:
    """One card embedded on its own, through forward()."""
    return _padded(model([card])).cards[0, 0]


def _unbatched_reference(
    model: CardEncoderModel, cards: Sequence[GenericCard]
) -> torch.Tensor:
    """The pre-padding computation for one example: embed its cards, then
    (MultiCardModel) self-attention over exactly those cards, no padding."""
    base = model.card_pipeline.embed(list(cards))
    if isinstance(model, MultiCardModel):
        return model.self_attention(base.unsqueeze(0)).squeeze(0)
    return base


# --- forward: output type and layout per input shape -------------------------


@pytest.mark.parametrize("make_model", MODELS)
def test_a_batch_of_single_cards_is_padded_one_card_wide(make_model: Any) -> None:
    with torch.no_grad():
        output = _padded(make_model()([A, B, C]))
    assert output.shape is InputShape.SINGLE_CARD
    assert output.cards.shape == (3, 1, _EMBED)
    assert output.lengths == (1, 1, 1)
    assert bool(output.mask.all())


@pytest.mark.parametrize("make_model", MODELS)
def test_a_batch_of_decks_is_padded_to_the_longest(make_model: Any) -> None:
    with torch.no_grad():
        output = _padded(make_model()([[A, B, C], [D]]))
    assert output.shape is InputShape.MULTI_CARD
    assert output.cards.shape == (2, 3, _EMBED)
    assert output.lengths == (3, 1)
    assert output.mask.tolist() == [[True, True, True], [True, False, False]]
    assert torch.equal(output.cards[1, 1:], torch.zeros(2, _EMBED))


@pytest.mark.parametrize("make_model", MODELS)
def test_a_multi_group_batch_stays_nested(make_model: Any) -> None:
    with torch.no_grad():
        output = make_model()([[[A], [B, C]], [[D], []]])
    assert isinstance(output, list)
    assert [[len(group) for group in example] for example in output] == [[1, 2], [1, 0]]
    assert output[0][1][0].shape == (_EMBED,)


@pytest.mark.parametrize("make_model", MODELS)
def test_forward_rejects_a_bare_card(make_model: Any) -> None:
    with pytest.raises(TypeError):
        make_model()(A)


@pytest.mark.parametrize("make_model", MODELS)
def test_forward_rejects_an_empty_batch(make_model: Any) -> None:
    with pytest.raises(ValueError):
        make_model()([])


@pytest.mark.parametrize("make_model", MODELS)
def test_forward_rejects_nesting_deeper_than_batched_multi_group(
    make_model: Any,
) -> None:
    with pytest.raises(ValueError):
        make_model()([[[[A]]]])


# --- equivalence with the unbatched computation ------------------------------


@pytest.mark.parametrize("make_model", MODELS)
def test_padded_decks_equal_each_deck_embedded_alone(make_model: Any) -> None:
    """Mixed lengths, an empty example and a 600-card outlier in one batch:
    every real card equals its deck embedded alone, without padding."""
    model = make_model()
    outlier = [A, B, C, D] * 150
    decks = [[A, B, C], [], [D], outlier, [B, A]]
    with torch.no_grad():
        output = _padded(model(decks))
        for index, deck in enumerate(decks):
            got = output.cards[index, : len(deck)]
            if deck:
                assert _close(got, _unbatched_reference(model, deck))
            assert torch.equal(
                output.cards[index, len(deck) :],
                torch.zeros_like(output.cards[index, len(deck) :]),
            )
    assert output.lengths == (3, 0, 1, 600, 2)


@pytest.mark.parametrize("make_model", MODELS)
def test_multi_group_groups_equal_each_group_embedded_alone(make_model: Any) -> None:
    model = make_model()
    batch = [[[A, B], []], [[C], [D, A, B]]]
    with torch.no_grad():
        output = model(batch)
        for example, groups in zip(output, batch):
            for got, group in zip(example, groups):
                if group:
                    assert _close(torch.stack(got), _unbatched_reference(model, group))
                else:
                    assert got == []


# --- the regression: batched single cards never see each other -------------


def test_multi_card_model_batched_single_cards_are_independent_of_batch_mates() -> None:
    model = _multi_model()
    with torch.no_grad():
        alone = _alone(model, A)
        with_one = _padded(model([A, B])).cards[0, 0]
        with_many = _padded(model([C, A, D, B])).cards[1, 0]

    assert _close(alone, with_one)
    assert _close(alone, with_many)


def test_multi_card_model_single_card_is_attention_over_itself() -> None:
    model = _multi_model()
    with torch.no_grad():
        batched = _padded(model([A, B, C])).cards[:, 0]
        for row, card in zip(batched, (A, B, C)):
            assert _close(row, _unbatched_reference(model, [card])[0])


def test_multi_card_model_contextualizes_within_a_multi_card_input() -> None:
    """A multi-card example does attend across its cards, so one card's
    embedding changes with its deck-mates."""
    model = _multi_model()
    with torch.no_grad():
        in_deck_ab = _deck(model, [A, B])[0]
        in_deck_ac = _deck(model, [A, C])[0]
        alone = _alone(model, A)

    assert not _close(in_deck_ab, in_deck_ac)
    assert not _close(in_deck_ab, alone)


@pytest.mark.parametrize("make_model", MODELS)
def test_a_group_is_permutation_equivariant(make_model: Any) -> None:
    """Reordering a group's cards only reorders its embeddings. The odd-one-
    out and card-in-contexts contrastive styles put their intruder and
    anchor at a fixed position (see ContrastiveBatch); a positional
    encoding would leak that position, so this test must then fail."""
    model = make_model()
    with torch.no_grad():
        in_order = _deck(model, [A, B, C, D])
        reordered = _deck(model, [D, B, A, C])

    for original, moved in ((0, 2), (1, 1), (2, 3), (3, 0)):
        assert _close(in_order[original], reordered[moved])


def test_multi_card_model_batched_decks_never_see_each_other() -> None:
    model = _multi_model()
    with torch.no_grad():
        deck_alone = _deck(model, [A, B])
        deck_with_other = _padded(model([[A, B], [C, D, A]])).to_nested()[0]

    for got, expected in zip(deck_with_other, deck_alone):
        assert _close(got, expected)


def test_multi_card_model_groups_within_one_example_never_see_each_other() -> None:
    model = _multi_model()
    with torch.no_grad():
        group_alone = model([[[A, B]]])[0][0]
        group_with_other = model([[[A, B], [C]]])[0][0]

    for got, expected in zip(group_with_other, group_alone):
        assert _close(got, expected)


def test_single_card_model_never_contextualizes() -> None:
    model = _single_model()
    with torch.no_grad():
        alone = _alone(model, A)
        in_deck = _deck(model, [B, A, C])[1]
        in_batch = _padded(model([D, A])).cards[1, 0]

    assert _close(alone, in_deck)
    assert _close(alone, in_batch)


@pytest.mark.parametrize("make_model", MODELS)
def test_output_order_follows_input_order(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    with torch.no_grad():
        forward_order = _padded(model([A, B, C])).cards[:, 0]
        reverse_order = _padded(model([C, B, A])).cards[:, 0]

    assert _close(forward_order, reverse_order.flip(0))


@pytest.mark.parametrize("make_model", MODELS)
def test_encoder_only_state_dict_is_every_weight(make_model: Any) -> None:
    model: CardEncoderModel = make_model()

    assert set(model.encoder_only_state_dict()) == set(model.state_dict())


def test_gradients_flow_through_batched_single_cards() -> None:
    model = _multi_model().train()
    embeddings = _padded(model([A, B])).cards[:, 0]
    # A plain sum is constant: normalized embeddings sum to 0
    loss = (embeddings * torch.arange(embeddings.shape[-1])).sum()

    loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.self_attention.parameters()
    )


def test_padded_decks_give_finite_gradients_with_an_empty_example() -> None:
    model = _multi_model().train()
    output = _padded(model([[A, B, C], [], [D]]))
    weights = torch.arange(1.0, _EMBED + 1)
    (output.cards * weights).sum().backward()

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())


# --- empty groups: real data (first-pick pools, no-blocker rows) ------------


@pytest.mark.parametrize("make_model", MODELS)
def test_empty_groups_and_examples_are_empty(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    with torch.no_grad():
        assert model([[[A], []]])[0][1] == []
        output = _padded(model([[A], []]))
    assert output.lengths == (1, 0)
    assert torch.equal(output.cards[1], torch.zeros(1, _EMBED))


@pytest.mark.parametrize("make_model", MODELS)
def test_empty_group_does_not_change_its_siblings(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    with torch.no_grad():
        with_empty = model([[[A, B], []]])[0][0]
        without = model([[[A, B]]])[0][0]

    for got, expected in zip(with_empty, without):
        assert _close(got, expected)


# --- train mode --------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
def test_train_mode_shapes_match_eval_mode(make_model: Any) -> None:
    model: CardEncoderModel = make_model().train()

    assert _padded(model([A, B])).cards.shape == (2, 1, _EMBED)
    assert _padded(model([[A, B], [C]])).cards.shape == (2, 2, _EMBED)


def test_train_mode_batched_single_cards_get_no_gradient_from_batch_mates() -> None:
    """Independence in train mode, through the real forward(). Card A's
    embedding has zero gradient with respect to card B's head output, the
    input to self-attention. So attention never mixes batch-mates, and
    neither does dropout."""
    model = _multi_model().train()
    captured: list[torch.Tensor] = []

    def keep_head_output(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        output.retain_grad()
        captured.append(output)

    hook = model.card_pipeline.embedding_head.register_forward_hook(keep_head_output)
    try:
        embeddings = _padded(model([A, B])).cards[:, 0]
    finally:
        hook.remove()
    # Weighted, because a plain sum of a LayerNorm output is constant (zero grad)
    weights = torch.arange(1.0, _EMBED + 1)
    (embeddings[0] * weights).sum().backward()

    (head_output,) = captured  # one batched head call for the whole batch
    assert head_output.grad is not None
    assert torch.all(head_output.grad[1] == 0)
    assert head_output.grad[0].abs().sum() > 0


def test_train_mode_padded_decks_get_no_gradient_from_other_decks() -> None:
    """The same probe across padded decks: deck 0's cards see no gradient
    from deck 1's head output, so padding and batching mix nothing."""
    model = _multi_model().train()
    captured: list[torch.Tensor] = []

    def keep_head_output(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        output.retain_grad()
        captured.append(output)

    hook = model.card_pipeline.embedding_head.register_forward_hook(keep_head_output)
    try:
        output = _padded(model([[A, B], [C, D, A]]))
    finally:
        hook.remove()
    weights = torch.arange(1.0, _EMBED + 1)
    (output.cards[0] * weights).sum().backward()

    (head_output,) = captured  # rows: deck 0's two cards, then deck 1's three
    assert head_output.grad is not None
    assert torch.all(head_output.grad[2:] == 0)
    assert head_output.grad[1].abs().sum() > 0


def test_train_mode_multi_card_input_does_mix_cards() -> None:
    """Control for the tests above: the same probe does see cross-card
    gradient when the cards form one group."""
    model = _multi_model().train()
    captured: list[torch.Tensor] = []

    def keep_head_output(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        output.retain_grad()
        captured.append(output)

    hook = model.card_pipeline.embedding_head.register_forward_hook(keep_head_output)
    try:
        embeddings = _deck(model, [A, B])
    finally:
        hook.remove()
    weights = torch.arange(1.0, _EMBED + 1)
    (embeddings[0] * weights).sum().backward()

    (head_output,) = captured
    assert head_output.grad is not None
    assert head_output.grad[1].abs().sum() > 0


@pytest.mark.parametrize("make_model", MODELS)
def test_forward_rejects_an_empty_first_group_or_example(make_model: Any) -> None:
    """Known limit, pinned deliberately: the shape probe (like Batch's) peeks
    at the first entry at each level. So an empty FIRST group or example
    can't be classified. Today's data constructors put the possibly-empty
    group second."""
    model: CardEncoderModel = make_model()
    with pytest.raises(ValueError):
        model([[], [A]])
    with pytest.raises(ValueError):
        model([[[], [A]]])


# --- embedding_dim and isolated_embeddings ----------------------------------


_HEADS = [
    LinearEmbeddingHead(HIDDEN, 5),
    ResidualMlpEmbeddingHead(HIDDEN, 5, hidden_dim=4, num_blocks=1),
]


@pytest.mark.parametrize("head", _HEADS)
def test_every_head_reports_the_widths_it_reads_and_produces(
    head: EmbeddingHead,
) -> None:
    assert head.input_dim == HIDDEN
    assert head.output_dim == 5
    assert head(torch.randn(2, HIDDEN)).shape == (2, 5)


@pytest.mark.parametrize("head", _HEADS)
def test_every_head_outputs_a_fixed_scale_embedding(head: EmbeddingHead) -> None:
    embeddings = head(torch.randn(2, HIDDEN))
    # Mean 0, variance 1 across the coordinates: L2 norm sqrt(width)
    assert torch.allclose(embeddings.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(embeddings.norm(dim=1), torch.full((2,), 5**0.5), atol=1e-3)


@pytest.mark.parametrize("make_model", MODELS)
def test_every_model_outputs_a_fixed_scale_embedding(make_model: Any) -> None:
    model = make_model()
    for embedding in _deck(model, [A, B, C]):
        assert embedding.norm().item() == pytest.approx(_EMBED**0.5, abs=1e-3)


@pytest.mark.parametrize("make_model", MODELS)
def test_embedding_dim_is_the_real_output_width(make_model: Any) -> None:
    model = make_model()
    assert model.embedding_dim == _EMBED
    assert _alone(model, A).shape == (model.embedding_dim,)


def test_multi_card_attention_is_sized_by_its_card_pipeline() -> None:
    model = MultiCardModel(fake_card_pipeline(6), num_heads=2, num_layers=1)
    assert model.embedding_dim == 6
    assert _deck(model, [A, B])[0].shape == (6,)


def test_multi_card_attention_layers_take_their_settings() -> None:
    model = MultiCardModel(
        fake_card_pipeline(6),
        num_heads=2,
        num_layers=3,
        ffn_dim=10,
        dropout=0.0,
        norm_first=True,
    )
    layers = model.self_attention.layers
    assert len(layers) == 3
    assert layers[0].linear1.out_features == 10
    assert layers[0].norm_first
    assert layers[0].dropout.p == 0.0
    assert _deck(model, [A, B])[0].shape == (6,)


@pytest.mark.parametrize("make_model", MODELS)
def test_isolated_embeddings_is_a_float32_cpu_table(make_model: Any) -> None:
    model = make_model()
    table = model.isolated_embeddings([A, B, C])
    assert table.shape == (3, _EMBED)
    assert table.dtype == torch.float32
    assert table.device.type == "cpu"
    assert not table.requires_grad


@pytest.mark.parametrize("make_model", MODELS)
def test_isolated_embeddings_of_no_cards_is_an_empty_table(make_model: Any) -> None:
    table = make_model().isolated_embeddings([])
    assert table.shape == (0, _EMBED)
    assert table.dtype == torch.float32


@pytest.mark.parametrize("make_model", MODELS)
def test_each_row_depends_only_on_its_own_card(make_model: Any) -> None:
    model = make_model()
    together = model.isolated_embeddings([A, B, C])
    for index, card in enumerate([A, B, C]):
        assert _close(together[index], model.isolated_embeddings([card])[0])


def test_multi_card_isolated_rows_differ_from_contextualized_ones() -> None:
    # Guards the choice of forward path: attention across A and B would
    # change A's embedding
    model = _multi_model()
    isolated = model.isolated_embeddings([A, B])
    with torch.no_grad():
        contextualized = _deck(model, [A, B])
    assert not _close(isolated[0], contextualized[0])


@pytest.mark.parametrize("make_model", MODELS)
def test_isolated_embeddings_are_deterministic_and_restore_train_mode(
    make_model: Any,
) -> None:
    model = make_model().train(True)
    first = model.isolated_embeddings([A, B])
    second = model.isolated_embeddings([A, B])
    assert torch.equal(first, second)  # eval mode inside: no dropout
    assert model.training is True


@pytest.mark.parametrize("make_model", MODELS)
def test_isolated_embeddings_at_bf16_come_back_float32(make_model: Any) -> None:
    model = make_model()
    table = model.isolated_embeddings([A, B], precision="bf16")
    assert table.dtype == torch.float32
    assert torch.allclose(table, model.isolated_embeddings([A, B]), atol=0.1)
