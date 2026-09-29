"""CardEncoderModel shape handling for SingleCardModel and MultiCardModel.

The key regression: forward() always reads its input as a batch. So a
MultiCardModel given a batch of single cards embeds each card on its own,
and no card attends to an unrelated batch-mate.

A tiny deterministic fake text encoder stands in for a real LM. Each
text maps to a fixed vector, independent of the rest of the batch.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
import torch

from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.embedding_head import (
    AttentionPoolingEmbeddingHead,
    EmbeddingHead,
    LinearEmbeddingHead,
    ResidualMlpEmbeddingHead,
)
from src.encoder_model.multi_card_model import MultiCardModel
from src.encoder_model.single_card_model import SingleCardModel
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder

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
    return SingleCardModel(
        FakeTextEncoder(), LinearEmbeddingHead(HIDDEN, _EMBED)
    ).eval()


def _multi_model() -> MultiCardModel:
    torch.manual_seed(0)
    model = MultiCardModel(
        FakeTextEncoder(),
        LinearEmbeddingHead(HIDDEN, _EMBED),
        num_heads=2,
        num_layers=1,
    )
    return model.eval()  # no dropout, so outputs are deterministic


MODELS = [_single_model, _multi_model]
A, B, C, D = (_card(n) for n in ("Alpha", "Bravo", "Charlie", "Delta"))


def _shape_of(output: Any) -> Any:
    """Output nesting as lists of tensor shapes, for structural comparison."""
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    return [_shape_of(o) for o in output]


def _close(x: torch.Tensor, y: torch.Tensor) -> bool:
    return torch.allclose(x, y, atol=1e-5)


# --- explicit methods: output shape per input shape -------------------------


@pytest.mark.parametrize("make_model", MODELS)
def test_every_explicit_method_nests_outputs_like_its_input(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    e = (_EMBED,)
    with torch.no_grad():
        assert _shape_of(model.forward_single_card(A)) == e
        assert _shape_of(model.forward_multi_card([A, B, C])) == [e, e, e]
        assert _shape_of(model.forward_multi_group([[A, B], [C]])) == [[e, e], [e]]
        assert _shape_of(model.forward_batched_single_card([A, B])) == [e, e]
        assert _shape_of(model.forward_batched_multi_card([[A, B], [C]])) == [
            [e, e],
            [e],
        ]
        assert _shape_of(model.forward_batched_multi_group([[[A], [B, C]], [[D]]])) == [
            [[e], [e, e]],
            [[e]],
        ]


# --- canonical forward: always batched --------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
def test_forward_routes_each_depth_to_its_batched_method(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    cases = [
        ([A, B, C], model.forward_batched_single_card),
        ([[A, B], [C, D]], model.forward_batched_multi_card),
        ([[[A], [B]], [[C, D]]], model.forward_batched_multi_group),
    ]
    with torch.no_grad():
        for batch, explicit in cases:
            via_forward = model(batch)
            via_explicit = explicit(batch)  # type: ignore[operator]
            assert _shape_of(via_forward) == _shape_of(via_explicit)
            flat_forward = torch.stack(list(_flatten(via_forward)))
            flat_explicit = torch.stack(list(_flatten(via_explicit)))
            assert _close(flat_forward, flat_explicit)


def _flatten(output: Any) -> Any:
    stack = [output]
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            yield item
        else:
            stack.extend(reversed(item))


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


# --- the regression: batched single cards never see each other -------------


def test_multi_card_model_batched_single_cards_are_independent_of_batch_mates() -> None:
    model = _multi_model()
    with torch.no_grad():
        alone = model([A])[0]
        with_one = model([A, B])[0]
        with_many = model([C, A, D, B])[1]

    assert _close(alone, with_one)
    assert _close(alone, with_many)


def test_multi_card_model_batched_single_card_equals_forward_single_card() -> None:
    model = _multi_model()
    with torch.no_grad():
        batched = model([A, B, C])
        singles = [model.forward_single_card(card) for card in (A, B, C)]

    for got, expected in zip(batched, singles):
        assert _close(got, expected)  # type: ignore[arg-type]


def test_multi_card_model_contextualizes_within_a_multi_card_input() -> None:
    """The explicit unbatched method does attend across the list, so one
    card's embedding changes with its deck-mates."""
    model = _multi_model()
    with torch.no_grad():
        in_deck_ab = model.forward_multi_card([A, B])[0]
        in_deck_ac = model.forward_multi_card([A, C])[0]
        alone = model.forward_single_card(A)

    assert not _close(in_deck_ab, in_deck_ac)
    assert not _close(in_deck_ab, alone)


def test_multi_card_model_batched_decks_never_see_each_other() -> None:
    model = _multi_model()
    with torch.no_grad():
        deck_alone = model([[A, B]])[0]
        deck_with_other = model([[A, B], [C, D]])[0]

    for got, expected in zip(deck_with_other, deck_alone):
        assert _close(got, expected)


def test_multi_card_model_groups_within_one_example_never_see_each_other() -> None:
    model = _multi_model()
    with torch.no_grad():
        group_alone = model.forward_multi_group([[A, B]])[0]
        group_with_other = model.forward_multi_group([[A, B], [C]])[0]

    for got, expected in zip(group_with_other, group_alone):
        assert _close(got, expected)


def test_single_card_model_never_contextualizes() -> None:
    model = _single_model()
    with torch.no_grad():
        alone = model.forward_single_card(A)
        in_deck = model.forward_multi_card([B, A, C])[1]
        in_batch = model([D, A])[1]

    assert _close(alone, in_deck)
    assert _close(alone, in_batch)  # type: ignore[arg-type]


@pytest.mark.parametrize("make_model", MODELS)
def test_output_order_follows_input_order(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    with torch.no_grad():
        forward_order = model([A, B, C])
        reverse_order = model([C, B, A])

    for got, expected in zip(forward_order, reversed(reverse_order)):
        assert _close(got, expected)  # type: ignore[arg-type]


@pytest.mark.parametrize("make_model", MODELS)
def test_encoder_only_state_dict_is_every_weight(make_model: Any) -> None:
    model: CardEncoderModel = make_model()

    assert set(model.encoder_only_state_dict()) == set(model.state_dict())


def test_gradients_flow_through_batched_single_cards() -> None:
    model = _multi_model().train()
    loss = torch.stack(model([A, B])).sum()  # type: ignore[arg-type]

    loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.self_attention.parameters()
    )


# --- empty groups: real data (first-pick pools, no-blocker rows) ------------


@pytest.mark.parametrize("make_model", MODELS)
def test_empty_groups_map_to_empty_lists(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    e = (_EMBED,)
    with torch.no_grad():
        assert _shape_of(model([[[A], []]])) == [[[e], []]]
        assert _shape_of(model([[A], []])) == [[e], []]
        assert model.forward_multi_card([]) == []
        assert model.forward_multi_group([[], []]) == [[], []]
        assert model.forward_batched_single_card([]) == []


@pytest.mark.parametrize("make_model", MODELS)
def test_empty_group_does_not_change_its_siblings(make_model: Any) -> None:
    model: CardEncoderModel = make_model()
    with torch.no_grad():
        with_empty = model.forward_multi_group([[A, B], []])[0]
        without = model.forward_multi_group([[A, B]])[0]

    for got, expected in zip(with_empty, without):
        assert _close(got, expected)


# --- train mode --------------------------------------------------------------


@pytest.mark.parametrize("make_model", MODELS)
def test_train_mode_shapes_match_eval_mode(make_model: Any) -> None:
    model: CardEncoderModel = make_model().train()
    e = (_EMBED,)

    assert _shape_of(model([A, B])) == [e, e]
    assert _shape_of(model([[A, B], [C]])) == [[e, e], [e]]


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

    hook = model.embedding_head.register_forward_hook(keep_head_output)
    try:
        embeddings = model([A, B])
    finally:
        hook.remove()
    # Weighted, because a plain sum of a LayerNorm output is constant (zero grad)
    weights = torch.arange(1.0, _EMBED + 1)
    (embeddings[0] * weights).sum().backward()  # type: ignore[operator]

    (head_output,) = captured  # one batched head call for the whole batch
    assert head_output.grad is not None
    assert torch.all(head_output.grad[1] == 0)
    assert head_output.grad[0].abs().sum() > 0


def test_train_mode_multi_card_input_does_mix_cards() -> None:
    """Control for the test above: the same probe does see cross-card
    gradient when the cards form one group."""
    model = _multi_model().train()
    captured: list[torch.Tensor] = []

    def keep_head_output(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        output.retain_grad()
        captured.append(output)

    hook = model.embedding_head.register_forward_hook(keep_head_output)
    try:
        embeddings = model.forward_multi_card([A, B])
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
    group second. Use the explicit methods for other layouts."""
    model: CardEncoderModel = make_model()
    with pytest.raises(ValueError):
        model([[], [A]])
    with pytest.raises(ValueError):
        model([[[], [A]]])
    assert model.forward_batched_multi_card([[], [A]])[0] == []


# --- embedding_dim and isolated_embeddings ----------------------------------


@pytest.mark.parametrize(
    "head",
    [
        LinearEmbeddingHead(HIDDEN, 5),
        ResidualMlpEmbeddingHead(HIDDEN, 5, hidden_dim=4, num_blocks=1),
        AttentionPoolingEmbeddingHead(HIDDEN, 5),
    ],
)
def test_every_head_reports_the_width_it_produces(head: EmbeddingHead) -> None:
    encoding = FakeTextEncoder().encode(["Alpha", "Bravo"])
    assert head.output_dim == 5
    assert head(encoding).shape == (2, 5)


@pytest.mark.parametrize("make_model", MODELS)
def test_embedding_dim_is_the_real_output_width(make_model: Any) -> None:
    model = make_model()
    assert model.embedding_dim == _EMBED
    assert model.forward_single_card(A).shape == (model.embedding_dim,)


def test_multi_card_attention_is_sized_by_its_head() -> None:
    model = MultiCardModel(
        FakeTextEncoder(), LinearEmbeddingHead(HIDDEN, 6), num_heads=2, num_layers=1
    )
    assert model.embedding_dim == 6
    assert model.forward_multi_card([A, B])[0].shape == (6,)


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
    contextualized = model.forward_multi_card([A, B])
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
