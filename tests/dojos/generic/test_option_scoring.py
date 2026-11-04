import torch

from src.dojos.generic.option_scoring import BilinearOptionScoringHead


def _options() -> tuple[torch.Tensor, torch.Tensor]:
    """Two examples: 3 options, then 2 options and one padding slot."""
    options = torch.randn(2, 3, 4)
    options[1, 2] = 0.0
    mask = torch.tensor([[True, True, True], [True, True, False]])
    return options, mask


class TestScore:
    def test_returns_one_logit_per_option_and_minus_inf_at_padding(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options, mask = _options()

        result = head.score(options, mask, context=None, context_mask=None)

        assert result.shape == (2, 3)
        assert bool(torch.isfinite(result[mask]).all())
        assert result[1, 2].item() == float("-inf")

    def test_matches_scoring_each_example_alone(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options, mask = _options()
        context = torch.randn(2, 4)

        batched = head.score(options, mask, context, context_mask=None)
        alone = head.score(options[1:, :2], mask[1:, :2], context[1:], None)

        assert torch.allclose(batched[1, :2], alone[0])

    def test_context_changes_the_score(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options, mask = _options()

        without_context = head.score(options, mask, None, None)
        with_context = head.score(options, mask, torch.randn(2, 4), None)

        assert not torch.allclose(without_context[mask], with_context[mask])

    def test_a_false_context_row_scores_as_if_unconditioned(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options, mask = _options()
        context = torch.randn(2, 4)
        context[1] = float("nan")  # garbage in the unreal row must not leak
        context_mask = torch.tensor([True, False])

        mixed = head.score(options, mask, context, context_mask)
        unconditioned = head.score(options, mask, None, None)

        assert torch.allclose(mixed[1], unconditioned[1])
        assert not torch.allclose(mixed[0], unconditioned[0])

    def test_gradients_flow_to_both_terms_when_context_given(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options, mask = _options()

        result = head.score(options, mask, torch.randn(2, 4), None)
        result[mask].sum().backward()

        assert head.option_term.weight.grad is not None
        assert head.context_projection.weight.grad is not None

    def test_gradients_flow_to_option_term_with_no_context(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options, mask = _options()

        result = head.score(options, mask, None, None)
        result[mask].sum().backward()

        assert head.option_term.weight.grad is not None
