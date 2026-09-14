import torch

from src.dojos.generic.option_scoring import BilinearOptionScoringHead


class TestScore:
    def test_returns_one_logit_per_option(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options = [torch.randn(4), torch.randn(4), torch.randn(4)]

        result = head.score(options, context=None)

        assert result.shape == (3,)

    def test_context_changes_the_score(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options = [torch.randn(4), torch.randn(4)]

        without_context = head.score(options, context=None)
        with_context = head.score(options, context=torch.randn(4))

        assert not torch.allclose(without_context, with_context)

    def test_gradients_flow_to_both_terms_when_context_given(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options = [torch.randn(4), torch.randn(4)]
        context = torch.randn(4)

        result = head.score(options, context=context)
        result.sum().backward()

        assert head.option_term.weight.grad is not None
        assert head.context_projection.weight.grad is not None

    def test_gradients_flow_to_option_term_with_no_context(self) -> None:
        head = BilinearOptionScoringHead(card_embedding_size=4)
        options = [torch.randn(4), torch.randn(4)]

        result = head.score(options, context=None)
        result.sum().backward()

        assert head.option_term.weight.grad is not None
