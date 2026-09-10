import pytest
import torch

from src.dojos.loss.soft_classification_loss import SoftClassificationLoss


class TestCalculate:
    def test_zero_loss_for_confident_correct_single_character_label(self) -> None:
        loss_calculator = SoftClassificationLoss(["a", "b"])
        # Large positive logit on the correct class, large negative on
        # the other - drives soft cross-entropy toward 0, same as
        # FixedClassificationLoss's degenerate (hard-label) case.
        decoder_output = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])

        result = loss_calculator.calculate(decoder_output, [{"a": 1.0}, {"b": 1.0}])

        assert result.item() == pytest.approx(0.0, abs=1e-3)

    def test_single_character_label_matches_hard_cross_entropy(self) -> None:
        # A single-character, probability-1.0 label should reduce to
        # ordinary cross-entropy: -log_softmax(logits)[correct_class].
        loss_calculator = SoftClassificationLoss(["a", "b"])
        decoder_output = torch.tensor([[0.0, 0.0]])

        result = loss_calculator.calculate(decoder_output, [{"a": 1.0}])

        # Uniform logits over 2 classes -> -log(1/2)
        assert result.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())

    def test_multi_character_soft_label_combines_both_entries(self) -> None:
        loss_calculator = SoftClassificationLoss(["a", "b"])
        decoder_output = torch.tensor([[0.0, 0.0]])

        result = loss_calculator.calculate(decoder_output, [{"a": 0.5, "b": 0.5}])

        # log_softmax is -log(2) for both classes here, so
        # -(0.5*-log(2) + 0.5*-log(2)) == log(2), same value as the
        # single-character case above - a coincidence of the uniform
        # logits, not a general equivalence; verifies both dict entries
        # actually contribute (a broken implementation reading only one
        # key would instead give 0.5 * log(2)).
        assert result.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())

    def test_averages_over_the_batch_not_sum(self) -> None:
        loss_calculator = SoftClassificationLoss(["a", "b"])
        single_example_output = torch.tensor([[0.0, 0.0]])
        single_example_loss = loss_calculator.calculate(
            single_example_output, [{"a": 1.0}]
        )

        batched_output = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        batched_loss = loss_calculator.calculate(
            batched_output, [{"a": 1.0}, {"a": 1.0}]
        )

        assert batched_loss.item() == pytest.approx(single_example_loss.item())

    def test_raises_on_mismatched_lengths(self) -> None:
        loss_calculator = SoftClassificationLoss(["a", "b"])
        decoder_output = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [{"a": 1.0}, {"b": 1.0}])

    def test_raises_on_character_outside_label_values(self) -> None:
        loss_calculator = SoftClassificationLoss(["a", "b"])
        decoder_output = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [{"c": 1.0}])
