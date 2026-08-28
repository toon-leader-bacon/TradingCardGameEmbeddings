# losses

A small shared library of reusable `Loss` Strategy objects a dojo can
inject into its own constructor instead of hand-rolling loss math.
`training/` never constructs, sees, or calls a `Loss` — it's entirely
internal to whichever dojo composed it into its own `compute_loss`.

## Files

- `loss.py` — `Loss[PredictionsT, LabelsT]`, the shared
  `typing.Protocol` every loss implements: `compute(predictions,
  labels) -> torch.Tensor`. `PredictionsT` is contravariant (used only
  as an input); `LabelsT` is invariant (wrapped in `list[LabelsT]`,
  and `list` itself is invariant).
- `cross_entropy_loss.py` — `CrossEntropyLoss`, a
  `Loss[torch.Tensor, int]`: standard cross-entropy over per-class
  logits, with configurable label smoothing, delegating to
  `torch.nn.CrossEntropyLoss` internally.

## How to run

```python
from src.dojos.losses.cross_entropy_loss import CrossEntropyLoss

loss = CrossEntropyLoss(label_smoothing=0.1)
loss.compute(predictions, labels)  # predictions: (batch, num_classes) logits
```
