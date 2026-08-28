"""The shared output type of every encoder_model type.

Canonical home for Embedding — encoder_model owns this type (see
plans/encoder_model.md and plans/training_pipeline.md's note that
Embedding is defined here), not dojos/ or training/, since
encoder_model must not depend on either of them.
"""

import torch

Embedding = torch.Tensor  # shape (embedding_dim,)
