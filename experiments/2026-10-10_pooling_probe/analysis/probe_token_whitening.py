"""Token whitening before mean pooling, scored like probe.py's rows.

Mean pooling is linear, so mean(whiten(tokens)) = (mean(tokens) - mu) @ P
with mu, P fitted on token states. This needs one pass per
(encoder, serialization) to collect token mean and covariance; the
mean-pooled vectors come from probe.py's cache (run probe.py first).

Run: PYTHONPATH=. python logs/pooling_probe/probe_token_whitening.py
Writes logs/pooling_probe/results_token_whitening.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import probe  # noqa: E402

COMPONENTS = [256, 768]
RELATIVE_EPS = 1e-5  # added to eigenvalues, relative to the largest


def token_statistics(option: probe.EncoderOption, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance (float64) over every real token of texts."""
    tokenizer = AutoTokenizer.from_pretrained(option.checkpoint)
    model = AutoModel.from_pretrained(option.checkpoint).eval().to(probe.DEVICE)
    total = None
    outer = None
    count = 0
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), 16):
        batch = [option.prefix + texts[i] for i in order[start : start + 16]]
        tokenized = tokenizer(batch, padding=True, truncation=True,
                              max_length=probe.MAX_LENGTH, return_tensors="pt").to(probe.DEVICE)
        with torch.no_grad():
            hidden = model(**tokenized).last_hidden_state
        tokens = hidden[tokenized["attention_mask"].bool()].double()
        total = tokens.sum(0) if total is None else total + tokens.sum(0)
        outer = tokens.T @ tokens if outer is None else outer + tokens.T @ tokens
        count += len(tokens)
    mean = total / count
    covariance = outer / count - torch.outer(mean, mean)
    return mean.cpu(), covariance.cpu()


def whitening_projection(covariance: torch.Tensor, components: int) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)[:components]
    eps = RELATIVE_EPS * eigenvalues.max()
    return eigenvectors[:, order] / torch.sqrt(eigenvalues[order].clamp(min=0) + eps)


def main() -> None:
    torch.set_num_threads(6)
    pool = probe.build_pool()
    labels = probe.label_sets(pool)
    rows = []
    for option in probe.ENCODERS:
        for serialization_name, serialize in probe.SERIALIZATIONS.items():
            pooled = torch.load(probe.CACHE / f"{option.name}.{serialization_name}.pt")["mean"]
            texts = [serialize(card)[0] for card in pool.cards]
            mean, covariance = token_statistics(option, texts)
            for components in COMPONENTS:
                projection = whitening_projection(covariance, components)
                x = ((pooled.double() - mean) @ projection).float()
                for post in ["raw", "center"]:
                    y = probe.post_process(x, post)
                    mean_cos, top_share = probe.anisotropy(y)
                    knn = {name: probe.knn_balanced_accuracy(y[idx], lab)
                           for name, kind, idx, lab in labels if kind == "cls"}
                    aucs = probe.deck_auc(y, pool)
                    row = {
                        "encoder": option.name,
                        "serialization": serialization_name,
                        "pooling": "mean",
                        "post": f"tokwhiten{components}+{post}",
                        "mean_cos": round(mean_cos, 3),
                        "top_pc_share": round(top_share, 3),
                        "knn_game_bacc": round(knn["game"], 3),
                        "knn_field_bacc": round(float(np.mean([v for k, v in knn.items() if k != "game"])), 3),
                        "deck_auc": round(float(np.mean(list(aucs.values()))), 3),
                        **{f"auc.{k}": round(v, 3) for k, v in aucs.items()},
                    }
                    rows.append(row)
                    print(f"{option.name:15s} {serialization_name:4s} {row['post']:18s} | cos {mean_cos:.3f} | "
                          f"knn game {row['knn_game_bacc']:.3f} field {row['knn_field_bacc']:.3f} | "
                          f"deck auc {row['deck_auc']:.3f}", flush=True)
    with open(probe.OUT / "results_token_whitening.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
