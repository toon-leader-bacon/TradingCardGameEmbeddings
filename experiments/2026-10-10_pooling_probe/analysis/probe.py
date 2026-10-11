"""Offline probe: how good is each frozen text-encoder / serialization /
pooling / post-processing option as the card vector our model starts from?

No training. For a fixed pool of cards (sampled decks + random cards per
game) it encodes every card once per (encoder, serialization), pools it
three ways and post-processes it four ways, then scores each variant on:

- anisotropy: mean pairwise cos, % variance on the top principal component;
- linear probes: 5-fold CV ridge R^2 / logistic balanced accuracy on a few
  card fields per game (affine-invariant, so only depends on encoder,
  serialization and pooling);
- cosine kNN probes: leave-one-out 10-NN balanced accuracy on the same class
  fields and on the game (depends on geometry, like our cosine losses);
- same-deck retrieval AUC: for a card in a sampled deck, P(cos to a random
  deck-mate > cos to a random same-game card outside the deck). 0.5 = chance.
  This is the contrastive dojos' task with no training.

Run (CPU; pass --threads to share cores with a training run):
    PYTHONPATH=. python experiments/2026-10-10_pooling_probe/analysis/probe.py [--threads 6]
Writes data/runs/2026-10-10_pooling_probe/legacy_logs/cache/*.pt (pooled vectors), results.csv and
prints a ranked table.
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from src.encoder_model.card_serialization import serialize_card_to_json
from src.schema.card import GenericCard
from src.schema.game_id import GameId
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox

OUT = Path("data/runs/2026-10-10_pooling_probe/legacy_logs")
CACHE = OUT / "cache"
SEED = 0
DECKS_PER_GAME = 25
RANDOM_CARDS_PER_GAME = 200
MAX_LENGTH = 384  # same cap as PretrainedTextEncoder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMES = [
    GameId.MTG,
    GameId.POKEMON,
    GameId.HEARTHSTONE,
    GameId.GWENT,
    GameId.SLAY_THE_SPIRE_2,
    GameId.FLESH_AND_BLOOD,
    GameId.DOMINION,
]


@dataclass(frozen=True)
class EncoderOption:
    name: str
    checkpoint: str
    prefix: str  # prepended to every text (task prefix some models expect)


ENCODERS = [
    EncoderOption("modernbert", "answerdotai/ModernBERT-base", ""),
    EncoderOption("nomic-embed", "nomic-ai/modernbert-embed-base", "clustering: "),
    EncoderOption("gte-modernbert", "Alibaba-NLP/gte-modernbert-base", ""),
]

# ---------------------------------------------------------------- labels


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_of(options: list[str]) -> Callable[[str], str | None]:
    def pick(text: Any) -> str | None:
        if not isinstance(text, str):
            return None
        return next((option for option in options if option in text), None)

    return pick


def _head(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else None


_MTG_TYPE = _first_of(
    ["Creature", "Land", "Instant", "Sorcery", "Enchantment", "Artifact", "Planeswalker"]
)

# game -> [(label name, "reg" | "cls", raw_content -> value or None)]
LABELS: dict[GameId, list[tuple[str, str, Callable[[dict], Any]]]] = {
    GameId.MTG: [
        ("cmc", "reg", lambda r: _num(r.get("cmc"))),
        ("type", "cls", lambda r: _MTG_TYPE(r.get("type_line"))),
        ("rarity", "cls", lambda r: r.get("rarity")),
    ],
    GameId.POKEMON: [
        ("supertype", "cls", lambda r: r.get("supertype")),
        ("hp", "reg", lambda r: _num(r.get("hp"))),
    ],
    GameId.HEARTHSTONE: [
        ("cost", "reg", lambda r: _num(r.get("cost"))),
        ("type", "cls", lambda r: r.get("type")),
        ("class", "cls", lambda r: r.get("cardClass")),
    ],
    GameId.GWENT: [
        ("provision", "reg", lambda r: _num(r.get("provision"))),
        ("faction", "cls", lambda r: r.get("faction")),
    ],
    GameId.SLAY_THE_SPIRE_2: [
        ("cost", "reg", lambda r: _num(r.get("cost"))),
        ("type", "cls", lambda r: r.get("type")),
        ("color", "cls", lambda r: r.get("color")),
    ],
    GameId.FLESH_AND_BLOOD: [
        ("pitch", "cls", lambda r: r.get("pitch")),
        ("cost", "reg", lambda r: _num(r.get("cost"))),
    ],
    GameId.DOMINION: [
        ("cost", "reg", lambda r: _num(r.get("cost"))),
        ("type", "cls", lambda r: _head(r.get("types"))),
    ],
}

# ---------------------------------------------------------------- card pool


@dataclass
class CardPool:
    cards: list[GenericCard]
    decks: list[tuple[GameId, list[int]]]  # indices into cards, unique per deck


def build_pool() -> CardPool:
    rng = random.Random(SEED)
    cards: list[GenericCard] = []
    index_of: dict[Any, int] = {}
    decks: list[tuple[GameId, list[int]]] = []

    def add(card: GenericCard) -> int:
        if card.nocab_uuid not in index_of:
            index_of[card.nocab_uuid] = len(cards)
            cards.append(card)
        return index_of[card.nocab_uuid]

    for game in GAMES:
        binder = CardBinder.load([CardBinder.default_output_path(game)])
        unknown = CardBinder.unknown_card_uuid(game)
        deck_path = DeckBox.default_output_path(game)
        if deck_path.exists():
            box = DeckBox.load([deck_path])
            taken = 0
            for deck_uuid, _, _ in box.uuids_ranked_randomly(game, SEED):
                deck = box.get_by_uuid(deck_uuid)
                if deck is None:
                    continue
                members = []
                for card_uuid in dict.fromkeys(deck.card_nocab_uuids):
                    card = binder.get_by_uuid(card_uuid)
                    if card is not None and card_uuid != unknown:
                        members.append(add(card))
                if len(members) >= 4:
                    decks.append((game, members))
                    taken += 1
                if taken >= DECKS_PER_GAME:
                    break
        all_cards = [c for c in binder.all_cards(game) if c.nocab_uuid != unknown]
        for card in rng.sample(all_cards, min(RANDOM_CARDS_PER_GAME, len(all_cards))):
            add(card)
        print(f"pool {game.value}: {sum(c.source_game == game for c in cards)} cards, "
              f"{sum(g == game for g, _ in decks)} decks", flush=True)
    return CardPool(cards, decks)


# ---------------------------------------------------------------- serialization
# Each returns (text, value character spans). A token is a "value token"
# when it overlaps a value span; keys, quotes, braces and commas are not.


def _json_value_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            k = j + 1
            while k < len(text) and text[k] == " ":
                k += 1
            if not (k < len(text) and text[k] == ":"):
                spans.append((i + 1, j))
            i = j + 1
        elif c in "{}[],: \n":
            i += 1
        else:  # number / true / false / null
            j = i
            while j < len(text) and text[j] not in ",}]":
                j += 1
            spans.append((i, j))
            i = j
    return spans


def serialize_json(card: GenericCard) -> tuple[str, list[tuple[int, int]]]:
    text = serialize_card_to_json(card)
    return text, _json_value_spans(text)


def serialize_key_value(card: GenericCard) -> tuple[str, list[tuple[int, int]]]:
    """'key: value' lines; scalar lists joined with ', '; nested dicts and
    lists of dicts flattened with a 'parent child' key."""
    lines: list[str] = []
    spans: list[tuple[int, int]] = []
    length = 0

    def emit(key: str, value: str) -> None:
        nonlocal length
        prefix = f"{key}: "
        start = length + len(prefix)
        lines.append(prefix + value)
        spans.append((start, start + len(value)))
        length += len(prefix) + len(value) + 1  # + newline

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix} {key}".strip(), inner)
        elif isinstance(value, list):
            if all(not isinstance(v, (dict, list)) for v in value):
                emit(prefix, ", ".join(str(v) for v in value))
            else:
                for inner in value:
                    walk(prefix, inner)
        else:
            emit(prefix, str(value))

    walk("", card.raw_content)
    return "\n".join(lines), spans


SERIALIZATIONS = {"json": serialize_json, "kv": serialize_key_value}

# ---------------------------------------------------------------- encoding


def encode(
    option: EncoderOption, texts: list[str], spans: list[list[tuple[int, int]]]
) -> dict[str, torch.Tensor]:
    """Pooled vectors (n, hidden) for pooling 'mean', 'cls', 'value', plus
    'tokens' (n,) real token counts."""
    tokenizer = AutoTokenizer.from_pretrained(option.checkpoint)
    model = AutoModel.from_pretrained(option.checkpoint).eval().to(DEVICE)
    shift = len(option.prefix)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    pooled: dict[str, list[torch.Tensor | None]] = {
        "mean": [None] * len(texts), "cls": [None] * len(texts), "value": [None] * len(texts)
    }
    token_counts = torch.zeros(len(texts))
    started = time.time()
    for batch_start in range(0, len(order), 16):
        batch = order[batch_start : batch_start + 16]
        tokenized = tokenizer(
            [option.prefix + texts[i] for i in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = tokenized.pop("offset_mapping")
        with torch.no_grad():
            hidden = model(**tokenized.to(DEVICE)).last_hidden_state.float().cpu()
        tokenized = tokenized.to("cpu")
        mask = tokenized["attention_mask"].float()
        for row, i in enumerate(batch):
            value_mask = torch.zeros_like(mask[row])
            for t, (a, b) in enumerate(offsets[row].tolist()):
                if mask[row, t] == 0 or b <= a:
                    continue
                a, b = a - shift, b - shift
                if any(a < s_end and b > s_start for s_start, s_end in spans[i]):
                    value_mask[t] = 1
            if value_mask.sum() == 0:
                value_mask = mask[row]
            h = hidden[row]
            pooled["mean"][i] = (h * mask[row, :, None]).sum(0) / mask[row].sum()
            pooled["cls"][i] = h[0]
            pooled["value"][i] = (h * value_mask[:, None]).sum(0) / value_mask.sum()
            token_counts[i] = mask[row].sum()
        done = batch_start + len(batch)
        if done % 400 < 16:
            print(f"  {option.name}: {done}/{len(texts)} ({time.time() - started:.0f}s)", flush=True)
    result = {name: torch.stack(vectors) for name, vectors in pooled.items()}  # type: ignore[arg-type]
    result["tokens"] = token_counts
    return result


# ---------------------------------------------------------------- post-processing


def post_process(x: torch.Tensor, how: str) -> torch.Tensor:
    if how == "raw":
        return x
    centered = x - x.mean(0)
    if how == "center":
        return centered
    _, s, v = torch.linalg.svd(centered, full_matrices=False)
    if how == "abtt8":  # all-but-the-top: drop the top 8 principal directions
        top = v[:8]
        return centered - centered @ top.T @ top
    if how == "whiten256":  # PCA to 256 dims, unit variance each
        components = v[:256]
        scale = s[:256] / math.sqrt(len(x) - 1)
        return (centered @ components.T) / scale
    raise ValueError(how)


POSTS = ["raw", "center", "abtt8", "whiten256"]

# ---------------------------------------------------------------- scores


def anisotropy(x: torch.Tensor) -> tuple[float, float]:
    unit = torch.nn.functional.normalize(x, dim=1)
    n = len(x)
    mean_cos = ((unit.sum(0).norm() ** 2 - n) / (n * (n - 1))).item()
    centered = x - x.mean(0)
    s = torch.linalg.svdvals(centered)
    top_share = (s[0] ** 2 / (s**2).sum()).item()
    return mean_cos, top_share


def label_sets(pool: CardPool) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """(name, kind, card indices, labels) for every usable probe label,
    including 'game' over the whole pool."""
    result = []
    games = np.array([c.source_game.value for c in pool.cards])
    result.append(("game", "cls", np.arange(len(pool.cards)), games))
    for game, specs in LABELS.items():
        for name, kind, extract in specs:
            indices, values = [], []
            for i, card in enumerate(pool.cards):
                if card.source_game != game:
                    continue
                value = extract(card.raw_content)
                if value is not None:
                    indices.append(i)
                    values.append(value)
            if kind == "cls":
                values_arr = np.array([str(v) for v in values])
                keep_classes = {v for v in set(values_arr) if (values_arr == v).sum() >= 5}
                keep = np.array([v in keep_classes for v in values_arr], dtype=bool)
                if len(keep_classes) < 2:
                    continue
                result.append((f"{game.value}.{name}", kind, np.array(indices)[keep], values_arr[keep]))
            else:
                if len(values) < 30:
                    continue
                result.append((f"{game.value}.{name}", kind, np.array(indices), np.array(values, dtype=float)))
    return result


def linear_probe(x: np.ndarray, kind: str, y: np.ndarray) -> float:
    if kind == "reg":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11)))
        folds = KFold(5, shuffle=True, random_state=SEED)
        return float(cross_val_score(model, x, y, cv=folds, scoring="r2").mean())
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    min_class = min(np.unique(y, return_counts=True)[1])
    folds = StratifiedKFold(min(5, min_class), shuffle=True, random_state=SEED)
    return float(cross_val_score(model, x, y, cv=folds, scoring="balanced_accuracy").mean())


def knn_balanced_accuracy(x: torch.Tensor, y: np.ndarray, k: int = 10) -> float:
    unit = torch.nn.functional.normalize(x, dim=1)
    sims = unit @ unit.T
    sims.fill_diagonal_(-2)
    neighbours = sims.topk(k, dim=1).indices.numpy()
    predicted = []
    for row in neighbours:
        values, counts = np.unique(y[row], return_counts=True)
        predicted.append(values[counts.argmax()])
    predicted_arr = np.array(predicted)
    return float(np.mean([(predicted_arr[y == c] == c).mean() for c in np.unique(y)]))


def deck_auc(x: torch.Tensor, pool: CardPool) -> dict[str, float]:
    unit = torch.nn.functional.normalize(x, dim=1)
    games = [c.source_game for c in pool.cards]
    per_game: dict[str, list[float]] = {}
    for game, members in pool.decks:
        member_set = set(members)
        others = [i for i, g in enumerate(games) if g == game and i not in member_set]
        if not others:
            continue
        sims_pos = unit[members] @ unit[members].T
        sims_neg = unit[members] @ unit[others].T  # (m, n_neg)
        neg_sorted = sims_neg.sort(dim=1).values
        aucs = []
        for a in range(len(members)):
            positives = torch.cat([sims_pos[a, :a], sims_pos[a, a + 1 :]])
            below = torch.searchsorted(neg_sorted[a].contiguous(), positives.contiguous())
            aucs.append((below.float() / neg_sorted.shape[1]).mean().item())
        per_game.setdefault(game.value, []).append(float(np.mean(aucs)))
    return {game: float(np.mean(values)) for game, values in per_game.items()}


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    CACHE.mkdir(parents=True, exist_ok=True)

    pool = build_pool()
    labels = label_sets(pool)
    print(f"{len(pool.cards)} cards, {len(pool.decks)} decks, {len(labels)} probe labels", flush=True)

    rows: list[dict[str, Any]] = []
    for option in ENCODERS:
        for serialization_name, serialize in SERIALIZATIONS.items():
            cache_file = CACHE / f"{option.name}.{serialization_name}.pt"
            if cache_file.exists():
                pooled = torch.load(cache_file)
            else:
                serialized = [serialize(card) for card in pool.cards]
                print(f"encoding {option.name} / {serialization_name}", flush=True)
                pooled = encode(option, [t for t, _ in serialized], [s for _, s in serialized])
                torch.save(pooled, cache_file)
            mean_tokens = pooled["tokens"].mean().item()
            for pooling in ["mean", "cls", "value"]:
                vectors = pooled[pooling]
                linear = {
                    name: linear_probe(vectors[idx].numpy(), kind, y)
                    for name, kind, idx, y in labels
                    if name != "game"
                }
                for post in POSTS:
                    x = post_process(vectors, post)
                    mean_cos, top_share = anisotropy(x)
                    knn = {
                        name: knn_balanced_accuracy(x[idx], y)
                        for name, kind, idx, y in labels
                        if kind == "cls"
                    }
                    aucs = deck_auc(x, pool)
                    row = {
                        "encoder": option.name,
                        "serialization": serialization_name,
                        "pooling": pooling,
                        "post": post,
                        "tokens": round(mean_tokens, 1),
                        "mean_cos": round(mean_cos, 3),
                        "top_pc_share": round(top_share, 3),
                        "linear_reg_r2": round(float(np.mean([v for k, v in linear.items() if dict((n, kd) for n, kd, _, _ in labels)[k] == "reg"])), 3),
                        "linear_cls_bacc": round(float(np.mean([v for k, v in linear.items() if dict((n, kd) for n, kd, _, _ in labels)[k] == "cls"])), 3),
                        "knn_game_bacc": round(knn["game"], 3),
                        "knn_field_bacc": round(float(np.mean([v for k, v in knn.items() if k != "game"])), 3),
                        "deck_auc": round(float(np.mean(list(aucs.values()))), 3),
                        **{f"auc.{k}": round(v, 3) for k, v in aucs.items()},
                        **{f"lin.{k}": round(v, 3) for k, v in linear.items()},
                        **{f"knn.{k}": round(v, 3) for k, v in knn.items()},
                    }
                    rows.append(row)
                    print(
                        f"{option.name:15s} {serialization_name:4s} {pooling:5s} {post:9s} | "
                        f"cos {mean_cos:.3f} top {top_share:.2f} | lin r2 {row['linear_reg_r2']:.3f} "
                        f"cls {row['linear_cls_bacc']:.3f} | knn game {row['knn_game_bacc']:.3f} "
                        f"field {row['knn_field_bacc']:.3f} | deck auc {row['deck_auc']:.3f}",
                        flush=True,
                    )

    fields = sorted({k for row in rows for k in row}, key=lambda k: (k.count("."), list(rows[0]).index(k) if k in rows[0] else 999, k))
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT / "results.csv")


if __name__ == "__main__":
    main()
