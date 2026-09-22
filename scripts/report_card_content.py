"""Reports how lean one source's card content is, in encoder tokens.

Ingests a source's raw data into a throwaway, in-memory CardBinder (nothing
is written under data/) and reports, for the cards' serialized text:
token percentiles, the share over 256 and 384 tokens, the costliest keys,
and how many cards serialize identically (should be 0). Run it when
converting or onboarding a card ingestion stage; see "What goes in
raw_content" in src/data_refinement/card_binder/README.md.

Usage (from the project root, in an environment with transformers):

    PYTHONPATH=. python scripts/report_card_content.py --list
    PYTHONPATH=. python scripts/report_card_content.py --source scryfall
    PYTHONPATH=. python scripts/report_card_content.py --source spire_codex \\
        --raw-path data/raw/spire_codex/cards.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer

from scripts.run_card_binder_ingestion import STAGES
from src.data_refinement.card_binder.card_binder import CardBinder
from src.encoder_model.card_serialization import serialize_card_to_json
from src.schema.card import GenericCard

_DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"
_KEY_COST_SAMPLE = 2000
_TOP_KEYS = 12
_LONG_CAPS = (256, 384)

TokenCounter = Callable[[str], int]


def build_cards(source: str, raw_path: Path | None) -> list[GenericCard]:
    """Ingest a source into an empty in-memory binder; return its cards.

    Inputs: source (key of STAGES), raw_path (Path | None, None uses the
        source's default raw location).
    Output: the binder's cards that have content (the "Unknown" sentinel
        has none and is left out).
    Side effects: reads the raw data; prints ingestion progress.
    Exceptions: KeyError for an unknown source; whatever the stage raises
        (e.g. FileNotFoundError when the raw data is missing).

    Example:
        >>> cards = build_cards("scryfall", None)
    """
    stage, default_raw_path = STAGES[source]
    binder = CardBinder.load([])
    stage.ingest(raw_path or default_raw_path(), binder)
    return [c for c in binder.all_cards(stage.SOURCE_GAME) if c.raw_content]


def make_token_counter(tokenizer_name: str) -> TokenCounter:
    """A function counting a text's tokens, without special tokens.

    Inputs: tokenizer_name (str): a Hugging Face tokenizer id.
    Output: Callable[[str], int].
    Side effects: may download the tokenizer on first use.
    Exceptions: whatever AutoTokenizer.from_pretrained raises.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def percentile(sorted_values: list[int], fraction: float) -> int:
    """The value at the given fraction (0-1) of an ascending list."""
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def print_token_summary(counts: list[int]) -> None:
    """Print median/p90/p99/max and the share of cards over each cap."""
    ordered = sorted(counts)
    print(
        f"tokens per card: median={percentile(ordered, 0.5)} "
        f"p90={percentile(ordered, 0.9)} p99={percentile(ordered, 0.99)} "
        f"max={ordered[-1]}"
    )
    shares = ", ".join(
        f"over {cap}: {sum(c > cap for c in counts) / len(counts):.1%}"
        for cap in _LONG_CAPS
    )
    print(f"share of cards {shares}")


def print_costliest_keys(cards: list[GenericCard], count_tokens: TokenCounter) -> None:
    """Print each top-level key's average tokens per card (compact JSON of
    its value only), costliest first, over a sample of cards."""
    sample = random.Random(0).sample(cards, min(_KEY_COST_SAMPLE, len(cards)))
    totals: Counter[str] = Counter()
    present: Counter[str] = Counter()
    for card in sample:
        for key, value in card.raw_content.items():
            totals[key] += count_tokens(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
            present[key] += 1
    print(f"costliest keys (avg tokens over {len(sample)} sampled cards):")
    for key, total in totals.most_common(_TOP_KEYS):
        print(
            f"  {key:32s} {total / len(sample):6.1f}  present in {present[key] / len(sample):4.0%}"
        )


def report(source: str, raw_path: Path | None, tokenizer_name: str) -> None:
    """Print the full content report for one source."""
    cards = build_cards(source, raw_path)
    count_tokens = make_token_counter(tokenizer_name)
    texts = [serialize_card_to_json(card) for card in cards]
    print(f"\n=== {source}: {len(cards)} cards with content ===")
    print_token_summary([count_tokens(text) for text in texts])
    duplicates = sum(n - 1 for n in Counter(texts).values() if n > 1)
    print(f"cards serializing identically to another card: {duplicates}")
    print_costliest_keys(cards, count_tokens)
    print("example:", texts[len(texts) // 2][:400])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="List sources and exit.")
    parser.add_argument("--source", choices=sorted(STAGES), help="Source to report.")
    parser.add_argument("--raw-path", type=Path, help="Override the raw data path.")
    parser.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list or not args.source:
        print("\n".join(sorted(STAGES)))
        return
    report(args.source, args.raw_path, args.tokenizer)


if __name__ == "__main__":
    main()
