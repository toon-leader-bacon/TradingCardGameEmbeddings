"""Field sharing over teacher-neighbour pairs, the T arm's view of
plans/teacher_full_test_readiness.md steps 2.6-2.7.

For each confident anchor, one neighbour drawn uniformly from its list
(as the teacher-contrastive dojo deals pairs) against one random pair of
confident cards. Prints, per top-level raw_content field, the share of
pairs with equal values and the lift; a high lift means the field alone
nearly names a neighbour, so it is a shortcut to mask.

Usage (project root, PYTHONPATH=.):
    python neighbour_field_sharing.py <game> <teacher.arrow> <neighbours.arrow> [<label>]
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.game_id import GameId
from src.teacher.neighbour_rules import TopKNeighbours
from src.teacher.teacher_neighbours import TeacherNeighbours
from src.teacher.teacher_table import TeacherTable

DRAWS_PER_ANCHOR = 4
SEED = 0


def load_fields(game: GameId) -> dict[str, dict[str, str]]:
    """card uuid -> {key: canonical JSON of its value}, skipping cards with
    no raw_content (the unknown sentinel)."""
    cards: dict[str, dict[str, str]] = {}
    with CardBinder.default_output_path(game).open(encoding="utf-8") as lines:
        for card in map(json.loads, lines):
            raw = card.get("raw_content") or {}
            if raw:
                cards[card["nocab_uuid"]] = {k: json.dumps(v, sort_keys=True) for k, v in raw.items()}
    return cards


def main() -> None:
    game = GameId(sys.argv[1])
    table = TeacherTable.open(Path(sys.argv[2]))
    neighbours = TeacherNeighbours.open(Path(sys.argv[3]), TopKNeighbours(k=32))
    label = sys.argv[4] if len(sys.argv) > 4 else sys.argv[2]
    fields = load_fields(game)
    uuids = [str(u) for u in table.card_uuids]
    anchors = [row for row in range(table.card_uuids.__len__()) if table.confident[row]]
    rng = random.Random(SEED)
    keys = Counter(k for card in fields.values() for k in card)
    same: Counter[str] = Counter()
    rand: Counter[str] = Counter()
    pairs = 0
    for row in anchors:
        partners = neighbours.neighbours_of(row)
        if len(partners) == 0:
            continue
        for _ in range(DRAWS_PER_ANCHOR):
            a = fields.get(uuids[row])
            b = fields.get(uuids[int(rng.choice(partners))])
            ra = fields.get(uuids[rng.choice(anchors)])
            rb = fields.get(uuids[rng.choice(anchors)])
            if a is None or b is None or ra is None or rb is None:
                continue
            pairs += 1
            for key in keys:
                same[key] += key in a and a.get(key) == b.get(key)
                rand[key] += key in ra and ra.get(key) == rb.get(key)
    print(f"\n== {game} {label}: {len(anchors):,} confident anchors, {pairs:,} pairs")
    print(f"{'field':24s} {'neighbour':>10s} {'random':>8s} {'lift':>6s}")
    for key in sorted(keys, key=lambda k: -same[k]):
        s, r = same[key] / pairs, rand[key] / pairs
        if s < 0.01 and r < 0.01:
            continue
        lift = s / r if r else float("inf")
        print(f"{key:24s} {s:10.0%} {r:8.0%} {lift:6.1f}")


if __name__ == "__main__":
    main()
