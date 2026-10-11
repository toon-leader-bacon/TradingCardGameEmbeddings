"""Card-embedding scale and similarity at init and at each saved checkpoint
of a run: argv[1] = config path, argv[2] = run directory."""

import sys
from pathlib import Path

import torch

from scripts.run_training import build_model
from src.training.dojo_catalog import DOJO_CATALOG
from src.training.run_config import parse_run_config, read_config_document

TEXTS = [
    '{"name": "Lightning Bolt", "game": "mtg", "mana_cost": "{R}", "type_line": "Instant", "oracle_text": "Lightning Bolt deals 3 damage to any target."}',
    '{"name": "Pikachu", "game": "pokemon", "hp": 60, "types": ["Lightning"], "attacks": [{"name": "Thunder Shock", "damage": "20"}]}',
    '{"name": "Village", "game": "dominion", "cost": 3, "types": ["Action"], "text": "+1 Card, +2 Actions"}',
    '{"name": "Snatch", "game": "flesh_and_blood", "pitch": 1, "cost": 0, "power": 4, "text": "If Snatch hits, draw a card."}',
    '{"name": "Bash", "game": "slay_the_spire_2", "cost": 2, "type": "Attack", "rarity": "Basic", "text": "Deal 8 damage. Apply 2 Vulnerable."}',
    '{"name": "Wrath of God", "game": "mtg", "mana_cost": "{2}{W}{W}", "type_line": "Sorcery", "oracle_text": "Destroy all creatures."}',
    '{"name": "Province", "game": "dominion", "cost": 8, "types": ["Victory"], "text": "6 VP"}',
    '{"name": "Professor Research", "game": "pokemon", "supertype": "Trainer", "rules": ["Discard your hand and draw 7 cards."]}',
    '{"name": "Llanowar Elves", "game": "mtg", "mana_cost": "{G}", "type_line": "Creature - Elf Druid", "power": "1", "toughness": "1"}',
    '{"name": "Defend", "game": "slay_the_spire_2", "cost": 1, "type": "Skill", "text": "Gain 5 Block."}',
    '{"name": "Copper", "game": "dominion", "cost": 0, "types": ["Treasure"], "text": "$1"}',
    '{"name": "Command and Conquer", "game": "flesh_and_blood", "cost": 2, "power": 6, "defense": 3, "pitch": 1}',
]


def mean_cosine(x: torch.Tensor) -> float:
    c = torch.nn.functional.cosine_similarity(x.unsqueeze(0), x.unsqueeze(1), dim=-1)
    return c[~torch.eye(len(x), dtype=torch.bool)].mean().item()


def report(tag: str, model: torch.nn.Module, bert: torch.Tensor) -> None:
    base = model.embedding_head(bert)
    out = (
        model.self_attention(base.unsqueeze(1)).squeeze(1)
        if hasattr(model, "self_attention")
        else base
    )
    centered = out - out.mean(0)
    print(
        f"{tag:10s} head norm {base.norm(dim=1).mean():7.2f} | final norm "
        f"{out.norm(dim=1).mean():7.2f} cos {mean_cosine(out):.3f} | shared-mean norm "
        f"{out.mean(0).norm():7.2f} spread {centered.norm(dim=1).mean():6.2f}"
    )


config = parse_run_config(read_config_document(Path(sys.argv[1])), tuple(DOJO_CATALOG))
run = Path(sys.argv[2])
torch.manual_seed(0)
model = build_model(config.model).eval()
with torch.no_grad():
    bert = model.text_encoder.encode(TEXTS)
    report("init", model, bert)
    for directory in sorted(run.glob("train_round*")) + [run / "latest"]:
        state = torch.load(directory / "encoder.pt", map_location="cpu")
        model.load_state_dict(state)
        report(directory.name, model, bert)
