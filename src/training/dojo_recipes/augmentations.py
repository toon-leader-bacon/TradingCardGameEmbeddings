"""A catalog dojo's train-only augmentations: its game's defaults, or the
run config's `mods:` override for it."""

import random

from src.dojos.augmentation_defaults import default_augmentations_for
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.game_id import GameId
from src.schema.type_hints import LabelT
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext


def build_augmentation_pipeline(
    name: str, game: GameId, context: DojoBuildContext
) -> ModPipeline:
    """The dojo's augmentations: context.mod_overrides[name] if present,
    else game's defaults. Each mod's seed is drawn from a stream seeded by
    (rng_seed, name, "mods"), so mods neither share a random stream with
    each other nor with the dojo's own sampling (seeded by rng_seed
    itself), and each dojo's mods differ.

    Inputs: name, game, context. Output: ModPipeline.
    Side effects: none. Exceptions: as a ModSpec's build.
    """
    specs = context.mod_overrides.get(name, default_augmentations_for(game))
    seeds = random.Random(f"{context.rng_seed}:{name}:mods")
    return ModPipeline([spec.build(seeds.randrange(2**32)) for spec in specs])


def append_augmentations(
    dojo: GenericDojo[LabelT], name: str, game: GameId, context: DojoBuildContext
) -> GenericDojo[LabelT]:
    """dojo with its augmentations (build_augmentation_pipeline) appended after
    its own task mods, which stay first and in order.

    Inputs: dojo (a just-built metric dojo, no batch yet), name, game,
        context. Output: dojo itself.
    Side effects: dojo.append_mods when there is anything to append.
    Exceptions: as a ModSpec's build.
    """
    mods = build_augmentation_pipeline(name, game, context).mods
    if mods:
        dojo.append_mods(mods)
    return dojo
