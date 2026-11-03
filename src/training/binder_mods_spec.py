"""BinderModsSpec: the run config's `binder_mods` section, parsed.

A light module (mod specs and a path, nothing that loads a binder), so
config parsing can import it without importing the bake in
binder_mods.py.
"""

from dataclasses import dataclass
from pathlib import Path

from src.dojos.mods.mod_specs import CardFieldModSpec


@dataclass(frozen=True)
class BinderModsSpec:
    """The mod pipeline applied once to every card of each game the run
    loads, before any dojo is built (see binder_mods.py).

    pipeline: card-field mod specs, in order; non-empty.
    seed: seeds the pipeline's mods; separate from the run's seed so arms
        that vary only the run seed share one saved bake.
    save_directory: where each game's modded binder is saved and reused
        from; None means it is applied in memory only.
    """

    pipeline: tuple[CardFieldModSpec, ...]
    seed: int = 0
    save_directory: Path | None = None

    def __post_init__(self) -> None:
        """Reject an empty pipeline or a non-card-field spec.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError for an empty pipeline; TypeError for a
            pipeline entry that is not a CardFieldModSpec.
        """
        if not self.pipeline:
            raise ValueError("binder_mods pipeline must not be empty")
        for spec in self.pipeline:
            if not isinstance(spec, CardFieldModSpec):
                raise TypeError(
                    f"binder_mods pipeline takes card-field mod specs, got {spec!r}"
                )
