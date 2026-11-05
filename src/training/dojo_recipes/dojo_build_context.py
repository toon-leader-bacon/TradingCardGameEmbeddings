"""DojoBuildContext, what every catalog recipe is built with, and the
DojoRecipe Protocol the catalog's entries satisfy."""

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from src.dojos.dojo import Dojo
from src.dojos.file_managers.neighbour_dealer import StudentSettings
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.mods.mod_specs import AugmentationRates, ModSpec
from src.schema.holdout import HoldoutSpec
from src.teacher.teacher_settings import TeacherViews
from src.training.dojo_recipes.card_shelf import CardShelf


@dataclass(frozen=True)
class DojoBuildContext:
    """What every recipe needs besides its own entry: the shared shelf and
    the run-wide settings every dojo must agree on.

    shelf: loaded card data. holdout: the plan's HoldoutSpec (the trainer
    requires every dojo to share it). card_embedding_size: the model's
    output embedding width. rng_seed: seeds each dojo's split shuffling, so
    a first split build is reproducible. mod_overrides: per dojo name,
    augmentation specs replacing that dojo's game defaults (an empty tuple
    means no augmentation); build_dojos rejects a name that is not built,
    and a deck spec on a dojo or group DECK_MOD_GROUPS does not allow.
    staple_thresholds: per contrastive dojo name, its staple-subsampling t
    (src/dojos/contrastive/staple_subsampling.py); a dojo not named keeps
    t = inf (no subsampling). build_dojos rejects a name that is not a
    contrastive dojo being built. regression_objective: the loss and
    baseline every regression dojo trains with (default: plain MSE); only
    the three regression cells take it (see bind_regression_objective).
    default_augmentation: rates rewriting every card-field mod of a game's
    default augmentations (not of mod_overrides); none by default.
    teacher: how each game's teacher per student split is built (the run
    config's `teacher:` section); None when the run has none, and then a teacher-sourced dojo
    refuses to build. student: how teacher-sourced dojos deal their
    pseudo-decks (the `student:` section; defaults when absent).
    """

    shelf: CardShelf
    holdout: HoldoutSpec
    card_embedding_size: int
    rng_seed: int
    mod_overrides: Mapping[str, tuple[ModSpec, ...]] = field(default_factory=dict)
    staple_thresholds: Mapping[str, float] = field(default_factory=dict)
    regression_objective: RegressionObjective = field(
        default_factory=RegressionObjective.mse
    )
    default_augmentation: AugmentationRates = AugmentationRates()
    teacher: TeacherViews | None = None
    student: StudentSettings = StudentSettings()


class DojoRecipe(Protocol):
    """How to build one catalog dojo."""

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """Construct the dojo, named name. Side effects: may load card
        data through context.shelf; may write split files on first build.
        Exceptions: FileNotFoundError for missing data; ValueError for a
        stale metric (the dojo's own version check)."""
        ...
