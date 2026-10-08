"""Every dojo a training run can name, and how to build it from disk.

A run config lists dojos by catalog key ("gwent_one.color_mask"). The key
becomes the dojo's `name`, so dojos from different games whose metric
files share a stem (gwent_one and dominiontabs both have set_mask) cannot
collide in split files, checkpoints or logs. Each entry is a recipe: which
game's CardBinder it needs, whether it needs a deck box, and which dojo
class to construct.

Loading is lazy and shared: a CardShelf loads each game's binder (and each
deck box) once, only when a named dojo needs it.

A recipe may point an existing dojo class at another metric's output
(metric_output): the sts2_runs keys reuse the sts_gg wrappers over
data/metrics/sts2_runs/, whose files share the sts_gg label columns.

Contrastive dojos read a game's final deck box directly (no metric).

Every dojo takes train-only augmentation mods: its game's defaults
(src/dojos/augmentation_defaults.py) unless the run config's `mods:` names
that dojo, which replaces them (an empty list turns augmentation off). A
metric dojo keeps its own task mods (e.g. a train_only=False mask) first
and in order; its augmentations run after them. Deck mods
(src/dojos/mods/deck_mods.py) are never a default: only the dojos in
DECK_MOD_GROUPS take them, and only when `mods:` lists them.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.card_binder.multi_game_card_lookup import MultiGameCardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.cross_game.rarity.translator_tables import (
    RARITY_TRANSLATORS,
)
from src.data_refinement.metrics.isotropic.deck_box_path import ISOTROPIC_DECK_BOX_PATH
from src.data_refinement.metrics.seventeenlands.deck_box_path import (
    REPLAY_DATA_DECK_BOX_PATH,
)
from src.data_refinement.metrics.sts2_runs import card_average_metrics as sts2_cards
from src.data_refinement.metrics.sts2_runs import deck_label_metrics as sts2_decks
from src.data_refinement.metrics.sts_gg.deck_box_path import STS_GG_DECK_BOX_PATH
from src.dojos.augmentation_defaults import default_augmentations_for
from src.dojos.contrastive.dojo import ContrastiveDojo
from src.dojos.contrastive.pair_constructor import SingleCardPairConstructor
from src.dojos.contrastive.staple_subsampling import (
    StapleSubsampling,
    cached_document_frequency,
)
from src.dojos.dojo import Dojo
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.dojos.final_decks import held_out_card_dojos as final_decks
from src.dojos.cross_game.rarity_tier_dojo import RarityTierDojo
from src.dojos.dominiontabs.cost_regression_dojo import CostRegressionDojo
from src.dojos.dominiontabs.masked_field_dojos import SetMaskDojo, TypeMaskDojo
from src.dojos.gwent_one import masked_field_dojos as gwent_one
from src.dojos.cardvault_fabtcg import card_mask_dojos as fabtcg_masks
from src.dojos.fabtcg_decklists import card_inclusion_dojos as fabtcg_inclusion
from src.dojos.fabtcg_decklists import deck_card_mask_dojos as fabtcg_decks
from src.dojos.hearthstonejson import card_mask_dojos as hearthstone_masks
from src.dojos.pokemon_tcg import card_mask_dojos as pokemon_masks
from src.dojos.scryfall import card_mask_dojos as scryfall_masks
from src.dojos.seventeenlands.draft_data import pack_card_tally_dojos as draft_tally
from src.dojos.seventeenlands.draft_data.pack_to_pick_choice_set_dojo import (
    PackToPickChoiceSetDojo,
)
from src.dojos.seventeenlands.draft_data.pick_number_decay_curve_dojo import (
    PickNumberDecayCurveDojo,
)
from src.dojos.seventeenlands.draft_data.pool_conditioned_pick_dojo import (
    PoolConditionedPickDojo,
)
from src.dojos.seventeenlands.game_data import game_card_average_dojos as game_cards
from src.dojos.seventeenlands.game_data import game_deck_label_dojos as game_decks
from src.dojos.seventeenlands.game_data.game_length_association_dojo import (
    GameLengthAssociationDojo,
)
from src.dojos.seventeenlands.game_data.on_play_win_rate_delta_dojo import (
    OnPlayWinRateDeltaDojo,
)
from src.dojos.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_dojo import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckDojo,
)
from src.dojos.seventeenlands.game_data.tutor_choice_rate_dojo import (
    TutorChoiceRateDojo,
)
from src.dojos.seventeenlands.game_data.tutor_target_rate_dojo import (
    TutorTargetRateDojo as GameTutorTargetRateDojo,
)
from src.dojos.seventeenlands.replay_data import (
    replay_turn_event_rate_dojos as replay_combat,
)
from src.dojos.seventeenlands.replay_data.attacker_blocker_combat_outcome_dojo import (  # noqa: E501
    AttackerBlockerCombatOutcomeDojo,
)
from src.dojos.seventeenlands.replay_data.average_turn_cast_dojo import (
    AverageTurnCastDojo,
)
from src.dojos.seventeenlands.replay_data.cast_rate_dojo import CastRateDojo
from src.dojos.seventeenlands.replay_data.combat_aggression_profile_dojo import (
    CombatAggressionProfileDojo,
)
from src.dojos.seventeenlands.replay_data.discard_rate_dojo import DiscardRateDojo
from src.dojos.seventeenlands.replay_data.turns_to_game_end_after_cast_dojo import (
    TurnsToGameEndAfterCastDojo,
)
from src.dojos.seventeenlands.replay_data.tutor_target_rate_dojo import (
    TutorTargetRateDojo as ReplayTutorTargetRateDojo,
)
from src.dojos.spire_codex import card_mask_dojos as sts2_masks
from src.dojos.sts2_runs.card_removal_pick_dojo import CardRemovalPickDojo
from src.dojos.sts2_runs.card_reward_pick_dojo import CardRewardPickDojo
from src.dojos.sts2_runs.card_upgrade_pick_dojo import CardUpgradePickDojo
from src.dojos.sts2_runs.shop_purchase_pick_dojo import ShopPurchasePickDojo
from src.dojos import isotropic
from src.dojos.sts_gg import card_average_dojos as sts_cards
from src.dojos.sts_gg.card_character_prediction_dojo import (
    CardCharacterPredictionDojo,
)
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.per_game_mod import PerGameMod
from src.dojos.mods.mod_specs import DeckModSpec, ModSpec
from src.dojos.play_gwent import card_inclusion_dojos as gwent_inclusion
from src.dojos.play_gwent.deck_card_mask_dojos import LeaderMaskedFromDeckDojo
from src.dojos.play_gwent.deck_label_dojos import GuideVotesDojo
from src.dojos.sts_gg import deck_label_dojos as sts_decks
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# Contrastive dojos keep their deck split index here (see DeckBoxDealer)
_CONTRASTIVE_INDEX_DIRECTORY = Path("data/splits/contrastive")

# TRAIN decks a staple-subsampling document frequency is counted over
_DOCUMENT_FREQUENCY_SAMPLE_DECKS = 20_000


class CardDojoConstructor(Protocol):
    """A dojo class built from a CardBinder alone (every gwent_one and
    dominiontabs dojo, every sts_gg and isotropic card-level dojo)."""

    def __call__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        *,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> GenericDojo: ...


class DeckDojoConstructor(Protocol):
    """A dojo class that also needs the DeckBox its metric was built from
    (every sts_gg and isotropic deck-level dojo, play_gwent's leader
    mask)."""

    def __call__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        *,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> GenericDojo: ...


class CardShelf:
    """Loads each game's CardBinder and each DeckBox at most once."""

    def __init__(self) -> None:
        """Side effects: none (nothing loads until asked)."""
        self._card_binders: dict[GameId, CardBinder] = {}
        self._deck_boxes: dict[Path, DeckBox] = {}

    def card_binder(self, game: GameId) -> CardBinder:
        """game's binder from CardBinder.default_output_path, loaded on
        first request.

        Inputs: game (GameId). Output: CardBinder.
        Side effects: reads the binder file on first call per game.
        Exceptions: FileNotFoundError if the binder file does not exist.

        Example:
            >>> CardShelf().card_binder(GameId.GWENT)
        """
        if game not in self._card_binders:
            path = CardBinder.default_output_path(game)
            if not path.exists():
                raise FileNotFoundError(
                    f"no {game.value} card binder at {path}; run "
                    "scripts/run_card_binder_ingestion.py first"
                )
            self._card_binders[game] = CardBinder.load([path])
        return self._card_binders[game]

    def card_lookup(self, games: Sequence[GameId]) -> MultiGameCardLookup:
        """One CardLookup over games' binders, sharing the ones already
        loaded (no merged copy of any game's cards).

        Inputs: games (non-empty). Output: MultiGameCardLookup.
        Side effects: loads any game's binder not yet loaded.
        Exceptions: FileNotFoundError as card_binder; ValueError if games
            is empty.

        Example:
            >>> CardShelf().card_lookup([GameId.GWENT, GameId.MTG])
        """
        return MultiGameCardLookup({game: self.card_binder(game) for game in games})

    def deck_box(self, path: Path) -> DeckBox:
        """The DeckBox at path, loaded on first request.

        Inputs: path (Path). Output: DeckBox.
        Side effects: reads the deck box on first call per path.
        Exceptions: FileNotFoundError if path does not exist (DeckBox.load
            would silently return an empty box).

        Example:
            >>> CardShelf().deck_box(STS_GG_DECK_BOX_PATH)
        """
        if path not in self._deck_boxes:
            if not path.exists():
                raise FileNotFoundError(
                    f"no deck box at {path}; run scripts/run_metrics.py for "
                    "the metric source that writes it"
                )
            self._deck_boxes[path] = DeckBox.load([path])
        return self._deck_boxes[path]


@dataclass(frozen=True)
class DojoBuildContext:
    """What every recipe needs besides its own entry: the shared shelf and
    the run-wide settings every dojo must agree on.

    shelf: loaded card data. holdout: the plan's HoldoutSpec (the trainer
    requires every dojo to share it). card_embedding_size: the model's
    embed_dim. rng_seed: seeds each dojo's split shuffling, so a first
    split build is reproducible. mod_overrides: per dojo name,
    augmentation specs replacing that dojo's game defaults (an empty tuple
    means no augmentation); build_dojos rejects a name that is not built,
    and a deck spec on a dojo or group DECK_MOD_GROUPS does not allow.
    staple_thresholds: per contrastive dojo name, its staple-subsampling t
    (src/dojos/contrastive/staple_subsampling.py); a dojo not named keeps
    t = inf (no subsampling). build_dojos rejects a name that is not a
    contrastive dojo being built.
    """

    shelf: CardShelf
    holdout: HoldoutSpec
    card_embedding_size: int
    rng_seed: int
    mod_overrides: Mapping[str, tuple[ModSpec, ...]] = field(default_factory=dict)
    staple_thresholds: Mapping[str, float] = field(default_factory=dict)


class DojoRecipe(Protocol):
    """How to build one catalog dojo."""

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """Construct the dojo, named name. Side effects: may load card
        data through context.shelf; may write split files on first build.
        Exceptions: FileNotFoundError for missing data; ValueError for a
        stale metric (the dojo's own version check)."""
        ...


@dataclass(frozen=True)
class CardDojoRecipe:
    """A dojo that needs only its game's CardBinder. metric_output, when
    set, replaces the dojo class's own metric file."""

    game: GameId
    dojo_class: CardDojoConstructor
    metric_output: Path | None = None

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build; augmentations as _with_augmentations."""
        dojo = self.dojo_class(
            context.shelf.card_binder(self.game),
            context.holdout,
            context.card_embedding_size,
            path_to_training_data=self.metric_output,
            name=name,
            rng_seed=context.rng_seed,
        )
        return _with_augmentations(dojo, name, self.game, context)


class MultiGameCardDojoConstructor(Protocol):
    """A dojo class built from one CardLookup spanning several games (the
    cross-game rarity dojo)."""

    def __call__(
        self,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        *,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> GenericDojo: ...


@dataclass(frozen=True)
class MultiGameCardDojoRecipe:
    """A dojo over the cards of several games, through one
    MultiGameCardLookup. Augmentations are each game's own defaults, run
    per card by a PerGameMod; a mod_overrides entry for the dojo replaces
    them for every game."""

    games: tuple[GameId, ...]
    dojo_class: MultiGameCardDojoConstructor

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build.

        Inputs: name, context.
        Output: the dojo, with a PerGameMod of every game's augmentations
            appended after its own task mods.
        Side effects: loads each game's binder through context.shelf; may
            write split files.
        Exceptions: as DojoRecipe.build.
        """
        dojo = self.dojo_class(
            context.shelf.card_lookup(self.games),
            context.holdout,
            context.card_embedding_size,
            name=name,
            rng_seed=context.rng_seed,
        )

        # One augmentation pipeline per game, dispatched by each card's game
        pipelines = {
            game: _augmentation_pipeline(name, game, context) for game in self.games
        }
        if any(pipeline.mods for pipeline in pipelines.values()):
            dojo.append_mods([PerGameMod(pipelines, train_only=True)])
        return dojo


@dataclass(frozen=True)
class DeckDojoRecipe:
    """A dojo that needs its game's CardBinder and a DeckBox. metric_output,
    when set, replaces the dojo class's own metric file."""

    game: GameId
    dojo_class: DeckDojoConstructor
    deck_box_path: Path
    metric_output: Path | None = None

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build; augmentations as _with_augmentations."""
        dojo = self.dojo_class(
            context.shelf.card_binder(self.game),
            context.holdout,
            context.shelf.deck_box(self.deck_box_path),
            context.card_embedding_size,
            path_to_training_data=self.metric_output,
            name=name,
            rng_seed=context.rng_seed,
        )
        return _with_augmentations(dojo, name, self.game, context)


@dataclass(frozen=True)
class ContrastiveDojoRecipe:
    """A single-card contrastive dojo over one game's final deck box:
    positives are cards drawn from the same deck.

    game: whose binder and deck box. deck_box_path: the game's
    data/final/decks/<game>.db. items_per_deck / decks_per_sample: see
    SingleCardPairConstructor / ContrastiveDojo.
    """

    game: GameId
    deck_box_path: Path
    items_per_deck: int = 2
    decks_per_sample: int = 16

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build. The dealer keeps its split index at
        data/splits/contrastive/<name>.db (seeded by context.rng_seed);
        mods come from context.mod_overrides[name] if present, else the
        game's defaults, each built with its own seed. A
        context.staple_thresholds[name] turns on staple subsampling, its
        document frequency cached beside the split index."""
        binder = context.shelf.card_binder(self.game)
        dealer = DeckBoxDealer(
            context.shelf.deck_box(self.deck_box_path),
            self.game,
            _contrastive_index_path(name),
            seed=context.rng_seed,
        )
        pair_constructor = SingleCardPairConstructor(
            self.items_per_deck,
            rng_seed=context.rng_seed,
            staple_subsampling=_staple_subsampling(name, dealer, context),
        )
        return ContrastiveDojo(
            dealer,
            pair_constructor,
            binder,
            context.holdout,
            decks_per_sample=self.decks_per_sample,
            name=name,
            mod_pipeline=_augmentation_pipeline(name, self.game, context),
        )


def _contrastive_index_path(name: str) -> Path:
    """Where a contrastive dojo's deck split index lives.
    Inputs: name. Output: Path. Side effects: none. Exceptions: none."""
    return _CONTRASTIVE_INDEX_DIRECTORY / f"{name}.db"


def _staple_subsampling(
    name: str, dealer: DeckBoxDealer, context: DojoBuildContext
) -> StapleSubsampling | None:
    """name's staple subsampling, or None (t = inf) when the run config
    names no threshold for it. Its document frequency comes from
    cached_document_frequency, cached at
    data/splits/contrastive/<name>.document_frequency.json.

    Inputs: name, dealer (the dojo's), context.
    Output: StapleSubsampling | None.
    Side effects: may count and cache the document frequency (reads decks
        through dealer; writes only the cache file).
    Exceptions: as cached_document_frequency and StapleSubsampling.
    """
    threshold = context.staple_thresholds.get(name)
    if threshold is None:
        return None
    cache_path = _CONTRASTIVE_INDEX_DIRECTORY / f"{name}.document_frequency.json"
    frequency = cached_document_frequency(
        dealer, cache_path, _DOCUMENT_FREQUENCY_SAMPLE_DECKS
    )
    return StapleSubsampling(threshold, frequency)


def _augmentation_pipeline(
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


def _with_augmentations(
    dojo: GenericDojo, name: str, game: GameId, context: DojoBuildContext
) -> GenericDojo:
    """dojo with its augmentations (_augmentation_pipeline) appended after
    its own task mods, which stay first and in order.

    Inputs: dojo (a just-built metric dojo, no batch yet), name, game,
        context. Output: dojo itself.
    Side effects: dojo.append_mods when there is anything to append.
    Exceptions: as a ModSpec's build.
    """
    mods = _augmentation_pipeline(name, game, context).mods
    if mods:
        dojo.append_mods(mods)
    return dojo


def _recipe_for_contrastive(game: GameId) -> ContrastiveDojoRecipe:
    """The contrastive recipe over game's final deck box, at
    DeckBox.default_output_path(game).

    Inputs: game. Output: ContrastiveDojoRecipe. Side effects: none.
    Exceptions: none.
    """
    return ContrastiveDojoRecipe(game, DeckBox.default_output_path(game))


def _recipe_for_gwent_one(dojo_class: CardDojoConstructor) -> CardDojoRecipe:
    """The recipe for a gwent_one card dojo class.

    Inputs: dojo_class. Output: its recipe. Side effects: none.
    Exceptions: none.
    """
    return CardDojoRecipe(GameId.GWENT, dojo_class)


def _recipe_for_sts_card(dojo_class: CardDojoConstructor) -> CardDojoRecipe:
    """The recipe for a sts_gg card-average dojo class.

    Inputs: dojo_class. Output: its recipe. Side effects: none.
    Exceptions: none.
    """
    return CardDojoRecipe(GameId.SLAY_THE_SPIRE_2, dojo_class)


def _recipe_for_sts_deck(dojo_class: DeckDojoConstructor) -> DeckDojoRecipe:
    """The recipe for a sts_gg deck-label dojo class.

    Inputs: dojo_class. Output: its recipe. Side effects: none.
    Exceptions: none.
    """
    return DeckDojoRecipe(GameId.SLAY_THE_SPIRE_2, dojo_class, STS_GG_DECK_BOX_PATH)


def _recipe_for_final_deck_box(
    game: GameId, dojo_class: DeckDojoConstructor
) -> DeckDojoRecipe:
    """The recipe for a dojo whose metric rows point into game's
    published deck box, at DeckBox.default_output_path(game).

    Inputs: game, dojo_class. Output: DeckDojoRecipe. Side effects: none.
    Exceptions: none.
    """
    return DeckDojoRecipe(game, dojo_class, DeckBox.default_output_path(game))


def _recipe_for_sts2_runs_card(
    dojo_class: CardDojoConstructor, metric_output: Path
) -> CardDojoRecipe:
    """A sts_gg card-average dojo class over an sts2_runs metric file.

    Inputs: dojo_class, metric_output (an sts2_runs metric's
        DEFAULT_OUTPUT_PATH). Output: CardDojoRecipe.
    Side effects: none. Exceptions: none.
    """
    return CardDojoRecipe(GameId.SLAY_THE_SPIRE_2, dojo_class, metric_output)


def _recipe_for_sts2_runs_deck(
    dojo_class: DeckDojoConstructor, metric_output: Path
) -> DeckDojoRecipe:
    """A sts_gg deck-label dojo class over an sts2_runs metric file, whose
    deck_uuids point into the published StS2 deck box (see
    metrics/sts2_runs/deck_label_metric.py).

    Inputs: dojo_class, metric_output (an sts2_runs metric's
        DEFAULT_OUTPUT_PATH). Output: DeckDojoRecipe.
    Side effects: none. Exceptions: none.
    """
    return DeckDojoRecipe(
        GameId.SLAY_THE_SPIRE_2,
        dojo_class,
        DeckBox.default_output_path(GameId.SLAY_THE_SPIRE_2),
        metric_output,
    )


def _recipe_for_isotropic_deck(dojo_class: DeckDojoConstructor) -> DeckDojoRecipe:
    """The recipe for an isotropic deck-level dojo class.
    Inputs: dojo_class. Output: DeckDojoRecipe. Side effects: none.
    Exceptions: none."""
    return DeckDojoRecipe(GameId.DOMINION, dojo_class, ISOTROPIC_DECK_BOX_PATH)


def _recipe_for_seventeenlands_card(dojo_class: CardDojoConstructor) -> CardDojoRecipe:
    """A 17lands dojo class that needs no deck box, trained on its
    metric's all-sets, all-formats slice (the dojo's default data_slice).

    Inputs: dojo_class. Output: CardDojoRecipe. Side effects: none.
    Exceptions: none.
    """
    return CardDojoRecipe(GameId.MTG, dojo_class)


def _recipe_for_seventeenlands_replay_deck(
    dojo_class: DeckDojoConstructor,
) -> DeckDojoRecipe:
    """A replay_data deck-level dojo class over replay_data's private
    deck box, trained on the all-sets, all-formats slice. (game_data deck
    dojos read the canonical MTG box: _recipe_for_final_deck_box.)

    Inputs: dojo_class. Output: DeckDojoRecipe. Side effects: none.
    Exceptions: none.
    """
    return DeckDojoRecipe(GameId.MTG, dojo_class, REPLAY_DATA_DECK_BOX_PATH)


# Keys are "<metric source>.<metric file stem>"; add a line to onboard a dojo
DOJO_CATALOG: Mapping[str, DojoRecipe] = {
    "gwent_one.armor_mask": _recipe_for_gwent_one(gwent_one.ArmorMaskDojo),
    "gwent_one.color_mask": _recipe_for_gwent_one(gwent_one.ColorMaskDojo),
    "gwent_one.faction_mask": _recipe_for_gwent_one(gwent_one.FactionMaskDojo),
    "gwent_one.power_mask": _recipe_for_gwent_one(gwent_one.PowerMaskDojo),
    "gwent_one.provision_mask": _recipe_for_gwent_one(gwent_one.ProvisionMaskDojo),
    "gwent_one.rarity_mask": _recipe_for_gwent_one(gwent_one.RarityMaskDojo),
    "gwent_one.set_mask": _recipe_for_gwent_one(gwent_one.SetMaskDojo),
    "gwent_one.type_mask": _recipe_for_gwent_one(gwent_one.TypeMaskDojo),
    "dominiontabs.cost_regression": CardDojoRecipe(GameId.DOMINION, CostRegressionDojo),
    "dominiontabs.set_mask": CardDojoRecipe(GameId.DOMINION, SetMaskDojo),
    "dominiontabs.type_mask": CardDojoRecipe(GameId.DOMINION, TypeMaskDojo),
    "scryfall.cmc_regression": CardDojoRecipe(
        GameId.MTG, scryfall_masks.CmcRegressionDojo
    ),
    "scryfall.card_type_mask": CardDojoRecipe(
        GameId.MTG, scryfall_masks.CardTypeMaskDojo
    ),
    "scryfall.rarity_mask": CardDojoRecipe(GameId.MTG, scryfall_masks.RarityMaskDojo),
    "scryfall.colors_mask": CardDojoRecipe(GameId.MTG, scryfall_masks.ColorsMaskDojo),
    "scryfall.power_regression": CardDojoRecipe(
        GameId.MTG, scryfall_masks.PowerRegressionDojo
    ),
    "scryfall.toughness_regression": CardDojoRecipe(
        GameId.MTG, scryfall_masks.ToughnessRegressionDojo
    ),
    "pokemon_tcg.hp_regression": CardDojoRecipe(
        GameId.POKEMON, pokemon_masks.HpRegressionDojo
    ),
    "pokemon_tcg.types_mask": CardDojoRecipe(
        GameId.POKEMON, pokemon_masks.TypesMaskDojo
    ),
    "pokemon_tcg.stage_mask": CardDojoRecipe(
        GameId.POKEMON, pokemon_masks.StageMaskDojo
    ),
    "pokemon_tcg.retreat_cost_regression": CardDojoRecipe(
        GameId.POKEMON, pokemon_masks.RetreatCostRegressionDojo
    ),
    "pokemon_tcg.weakness_mask": CardDojoRecipe(
        GameId.POKEMON, pokemon_masks.WeaknessMaskDojo
    ),
    "cardvault_fabtcg.pitch_mask": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_masks.PitchMaskDojo
    ),
    "cardvault_fabtcg.cost_regression": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_masks.CostRegressionDojo
    ),
    "cardvault_fabtcg.power_regression": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_masks.PowerRegressionDojo
    ),
    "cardvault_fabtcg.defense_regression": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_masks.DefenseRegressionDojo
    ),
    "cardvault_fabtcg.class_mask": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_masks.ClassMaskDojo
    ),
    "cardvault_fabtcg.card_type_mask": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_masks.CardTypeMaskDojo
    ),
    "spire_codex.cost_mask": CardDojoRecipe(
        GameId.SLAY_THE_SPIRE_2, sts2_masks.CostMaskDojo
    ),
    "spire_codex.card_type_mask": CardDojoRecipe(
        GameId.SLAY_THE_SPIRE_2, sts2_masks.CardTypeMaskDojo
    ),
    "spire_codex.rarity_mask": CardDojoRecipe(
        GameId.SLAY_THE_SPIRE_2, sts2_masks.RarityMaskDojo
    ),
    "spire_codex.color_mask": CardDojoRecipe(
        GameId.SLAY_THE_SPIRE_2, sts2_masks.ColorMaskDojo
    ),
    "hearthstonejson.cost_regression": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.CostRegressionDojo
    ),
    "hearthstonejson.attack_regression": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.AttackRegressionDojo
    ),
    "hearthstonejson.health_regression": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.HealthRegressionDojo
    ),
    "hearthstonejson.class_mask": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.ClassMaskDojo
    ),
    "hearthstonejson.rarity_mask": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.RarityMaskDojo
    ),
    "hearthstonejson.card_type_mask": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.CardTypeMaskDojo
    ),
    "hearthstonejson.races_mask": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.RacesMaskDojo
    ),
    "hearthstonejson.spell_school_mask": CardDojoRecipe(
        GameId.HEARTHSTONE, hearthstone_masks.SpellSchoolMaskDojo
    ),
    "isotropic.average_copies_bought": CardDojoRecipe(
        GameId.DOMINION, isotropic.AverageCopiesBoughtDojo
    ),
    "isotropic.copies_bought_distribution": CardDojoRecipe(
        GameId.DOMINION, isotropic.CopiesBoughtDistributionDojo
    ),
    "isotropic.opening_buy_rate": CardDojoRecipe(
        GameId.DOMINION, isotropic.OpeningBuyRateDojo
    ),
    "isotropic.pile_exhaustion_rate": CardDojoRecipe(
        GameId.DOMINION, isotropic.PileExhaustionRateDojo
    ),
    "isotropic.turn_count_association": CardDojoRecipe(
        GameId.DOMINION, isotropic.TurnCountAssociationDojo
    ),
    "isotropic.full_deck_win_prediction": _recipe_for_isotropic_deck(
        isotropic.FullDeckWinPredictionDojo
    ),
    "isotropic.kingdom_ending_type": _recipe_for_isotropic_deck(
        isotropic.KingdomEndingTypeDojo
    ),
    "isotropic.kingdom_game_length": _recipe_for_isotropic_deck(
        isotropic.KingdomGameLengthDojo
    ),
    "isotropic.mid_game_next_turn_action_count": _recipe_for_isotropic_deck(
        isotropic.NextTurnActionCountDojo
    ),
    "isotropic.winning_deck_masked_card": _recipe_for_isotropic_deck(
        isotropic.WinningDeckMaskedCardDojo
    ),
    "isotropic.veto_rate": CardDojoRecipe(GameId.DOMINION, isotropic.VetoRateDojo),
    "isotropic.kingdom_opening_buy_prediction": _recipe_for_isotropic_deck(
        isotropic.KingdomOpeningBuyDojo
    ),
    "isotropic.kingdom_veto_prediction": _recipe_for_isotropic_deck(
        isotropic.KingdomVetoDojo
    ),
    "isotropic.mid_game_next_buy": _recipe_for_isotropic_deck(isotropic.NextBuyDojo),
    "isotropic.mid_game_next_trashed_card": _recipe_for_isotropic_deck(
        isotropic.NextTrashedCardDojo
    ),
    "isotropic.deck_pair_winner": _recipe_for_isotropic_deck(
        isotropic.DeckPairWinnerDojo
    ),
    "isotropic.mid_game_deck_pair_winner": _recipe_for_isotropic_deck(
        isotropic.MidGameDeckPairWinnerDojo
    ),
    "isotropic.mid_game_win_probability": _recipe_for_isotropic_deck(
        isotropic.EventualWinDojo
    ),
    "isotropic.opening_buy_outcome": _recipe_for_isotropic_deck(
        isotropic.OpeningBuyOutcomeDojo
    ),
    "isotropic.winning_deck_membership": _recipe_for_isotropic_deck(
        isotropic.WinningDeckMembershipDojo
    ),
    "isotropic.kingdom_ending_pile_prediction": _recipe_for_isotropic_deck(
        isotropic.KingdomEndingPileDojo
    ),
    "isotropic.winning_deck_count": _recipe_for_isotropic_deck(
        isotropic.WinningDeckCountDojo
    ),
    "isotropic.deck_card_set_copy_count": _recipe_for_isotropic_deck(
        isotropic.DeckCardSetCopyCountDojo
    ),
    # play_gwent writes its decks into the final gwent deck box
    "play_gwent.leader_masked_from_deck": DeckDojoRecipe(
        GameId.GWENT,
        LeaderMaskedFromDeckDojo,
        DeckBox.default_output_path(GameId.GWENT),
    ),
    # One rarity ladder shared by six games, trained with one head
    "cross_game.rarity_tier": MultiGameCardDojoRecipe(
        tuple(RARITY_TRANSLATORS), RarityTierDojo
    ),
    # [cards offered, deck so far] -> the card taken, or none
    "sts2_runs.card_reward_pick": _recipe_for_sts_card(CardRewardPickDojo),
    "sts2_runs.shop_purchase_pick": _recipe_for_sts_card(ShopPurchasePickDojo),
    "sts2_runs.card_removal_pick": _recipe_for_sts_card(CardRemovalPickDojo),
    "sts2_runs.card_upgrade_pick": _recipe_for_sts_card(CardUpgradePickDojo),
    # sts_gg lists winning runs only, so its win, card_win_rate,
    # card_win_rate_at_act2 and killed_by labels are constant and have no
    # key here (see metrics/sts_gg/deck_label_metrics.py's WINS ONLY note)
    "sts_gg.ascension_prediction": _recipe_for_sts_deck(sts_decks.DeckAscensionDojo),
    "sts_gg.card_character_prediction": _recipe_for_sts_card(
        CardCharacterPredictionDojo
    ),
    "sts_gg.card_deck_size": _recipe_for_sts_card(sts_cards.CardDeckSizeDojo),
    "sts_gg.card_elites_killed": _recipe_for_sts_card(sts_cards.CardElitesKilledDojo),
    "sts_gg.card_floors_cleared": _recipe_for_sts_card(sts_cards.CardFloorsClearedDojo),
    "sts_gg.card_relic_count": _recipe_for_sts_card(sts_cards.CardRelicCountDojo),
    "sts_gg.card_total_cards_picked": _recipe_for_sts_card(
        sts_cards.CardTotalCardsPickedDojo
    ),
    "sts_gg.card_total_combats": _recipe_for_sts_card(sts_cards.CardTotalCombatsDojo),
    "sts_gg.card_total_damage_taken": _recipe_for_sts_card(
        sts_cards.CardTotalDamageTakenDojo
    ),
    "sts_gg.card_total_turns": _recipe_for_sts_card(sts_cards.CardTotalTurnsDojo),
    "sts_gg.card_upgrade_rate": _recipe_for_sts_card(sts_cards.CardUpgradeRateDojo),
    "sts_gg.character_prediction": _recipe_for_sts_deck(sts_decks.CharacterDojo),
    "sts_gg.elites_killed": _recipe_for_sts_deck(sts_decks.DeckElitesKilledDojo),
    "sts_gg.floors_cleared": _recipe_for_sts_deck(sts_decks.DeckFloorsClearedDojo),
    "sts_gg.relic_count": _recipe_for_sts_deck(sts_decks.DeckRelicCountDojo),
    "sts_gg.total_cards_picked": _recipe_for_sts_deck(
        sts_decks.DeckTotalCardsPickedDojo
    ),
    "sts_gg.total_cards_skipped": _recipe_for_sts_deck(
        sts_decks.DeckTotalCardsSkippedDojo
    ),
    "sts_gg.total_combats": _recipe_for_sts_deck(sts_decks.DeckTotalCombatsDojo),
    "sts_gg.total_damage_taken": _recipe_for_sts_deck(
        sts_decks.DeckTotalDamageTakenDojo
    ),
    "sts_gg.total_turns": _recipe_for_sts_deck(sts_decks.DeckTotalTurnsDojo),
    "final_decks.held_out_card_pokemon": _recipe_for_final_deck_box(
        GameId.POKEMON, final_decks.PokemonHeldOutCardDojo
    ),
    "final_decks.held_out_card_flesh_and_blood": _recipe_for_final_deck_box(
        GameId.FLESH_AND_BLOOD, final_decks.FleshAndBloodHeldOutCardDojo
    ),
    "final_decks.held_out_card_gwent": _recipe_for_final_deck_box(
        GameId.GWENT, final_decks.GwentHeldOutCardDojo
    ),
    "final_decks.held_out_card_dominion": _recipe_for_final_deck_box(
        GameId.DOMINION, final_decks.DominionHeldOutCardDojo
    ),
    "final_decks.held_out_card_slay_the_spire_2": _recipe_for_final_deck_box(
        GameId.SLAY_THE_SPIRE_2, final_decks.SlayTheSpire2HeldOutCardDojo
    ),
    "final_decks.held_out_card_mtg": _recipe_for_final_deck_box(
        GameId.MTG, final_decks.MtgHeldOutCardDojo
    ),
    # spire_codex + sts2runs runs, through the sts_gg wrappers (see the
    # module docstring)
    "sts2_runs.ascension_prediction": _recipe_for_sts2_runs_deck(
        sts_decks.DeckAscensionDojo,
        sts2_decks.AscensionPredictionMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_deck_size": _recipe_for_sts2_runs_card(
        sts_cards.CardDeckSizeDojo, sts2_cards.CardDeckSizeMetric.DEFAULT_OUTPUT_PATH
    ),
    "sts2_runs.card_elites_killed": _recipe_for_sts2_runs_card(
        sts_cards.CardElitesKilledDojo,
        sts2_cards.CardElitesKilledMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_floors_cleared": _recipe_for_sts2_runs_card(
        sts_cards.CardFloorsClearedDojo,
        sts2_cards.CardFloorsClearedMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_relic_count": _recipe_for_sts2_runs_card(
        sts_cards.CardRelicCountDojo,
        sts2_cards.CardRelicCountMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_total_cards_picked": _recipe_for_sts2_runs_card(
        sts_cards.CardTotalCardsPickedDojo,
        sts2_cards.CardTotalCardsPickedMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_total_combats": _recipe_for_sts2_runs_card(
        sts_cards.CardTotalCombatsDojo,
        sts2_cards.CardTotalCombatsMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_total_damage_taken": _recipe_for_sts2_runs_card(
        sts_cards.CardTotalDamageTakenDojo,
        sts2_cards.CardTotalDamageTakenMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_total_turns": _recipe_for_sts2_runs_card(
        sts_cards.CardTotalTurnsDojo,
        sts2_cards.CardTotalTurnsMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_upgrade_rate": _recipe_for_sts2_runs_card(
        sts_cards.CardUpgradeRateDojo,
        sts2_cards.CardUpgradeRateMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.card_win_rate": _recipe_for_sts2_runs_card(
        sts_cards.CardWinRateDojo, sts2_cards.CardWinRateMetric.DEFAULT_OUTPUT_PATH
    ),
    "sts2_runs.card_win_rate_at_act2": _recipe_for_sts2_runs_card(
        sts_cards.CardWinRateAtAct2Dojo,
        sts2_cards.CardWinRateAtAct2Metric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.character_prediction": _recipe_for_sts2_runs_deck(
        sts_decks.CharacterDojo,
        sts2_decks.CharacterPredictionMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.elites_killed": _recipe_for_sts2_runs_deck(
        sts_decks.DeckElitesKilledDojo,
        sts2_decks.ElitesKilledMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.floors_cleared": _recipe_for_sts2_runs_deck(
        sts_decks.DeckFloorsClearedDojo,
        sts2_decks.FloorsClearedMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.killed_by": _recipe_for_sts2_runs_deck(
        sts_decks.KilledByDojo, sts2_decks.KilledByMetric.DEFAULT_OUTPUT_PATH
    ),
    "sts2_runs.relic_count": _recipe_for_sts2_runs_deck(
        sts_decks.DeckRelicCountDojo, sts2_decks.RelicCountMetric.DEFAULT_OUTPUT_PATH
    ),
    "sts2_runs.total_cards_picked": _recipe_for_sts2_runs_deck(
        sts_decks.DeckTotalCardsPickedDojo,
        sts2_decks.TotalCardsPickedMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.total_cards_skipped": _recipe_for_sts2_runs_deck(
        sts_decks.DeckTotalCardsSkippedDojo,
        sts2_decks.TotalCardsSkippedMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.total_combats": _recipe_for_sts2_runs_deck(
        sts_decks.DeckTotalCombatsDojo,
        sts2_decks.TotalCombatsMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.total_damage_taken": _recipe_for_sts2_runs_deck(
        sts_decks.DeckTotalDamageTakenDojo,
        sts2_decks.TotalDamageTakenMetric.DEFAULT_OUTPUT_PATH,
    ),
    "sts2_runs.total_turns": _recipe_for_sts2_runs_deck(
        sts_decks.DeckTotalTurnsDojo, sts2_decks.TotalTurnsMetric.DEFAULT_OUTPUT_PATH
    ),
    "sts2_runs.win": _recipe_for_sts2_runs_deck(
        sts_decks.WinDojo, sts2_decks.WinMetric.DEFAULT_OUTPUT_PATH
    ),
    # fabtcg_decklists rows point into the published FaB deck box
    "fabtcg_decklists.hero_masked_from_deck": _recipe_for_final_deck_box(
        GameId.FLESH_AND_BLOOD, fabtcg_decks.HeroMaskedFromDeckDojo
    ),
    "fabtcg_decklists.card_inclusion_rate": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_inclusion.CardInclusionRateDojo
    ),
    "fabtcg_decklists.hero_conditioned_inclusion": CardDojoRecipe(
        GameId.FLESH_AND_BLOOD, fabtcg_inclusion.HeroConditionedInclusionDojo
    ),
    # play_gwent guide metrics read the published Gwent box, never write it
    "play_gwent.card_inclusion_rate": CardDojoRecipe(
        GameId.GWENT, gwent_inclusion.CardInclusionRateDojo
    ),
    "play_gwent.faction_conditioned_inclusion": CardDojoRecipe(
        GameId.GWENT, gwent_inclusion.FactionConditionedInclusionDojo
    ),
    "play_gwent.guide_votes": _recipe_for_final_deck_box(GameId.GWENT, GuideVotesDojo),
    # 17lands: each key trains on its metric's all-sets, all-formats slice
    # (data/metrics/seventeenlands/<family>/slices/<stem>.all.parquet);
    # game_data deck rows point into the canonical MTG deck box
    "seventeenlands_draft_data.card_take_rate": _recipe_for_seventeenlands_card(
        draft_tally.CardTakeRateDojo
    ),
    "seventeenlands_draft_data.first_pick_rate": _recipe_for_seventeenlands_card(
        draft_tally.FirstPickRateDojo
    ),
    "seventeenlands_draft_data.rank_stratified_take_rate": (
        _recipe_for_seventeenlands_card(draft_tally.RankStratifiedTakeRateDojo)
    ),
    "seventeenlands_draft_data.pick_number_decay_curve": (
        _recipe_for_seventeenlands_card(PickNumberDecayCurveDojo)
    ),
    "seventeenlands_draft_data.pack_to_pick_choice_set": (
        _recipe_for_seventeenlands_card(PackToPickChoiceSetDojo)
    ),
    "seventeenlands_draft_data.pool_conditioned_pick": (
        _recipe_for_seventeenlands_card(PoolConditionedPickDojo)
    ),
    "seventeenlands_game_data.win_rate_when_in_deck": (
        _recipe_for_seventeenlands_card(game_cards.WinRateWhenInDeckDojo)
    ),
    "seventeenlands_game_data.opening_hand_win_rate": (
        _recipe_for_seventeenlands_card(game_cards.OpeningHandWinRateDojo)
    ),
    "seventeenlands_game_data.drawn_win_rate": _recipe_for_seventeenlands_card(
        game_cards.DrawnWinRateDojo
    ),
    "seventeenlands_game_data.game_length_association": (
        _recipe_for_seventeenlands_card(GameLengthAssociationDojo)
    ),
    "seventeenlands_game_data.on_play_win_rate_delta": (
        _recipe_for_seventeenlands_card(OnPlayWinRateDeltaDojo)
    ),
    "seventeenlands_game_data.tutor_target_rate": _recipe_for_seventeenlands_card(
        GameTutorTargetRateDojo
    ),
    "seventeenlands_game_data.tutor_choice_rate": _recipe_for_seventeenlands_card(
        TutorChoiceRateDojo
    ),
    "seventeenlands_game_data.deck_win_prediction": _recipe_for_final_deck_box(
        GameId.MTG, game_decks.DeckWinPredictionDojo
    ),
    "seventeenlands_game_data.deck_game_length_prediction": (
        _recipe_for_final_deck_box(GameId.MTG, game_decks.DeckGameLengthPredictionDojo)
    ),
    "seventeenlands_game_data.deck_rank_tier_prediction": (
        _recipe_for_final_deck_box(GameId.MTG, game_decks.DeckRankTierPredictionDojo)
    ),
    "seventeenlands_game_data.on_play_win_rate_sensitivity_by_deck": (
        _recipe_for_final_deck_box(GameId.MTG, OnPlayWinRateSensitivityByDeckDojo)
    ),
    "seventeenlands_replay_data.average_turn_cast": _recipe_for_seventeenlands_card(
        AverageTurnCastDojo
    ),
    "seventeenlands_replay_data.cast_rate": _recipe_for_seventeenlands_card(
        CastRateDojo
    ),
    "seventeenlands_replay_data.turns_to_game_end_after_cast": (
        _recipe_for_seventeenlands_card(TurnsToGameEndAfterCastDojo)
    ),
    "seventeenlands_replay_data.discard_rate": _recipe_for_seventeenlands_card(
        DiscardRateDojo
    ),
    "seventeenlands_replay_data.tutor_target_rate": _recipe_for_seventeenlands_card(
        ReplayTutorTargetRateDojo
    ),
    "seventeenlands_replay_data.combat_kill_involvement_rate": (
        _recipe_for_seventeenlands_card(replay_combat.CombatKillInvolvementRateDojo)
    ),
    "seventeenlands_replay_data.combat_damage_push_through_rate": (
        _recipe_for_seventeenlands_card(replay_combat.CombatDamagePushThroughRateDojo)
    ),
    "seventeenlands_replay_data.combat_aggression_profile": (
        _recipe_for_seventeenlands_replay_deck(CombatAggressionProfileDojo)
    ),
    "seventeenlands_replay_data.attacker_blocker_combat_outcome": (
        _recipe_for_seventeenlands_card(AttackerBlockerCombatOutcomeDojo)
    ),
    "contrastive.gwent": _recipe_for_contrastive(GameId.GWENT),
    "contrastive.flesh_and_blood": _recipe_for_contrastive(GameId.FLESH_AND_BLOOD),
    "contrastive.slay_the_spire_2": _recipe_for_contrastive(GameId.SLAY_THE_SPIRE_2),
    "contrastive.mtg": _recipe_for_contrastive(GameId.MTG),
    "contrastive.pokemon": _recipe_for_contrastive(GameId.POKEMON),
    "contrastive.dominion": _recipe_for_contrastive(GameId.DOMINION),
}


# Opt-in deck mods (src/dojos/mods/deck_mods.py): the multi-card metric
# dojos a run config may thin, and which input groups (a multi-card input
# is group 0). Data knowledge, and the only gate: a dojo not listed takes
# no deck mods. Never listed: a label that is deck size, a copy count or a
# run-length total that tracks deck size (sts total_*, floors, elites,
# combats, relic count, card_deck_size, isotropic copy counts); a held-out
# card dojo (thinning changes the answer set); an option-selection dojo's
# options group (the label indexes it), so a MultiCardOptionSelection dojo
# (options are its whole input) is never listed; a kingdom or single-card
# group (fixed game context, not a deck); a group whose label depends on
# exactly which cards it holds (17lands attacker/blocker combat outcome).
DECK_MOD_GROUPS: Mapping[str, frozenset[int]] = {
    "sts_gg.ascension_prediction": frozenset({0}),
    "sts_gg.character_prediction": frozenset({0}),
    "sts2_runs.ascension_prediction": frozenset({0}),
    "sts2_runs.character_prediction": frozenset({0}),
    "sts2_runs.killed_by": frozenset({0}),
    "sts2_runs.win": frozenset({0}),
    "play_gwent.guide_votes": frozenset({0}),
    "play_gwent.leader_masked_from_deck": frozenset({0}),
    "fabtcg_decklists.hero_masked_from_deck": frozenset({0}),
    "isotropic.full_deck_win_prediction": frozenset({0}),
    # [partial deck, kingdom]: the partial deck only
    "isotropic.mid_game_win_probability": frozenset({0}),
    # Both decks of a symmetric pair
    "isotropic.deck_pair_winner": frozenset({0, 1}),
    "isotropic.mid_game_deck_pair_winner": frozenset({0, 1}),
    # [options, partial deck]: the options group (0) is never thinned
    "isotropic.mid_game_next_buy": frozenset({1}),
    # 17lands game decks: 40-card limited decks; no label tracks deck size
    "seventeenlands_game_data.deck_win_prediction": frozenset({0}),
    "seventeenlands_game_data.deck_game_length_prediction": frozenset({0}),
    "seventeenlands_game_data.deck_rank_tier_prediction": frozenset({0}),
    "seventeenlands_game_data.on_play_win_rate_sensitivity_by_deck": frozenset({0}),
    "seventeenlands_replay_data.combat_aggression_profile": frozenset({0}),
    # [pack options, pool so far]: the pool only
    "seventeenlands_draft_data.pool_conditioned_pick": frozenset({1}),
    # [cards offered, deck so far]: the deck only
    "sts2_runs.card_reward_pick": frozenset({1}),
    "sts2_runs.shop_purchase_pick": frozenset({1}),
    "sts2_runs.card_removal_pick": frozenset({1}),
    "sts2_runs.card_upgrade_pick": frozenset({1}),
}


def build_dojos(names: Sequence[str], context: DojoBuildContext) -> list[Dojo]:
    """Build every named catalog dojo, in order.

    Inputs: names (Sequence[str], DOJO_CATALOG keys), context
        (DojoBuildContext).
    Output: list[Dojo], same order as names, each named by its key.
    Side effects: loads card data through context.shelf; a dojo whose
        split files do not exist yet writes them (seeded by
        context.rng_seed).
    Exceptions: ValueError naming every unknown key, any
        context.mod_overrides name that is not in names, or a deck spec
        DECK_MOD_GROUPS does not allow, raised before any dojo is built
        (so a typo does not cost a binder load); whatever a recipe's
        build raises.

    Example:
        >>> context = DojoBuildContext(CardShelf(), holdout, 256, rng_seed=0)
        >>> build_dojos(["gwent_one.color_mask"], context)[0].name
        'gwent_one.color_mask'
    """
    result: list[Dojo] = []

    # Validate inputs: every name is in the catalog
    unknown = [name for name in names if name not in DOJO_CATALOG]
    if unknown:
        raise ValueError(f"unknown dojos {unknown}; see DOJO_CATALOG")
    _require_valid_overrides(context.mod_overrides, names)
    _require_contrastive_thresholds(context.staple_thresholds, names)

    # Build each dojo from its recipe (each attaches its own augmentations)
    for name in names:
        result.append(DOJO_CATALOG[name].build(name, context))
    return result


def _require_valid_overrides(
    mod_overrides: Mapping[str, tuple[ModSpec, ...]], names: Sequence[str]
) -> None:
    """Inputs: mod_overrides, names (the dojos being built). Output: none.
    Side effects: none.
    Exceptions: ValueError if an override names a dojo not being built
        (it would be ignored), or as _require_allowed_specs.
    """
    not_built = sorted(set(mod_overrides) - set(names))
    if not_built:
        raise ValueError(f"mods override dojos {not_built} that are not being built")
    for name, specs in mod_overrides.items():
        _require_allowed_specs(name, specs)


def _require_contrastive_thresholds(
    staple_thresholds: Mapping[str, float], names: Sequence[str]
) -> None:
    """Inputs: staple_thresholds, names (the dojos being built).
    Output: none. Side effects: none.
    Exceptions: ValueError if a threshold names a dojo not being built or
        one that is not contrastive (it would be ignored).
    """
    refused = sorted(
        name
        for name in staple_thresholds
        if name not in names
        or not isinstance(DOJO_CATALOG[name], ContrastiveDojoRecipe)
    )
    if refused:
        raise ValueError(
            f"staple_subsampling names {refused}, which are not contrastive "
            "dojos being built"
        )


def _require_allowed_specs(name: str, specs: tuple[ModSpec, ...]) -> None:
    """Card-field specs suit every dojo; a deck spec needs a dojo in
    DECK_MOD_GROUPS and only the groups listed there.

    Inputs: name (a catalog dojo), specs (its override). Output: none.
    Side effects: none.
    Exceptions: ValueError for a deck spec on a dojo not in
        DECK_MOD_GROUPS, or naming a group it does not allow.
    """
    for spec in specs:
        if not isinstance(spec, DeckModSpec):
            continue
        spec_name = type(spec).__name__
        if name not in DECK_MOD_GROUPS:
            raise ValueError(
                f"{name} takes no deck mods ({spec_name}); see DECK_MOD_GROUPS"
            )
        allowed = DECK_MOD_GROUPS[name]
        refused = sorted(set(spec.groups) - allowed)
        if refused:
            raise ValueError(
                f"{name} may thin groups {sorted(allowed)} only "
                f"(DECK_MOD_GROUPS); {spec_name} names {refused}"
            )
