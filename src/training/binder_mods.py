"""Binder mods: one mod pipeline applied to every card of a game before
training, optionally saved and reused by later runs.

modded_binder(game, spec) gives a ModdedCardBinder. Its cards went
through spec.pipeline (all but the game's Unknown sentinel), and its
version_for(game) answers the clean binder's version. Card snapshots,
metric dojos and the contrastive dojo all check version_for against
versions stamped from the clean binder, and their uuids, labels and
deck boxes were computed from clean cards, so they accept it unchanged.

With spec.save_directory set, each game is saved as <game>.jsonl (plus
its alias ledger) and <game>.stamp.json. The stamp is moved into place
last and is what marks a saved bake as complete: a later run whose spec
matches the stamp loads the file instead of loading the clean binder and
applying the pipeline again. A stamp made by a different pipeline or
seed aborts the run (BinderModsMismatchError). The stamp does not cover
mod code or later re-ingests of the clean binder: after changing a mod's
code or re-ingesting, delete the directory (a re-ingest otherwise shows
up as a StaleSnapshotError naming the stamp's old source version).

ORDER: binder mods run before every dojo mod and before label
extraction. A binder mod that renames or moves keys (key_case,
flatten_keys) stops a dojo's field masks from matching; see the
ordering note in src/dojos/mods/mod_pipeline.py.
"""

import dataclasses
import hashlib
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.card import GenericCard
from src.schema.game_id import GameId
from src.training.binder_mods_spec import BinderModsSpec

_logger = logging.getLogger(__name__)

# Bump when the saved file or stamp layout changes
BINDER_MODS_FORMAT_VERSION = 1
_STAMP_NAME = "{game}.stamp.json"
_BINDER_NAME = "{game}.jsonl"


class BinderModsMismatchError(ValueError):
    """A saved bake exists at the save directory but was made by a
    different pipeline, seed or format version."""


class ModdedCardBinder(CardBinder):
    """One game's cards after a binder mod pipeline.

    version_for(game) answers the version of the clean binder the cards
    were made from, not a hash of this binder's own content. Built only
    through baked_from_clean_file or from_saved_bake, so it never holds
    unmodded cards. Training never mutates it; a mutation would leave
    version_for answering the clean version. (The inherited static
    CardBinder.load still returns a plain CardBinder.)
    """

    def __init__(self, game: GameId, clean_version: str) -> None:
        """Construct empty; use the classmethods, which fill it.

        Inputs: game (the one game it holds), clean_version (the clean
            binder's version_for(game)). Output: none.
        Side effects: none. Exceptions: none.
        """
        super().__init__()
        self._game = game
        self._clean_version = clean_version

    @classmethod
    def baked_from_clean_file(
        cls, path: Path, game: GameId, spec: BinderModsSpec
    ) -> "ModdedCardBinder":
        """game's clean binder file with spec.pipeline applied to every
        card but the Unknown sentinel, in nocab_uuid order so a seeded
        pipeline gives the same binder every time.

        Inputs: path (a clean <game>.jsonl), game, spec.
        Output: ModdedCardBinder.
        Side effects: reads path and its alias ledger.
        Exceptions: as CardBinder.load; TypeError, or whatever a mod
            raises, as _modded_card.

        Example:
            >>> ModdedCardBinder.baked_from_clean_file(clean_path, GameId.GWENT, spec)
        """
        # Load the clean cards and record their version (CardBinder's own
        # hash, read before any change) as the clean version
        result = cls(game, clean_version="")
        result._add_files([path])
        result._clean_version = CardBinder.version_for(result, game)

        # Mod each card; the sentinel is one fixed placeholder card that
        # every consumer treats as the same well-known card
        pipeline = _built_pipeline(spec)
        unknown_uuid = CardBinder.unknown_card_uuid(game)
        for card in sorted(result.all_cards(game), key=lambda card: card.nocab_uuid):
            if card.nocab_uuid == unknown_uuid:
                continue
            result.replace(card.nocab_uuid, _modded_card(pipeline, card))
        return result

    @classmethod
    def from_saved_bake(
        cls, path: Path, game: GameId, clean_version: str
    ) -> "ModdedCardBinder":
        """A saved bake loaded back, answering clean_version for game.

        Inputs: path (a saved <game>.jsonl), game, clean_version (from
            its stamp). Output: ModdedCardBinder.
        Side effects: reads path and its alias ledger.
        Exceptions: as CardBinder.load.

        Example:
            >>> ModdedCardBinder.from_saved_bake(
            ...     Path("bake/gwent.jsonl"), GameId.GWENT, stamp.source_binder_version
            ... )
        """
        result = cls(game, clean_version)
        result._add_files([path])
        return result

    def version_for(self, source_game: GameId) -> str:
        """The clean binder's version for the held game; for another game,
        CardBinder's hash (the empty digest: it holds no cards of it).

        Inputs: source_game. Output: sha256 hex digest.
        Side effects: none. Exceptions: none.

        Example:
            >>> binder = modded_binder(GameId.GWENT, spec)
            >>> binder.version_for(GameId.GWENT) == clean.version_for(GameId.GWENT)
            True
        """
        if source_game == self._game:
            return self._clean_version
        return super().version_for(source_game)


@dataclass(frozen=True)
class _BinderModsStamp:
    """What one saved game bake was made from (<game>.stamp.json).

    format_version: BINDER_MODS_FORMAT_VERSION when written.
    source_binder_version: the clean binder's version_for(game).
    mods_fingerprint: sha256 of repr(spec.pipeline) when written.
    seed: spec.seed when written.
    """

    format_version: int
    source_binder_version: str
    mods_fingerprint: str
    seed: int

    @classmethod
    def made_by(
        cls, spec: BinderModsSpec, source_binder_version: str
    ) -> "_BinderModsStamp":
        """The stamp a bake of spec from that clean version carries; the
        one definition used both to write a stamp and to check one.

        Inputs: spec, source_binder_version. Output: _BinderModsStamp.
        Side effects: none. Exceptions: none.
        """
        return cls(
            format_version=BINDER_MODS_FORMAT_VERSION,
            source_binder_version=source_binder_version,
            mods_fingerprint=_mods_fingerprint(spec),
            seed=spec.seed,
        )

    @classmethod
    def read(cls, path: Path) -> "_BinderModsStamp":
        """Inputs: path. Output: the stamp. Side effects: reads path.
        Exceptions: OSError; ValueError for a malformed file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            return cls(
                format_version=int(raw["format_version"]),
                source_binder_version=str(raw["source_binder_version"]),
                mods_fingerprint=str(raw["mods_fingerprint"]),
                seed=int(raw["seed"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed binder mods stamp {path}: {error}") from error

    def write(self, path: Path) -> None:
        """Inputs: path. Output: none. Side effects: writes path (JSON).
        Exceptions: OSError."""
        path.write_text(
            json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8"
        )


def modded_binder(game: GameId, spec: BinderModsSpec) -> ModdedCardBinder:
    """game's cards with spec.pipeline applied: the saved bake when spec's
    save directory holds one whose stamp matches spec, otherwise the clean
    binder modded now (and saved, when spec.save_directory is set).

    Inputs: game; spec.
    Output: ModdedCardBinder holding game's modded cards.
    Side effects: reads the saved bake or the clean binder file; may write
        <game>.jsonl, its alias ledger and <game>.stamp.json under
        spec.save_directory; logs one INFO line.
    Exceptions: FileNotFoundError if game has no clean binder;
        BinderModsMismatchError if a saved stamp disagrees with spec;
        OSError on a failed write; TypeError, or whatever a mod raises,
        as _modded_card.

    Example:
        >>> binder = modded_binder(GameId.GWENT, BinderModsSpec((KeyCaseSpec(),)))
        >>> binder.version_for(GameId.GWENT) == clean.version_for(GameId.GWENT)
        True
    """
    # Reuse a saved bake when its stamp says it was made by this spec
    saved = _load_saved_bake(game, spec)
    if saved is not None:
        _logger.info(
            "binder mods for %s: loaded from %s (clean binder %s...)",
            game.value,
            spec.save_directory,
            saved.version_for(game)[:12],
        )
        return saved

    # Otherwise mod the clean binder now, then save it for later runs
    result = ModdedCardBinder.baked_from_clean_file(
        CardBinder.existing_output_path(game), game, spec
    )
    if spec.save_directory is not None:
        _save_bake(result, game, spec, spec.save_directory)
    _logger.info("binder mods for %s: applied to the clean binder", game.value)
    return result


def require_writable_save_directory(spec: BinderModsSpec) -> None:
    """Create spec.save_directory if needed and confirm a file can be
    written in it, so a bad path fails before any binder is loaded; no-op
    when it is None. The probe file's name is unique per process, since
    arms may run at once.

    Inputs: spec. Output: none.
    Side effects: may create the directory; writes and deletes a probe file.
    Exceptions: OSError if the directory cannot be created or written to.

    Example:
        >>> require_writable_save_directory(config.binder_mods)
    """
    if spec.save_directory is None:
        return
    spec.save_directory.mkdir(parents=True, exist_ok=True)
    probe = spec.save_directory / f".write_probe_{os.getpid()}"
    probe.write_text("", encoding="utf-8")
    probe.unlink()


def _load_saved_bake(game: GameId, spec: BinderModsSpec) -> ModdedCardBinder | None:
    """The saved bake for game under spec.save_directory, or None when
    there is no save directory or no stamp for game (an unfinished save
    has no stamp). The read stamp must equal
    _BinderModsStamp.made_by(spec, read.source_binder_version).

    Inputs: game, spec. Output: ModdedCardBinder or None.
    Side effects: reads the stamp and, on a match, the saved binder.
    Exceptions: BinderModsMismatchError if the stamp differs from that
        expected stamp; as _BinderModsStamp.read and
        ModdedCardBinder.from_saved_bake.
    """
    if spec.save_directory is None:
        return None
    stamp_path = spec.save_directory / _STAMP_NAME.format(game=game.value)
    if not stamp_path.exists():
        return None

    # The stamp must be the one this spec would write over the same clean binder
    read = _BinderModsStamp.read(stamp_path)
    expected = _BinderModsStamp.made_by(spec, read.source_binder_version)
    if read != expected:
        raise BinderModsMismatchError(
            f"{stamp_path} was made by another binder_mods pipeline, seed or "
            f"format ({read}, want {expected}); delete {spec.save_directory} "
            "or pick another save_directory"
        )
    return ModdedCardBinder.from_saved_bake(
        spec.save_directory / _BINDER_NAME.format(game=game.value),
        game,
        read.source_binder_version,
    )


def _built_pipeline(spec: BinderModsSpec) -> ModPipeline:
    """spec.pipeline built into a ModPipeline, each mod seeded from a
    stream seeded by (spec.seed, "binder_mods"), as
    dojo_recipes/augmentations.py seeds a dojo's mods. Built once per game
    (never shared across games), so a game's bake does not depend on which
    other games the run loads.

    Inputs: spec. Output: ModPipeline. Side effects: none.
    Exceptions: as a ModSpec's build.
    """
    seeds = random.Random(f"{spec.seed}:binder_mods")
    return ModPipeline(
        [mod_spec.build(seeds.randrange(2**32)) for mod_spec in spec.pipeline]
    )


def _modded_card(pipeline: ModPipeline, card: GenericCard) -> GenericCard:
    """card after every mod of pipeline (train_only mods included: the
    bake is data, not a training-time step). The result is cacheable: a
    random mod stamps its card do-not-cache, but a baked card is the one
    fixed text for its uuid for the whole run.

    Inputs: pipeline, card. Output: GenericCard (card itself when no mod
        changed it). Side effects: advances the mods' random state.
    Exceptions: TypeError if the pipeline returns something other than a
        GenericCard; whatever a mod raises.
    """
    modded, _ = pipeline.apply_single((card, None), is_training=True)
    if not isinstance(modded, GenericCard):
        raise TypeError(f"binder mods turned a card into {type(modded).__name__}")
    if modded.cacheable:
        return modded
    return dataclasses.replace(modded, cacheable=True)


def _save_bake(
    binder: ModdedCardBinder, game: GameId, spec: BinderModsSpec, save_directory: Path
) -> None:
    """Write game's modded binder, alias ledger and stamp into a temp
    directory inside save_directory (named per process, since arms
    may run at once), then os.replace each into place, the stamp last, so
    no run ever reads a half-written bake as complete.

    Inputs: binder, game, spec (for the stamp), save_directory
        (spec.save_directory, checked not None by the caller).
    Output: none. Side effects: writes three files; removes the temp
        directory. Exceptions: OSError.
    """
    binder_name = _BINDER_NAME.format(game=game.value)
    stamp_name = _STAMP_NAME.format(game=game.value)
    staging = save_directory / f".staging_{game.value}_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        # Write everything into staging: the binder (CardBinder.save also
        # writes its alias ledger beside it), then the stamp
        binder.save(staging / binder_name, game)
        _BinderModsStamp.made_by(spec, binder.version_for(game)).write(
            staging / stamp_name
        )

        # Move the data files first and the stamp last: the stamp marks
        # the bake complete
        for staged in sorted(staging.iterdir()):
            if staged.name != stamp_name:
                os.replace(staged, save_directory / staged.name)
        os.replace(staging / stamp_name, save_directory / stamp_name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _mods_fingerprint(spec: BinderModsSpec) -> str:
    """sha256 hex of repr(spec.pipeline): the specs are frozen dataclasses
    of tuples, enums and scalars, so the repr is stable across processes.

    Inputs: spec. Output: str. Side effects: none. Exceptions: none.
    """
    return hashlib.sha256(repr(spec.pipeline).encode("utf-8")).hexdigest()
