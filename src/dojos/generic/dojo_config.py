"""DojoConfig - a parameter object bundling GenericDojo's configuration
values (as opposed to its real collaborators - card_lookup, holdout,
decoder_head, loss_calculator, mod_pipeline, deck_box - which stay
direct constructor parameters, dependency injection rather than
configuration).

Deliberately NOT included here: path_to_training_data. Unlike every
field below, it is inherently specific to which metric a given leaf
wrapper wraps (a wrapper class already has a sensible per-metric
default for it via that metric's own DEFAULT_OUTPUT_PATH - see e.g.
src/dojos/seventeenlands/draft_data/pack_card_tally_dojos.py's
CardTakeRateDojo). It stays its own constructor parameter everywhere.

This class is a dumb value object on purpose: fields and their
documentation only, no methods, no resolution logic. Every field here
can legitimately be None/left at its default, meaning "caller didn't
set this" - GenericDojo.__init__ is the ONLY place that decides what a
None value falls back to (see its own docstring), so that logic never
has to be duplicated here or in any subclass. See plans/dojo_config.md
for the full design writeup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class DojoConfig:
    # region Identity
    # This dojo's name: the Trainer's dict key (Trainer.__init__ raises
    # "dojo names must be unique" if two dojos in one run resolve to the
    # same name) and, by default, this dojo's split-file prefix too (see
    # output_file_prefix below - GenericDojo resolves it from this
    # field, not straight from path_to_training_data.stem, so setting
    # name alone avoids a split-file collision for free).
    #
    # None (the default) falls back to path_to_training_data.stem -
    # fine for the common case of one dojo instance per metric file.
    #
    # STRONGLY ENCOURAGED to set explicitly whenever more than one dojo
    # is built from metric files that share a bare filename (e.g. the
    # same metric name under two different 17lands expansion/format
    # directories, such as .../KTK/TradSealed/deck_win_prediction.parquet
    # and .../MSH/PremierDraft/deck_win_prediction.parquet) - both would
    # otherwise resolve to the same name and collide.
    name: str | None = None
    # endregion

    # region Split file management
    # Where this dojo's train/test/validation split files live and how
    # they're (re)built. These four fields matter together: two dojo
    # instances that resolve to the same (output_directory,
    # output_file_prefix) pair will silently read/overwrite each
    # other's splits.

    # Directory every split file is written under. Defaults to the
    # directory every dojo in this project has always used. Override
    # when a dojo instance must not share split storage with the
    # default location - e.g. comparing two seedings of the same metric
    # side by side, or test isolation (every test in this project's own
    # suite that builds a dojo from a file named e.g. "source.parquet"
    # must set this to a tmp_path-scoped directory, or it will silently
    # collide with every other test doing the same).
    output_directory: Path = Path("data/splits")

    # Filename prefix for this dojo's three split files
    # (<prefix>_{train,test,validation}.parquet). None (the default)
    # falls back to the resolved name above (see GenericDojo.__init__),
    # NOT directly to path_to_training_data.stem - so setting name alone
    # already avoids a split-file collision without also having to set
    # this separately. Set independently only for the rare case of
    # wanting a dojo's split files named differently from its
    # Trainer-facing name.
    output_file_prefix: str | None = None

    # False (default): if this dojo's three split files already exist
    # at (output_directory, output_file_prefix), reuse them as-is and
    # skip re-running make_splits() - the expensive, whole-source-file
    # streaming step. True always rebuilds them regardless of what's
    # already on disk - e.g. after path_to_training_data has been
    # regenerated with different rows, or the split_ratios/shuffle
    # policy changed.
    force_resplit: bool = False

    # Seed for split shuffling. None (default) is non-deterministic -
    # every construction produces a different train/test/validation
    # row assignment, so two separate builds are not directly
    # comparable and a restart with force_resplit=True (or any first-
    # ever build) reshuffles differently each time. Pass an explicit
    # int for reproducible splits across restarts/comparisons - this is
    # rarely exercised in practice (it mostly matters when
    # deliberately rebuilding the same dojo's splits more than once to
    # compare configurations), but costs nothing to support.
    rng_seed: int | None = None
    # endregion

    # region Version checking
    # See plans/card_binder_versioning.md for the full mechanism:
    # path_to_training_data's embedded CardBinder (and, when relevant,
    # DeckBox) version is checked against the live card_lookup/deck_box
    # given to this dojo, before splitting.

    # True (default): a missing or mismatched version raises
    # immediately, before make_splits() runs - fail loud, not silent
    # staleness. False is the explicit "I know it's stale, proceed
    # anyway" escape hatch, never the default.
    strict_version_check: bool = True
    # endregion
