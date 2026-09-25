import dataclasses
from pathlib import Path

from src.dojos.generic.dojo_config import DojoConfig


def test_defaults_preserve_pre_dojo_config_behavior() -> None:
    config = DojoConfig()

    assert config.name is None
    assert config.output_directory == Path("data/splits")
    assert config.output_file_prefix is None
    assert config.force_resplit is False
    assert config.rng_seed is None
    assert config.strict_version_check is True


def test_is_frozen() -> None:
    config = DojoConfig()

    try:
        config.name = "renamed"  # type: ignore[misc]
        assert False, "expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_shared_default_instance_is_safe_across_multiple_uses() -> None:
    # A frozen dataclass has no mutable-default-argument hazard, so the
    # same DojoConfig() instance can be reused as a literal default
    # value (as GenericDojo and every task-shape class do) without one
    # caller's config leaking into another's.
    first = DojoConfig()
    second = DojoConfig()

    assert first == second
    assert first is not second


def test_partial_override_via_replace_keeps_untouched_fields() -> None:
    caller_config = DojoConfig(force_resplit=True)

    resolved = dataclasses.replace(
        caller_config, name=caller_config.name or "wrapper_default_name"
    )

    assert resolved.name == "wrapper_default_name"
    assert resolved.force_resplit is True
    assert caller_config.name is None  # replace() does not mutate the original


def test_partial_override_via_replace_does_not_clobber_an_explicit_value() -> None:
    caller_config = DojoConfig(name="caller_set_this")

    resolved = dataclasses.replace(
        caller_config, name=caller_config.name or "wrapper_default_name"
    )

    assert resolved.name == "caller_set_this"


def test_subclass_can_add_its_own_fields() -> None:
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class FancyDojoConfig(DojoConfig):
        extra_field: float = 1.0

    fancy = FancyDojoConfig(name="a", extra_field=2.0)

    assert isinstance(fancy, DojoConfig)
    assert fancy.name == "a"
    assert fancy.extra_field == 2.0
