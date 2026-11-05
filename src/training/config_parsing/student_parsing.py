"""The run config's `student:` section -> StudentSettings.

    student:                          # optional; every key optional
      min_neighbours_per_deck: 1      # anchors with fewer are not dealt
      max_neighbours_per_deck: 1      # neighbours per pseudo-deck

An absent section or key keeps StudentSettings' default.
"""

from src.dojos.file_managers.neighbour_dealer import StudentSettings
from src.training.config_parsing.config_section import (
    ConfigSection,
    build_flat_dataclass,
)


def parse_student(section: ConfigSection) -> StudentSettings:
    """A `student:` section -> StudentSettings; an absent key keeps
    StudentSettings' default.

    Inputs: section ("student").
    Output: StudentSettings.
    Side effects: marks keys read.
    Exceptions: ValueError naming the location for an unknown or wrongly
        typed key, or what
        StudentSettings' own validation rejects.

    Example:
        >>> parse_student(top.required_section("student")).max_neighbours_per_deck
        1
    """
    return build_flat_dataclass(StudentSettings, section)
