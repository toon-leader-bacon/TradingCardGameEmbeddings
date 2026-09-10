from src.data_refinement.metrics.play_gwent.leader_labels import LEADER_NAMES


def test_no_duplicate_names() -> None:
    assert len(LEADER_NAMES) == len(set(LEADER_NAMES))


def test_matches_last_generated_count() -> None:
    assert len(LEADER_NAMES) == 42
