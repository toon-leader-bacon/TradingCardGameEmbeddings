"""The evicting slot policies (FirstInFirstOut, LeastRecentlyUsed,
LeastFrequentlyUsed) and the SecondOfferAdmission decorator: what each
admits and evicts, and that none evicts a pinned key."""

from typing import Callable

import pytest

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.slot_policy import (
    FillUntilFull,
    FirstInFirstOut,
    LeastFrequentlyUsed,
    LeastRecentlyUsed,
    SecondOfferAdmission,
    SlotPolicy,
)


def _key(text: str) -> CardTextKey:
    return CardTextKey.of_text(text)


A, B, C, D = (_key(text) for text in "abcd")

EVICTING_POLICIES: list[Callable[[int], SlotPolicy]] = [
    FirstInFirstOut,
    LeastRecentlyUsed,
    LeastFrequentlyUsed,
]


def _filled(policy: SlotPolicy, *keys: CardTextKey) -> SlotPolicy:
    """policy after admitting keys one at a time into free rows."""
    for key in keys:
        assert policy.admit([key], free_rows=1, pinned=set()).admitted == [key]
    return policy


@pytest.mark.parametrize("build", EVICTING_POLICIES)
class TestEveryEvictingPolicy:
    def test_admits_into_free_rows_without_evicting(self, build) -> None:
        decision = build(2).admit([A, B], free_rows=2, pinned=set())
        assert (decision.admitted, decision.evicted) == ([A, B], [])

    def test_the_first_capacity_rows_offered_win(self, build) -> None:
        assert build(2).admit([A, B, C], free_rows=2, pinned=set()).admitted == [A, B]

    def test_an_empty_offer_changes_nothing(self, build) -> None:
        decision = _filled(build(1), A).admit([], free_rows=0, pinned=set())
        assert (decision.admitted, decision.evicted) == ([], [])

    def test_never_evicts_a_pinned_key(self, build) -> None:
        policy = _filled(build(2), A, B)
        decision = policy.admit([C], free_rows=0, pinned={A})
        assert (decision.admitted, decision.evicted) == ([C], [B])

    def test_counts_keys_refused_because_every_held_row_is_pinned(self, build) -> None:
        policy = _filled(build(2), A, B)
        decision = policy.admit([C, D], free_rows=0, pinned={A, B})
        assert (decision.admitted, decision.refused_for_pins) == ([], 2)

    def test_pins_on_keys_not_held_do_not_shrink_the_room(self, build) -> None:
        policy = _filled(build(1), A)
        decision = policy.admit([B], free_rows=0, pinned={C})
        assert (decision.admitted, decision.evicted) == ([B], [A])

    def test_rejects_a_capacity_below_one(self, build) -> None:
        with pytest.raises(ValueError, match="capacity_rows"):
            build(0)


class TestFirstInFirstOut:
    def test_evicts_the_oldest_admitted_whatever_was_hit(self) -> None:
        policy = _filled(FirstInFirstOut(2), A, B)
        policy.record_hits([A, A])
        assert policy.admit([C], free_rows=0, pinned=set()).evicted == [A]


class TestLeastRecentlyUsed:
    def test_a_hit_moves_a_key_to_the_newest_end(self) -> None:
        policy = _filled(LeastRecentlyUsed(2), A, B)
        policy.record_hits([A])
        assert policy.admit([C], free_rows=0, pinned=set()).evicted == [B]

    def test_hits_on_keys_not_held_are_ignored(self) -> None:
        policy = _filled(LeastRecentlyUsed(1), A)
        policy.record_hits([B])
        assert policy.admit([C], free_rows=0, pinned=set()).evicted == [A]


class TestLeastFrequentlyUsed:
    def test_evicts_the_least_hit_key(self) -> None:
        policy = _filled(LeastFrequentlyUsed(3), A, B, C)
        policy.record_hits([A, A, B, C, C])
        assert policy.admit([D], free_rows=0, pinned=set()).evicted == [B]

    def test_ties_go_to_the_oldest(self) -> None:
        policy = _filled(LeastFrequentlyUsed(2), A, B)
        policy.record_hits([B, A])
        assert policy.admit([C], free_rows=0, pinned=set()).evicted == [B]

    def test_a_new_key_starts_at_zero_hits(self) -> None:
        policy = _filled(LeastFrequentlyUsed(2), A, B)
        policy.record_hits([A, B])
        assert policy.admit([C], free_rows=0, pinned=set()).evicted == [A]
        # C has 0 hits, B has 1: C goes next
        assert policy.admit([D], free_rows=0, pinned=set()).evicted == [C]

    def test_an_evicted_key_forgets_its_hits(self) -> None:
        policy = _filled(LeastFrequentlyUsed(1), A)
        policy.record_hits([A, A, A])
        _filled_again = policy.admit([B], free_rows=0, pinned=set())
        assert _filled_again.evicted == [A]
        policy.record_hits([A])  # not held: ignored
        assert policy.admit([A], free_rows=0, pinned=set()).evicted == [B]

    def test_evicts_across_buckets_skipping_pinned_keys(self) -> None:
        policy = _filled(LeastFrequentlyUsed(3), A, B, C)
        policy.record_hits([B, C, C])
        decision = policy.admit([D, _key("e")], free_rows=0, pinned={A})
        assert decision.evicted == [B, C]


class TestSecondOfferAdmission:
    def test_a_key_is_admitted_on_its_second_offer(self) -> None:
        policy = SecondOfferAdmission(LeastRecentlyUsed(2), remembered_keys=2)
        assert policy.admit([A], free_rows=2, pinned=set()).admitted == []
        assert policy.admit([A], free_rows=2, pinned=set()).admitted == [A]

    def test_an_admitted_key_is_forgotten(self) -> None:
        policy = SecondOfferAdmission(FillUntilFull(), remembered_keys=2)
        policy.admit([A], free_rows=2, pinned=set())
        policy.admit([A], free_rows=2, pinned=set())
        # Offered again (say, after the store evicted it): a first offer
        assert policy.admit([A], free_rows=2, pinned=set()).admitted == []

    def test_only_repeats_reach_the_inner_policy(self) -> None:
        policy = SecondOfferAdmission(FillUntilFull(), remembered_keys=4)
        policy.admit([A], free_rows=4, pinned=set())
        assert policy.admit([B, A, C], free_rows=4, pinned=set()).admitted == [A]

    def test_a_refused_repeat_stays_remembered(self) -> None:
        policy = SecondOfferAdmission(FillUntilFull(), remembered_keys=2)
        policy.admit([A], free_rows=0, pinned=set())
        assert policy.admit([A], free_rows=0, pinned=set()).admitted == []
        assert policy.admit([A], free_rows=1, pinned=set()).admitted == [A]

    def test_the_oldest_first_offer_is_forgotten_past_the_bound(self) -> None:
        policy = SecondOfferAdmission(FillUntilFull(), remembered_keys=2)
        policy.admit([A, B, C], free_rows=4, pinned=set())
        assert policy.admit([A, B, C], free_rows=4, pinned=set()).admitted == [B, C]

    def test_a_repeat_pushed_out_in_the_same_offer_is_still_admitted(self) -> None:
        policy = SecondOfferAdmission(FillUntilFull(), remembered_keys=1)
        policy.admit([A], free_rows=2, pinned=set())
        # A is a repeat, then B's first offer pushes A out of the memory
        assert policy.admit([A, B], free_rows=2, pinned=set()).admitted == [A]
        assert policy.admit([B], free_rows=2, pinned=set()).admitted == [B]

    def test_hits_reach_the_inner_policy(self) -> None:
        inner = LeastRecentlyUsed(2)
        policy = SecondOfferAdmission(inner, remembered_keys=4)
        for _ in range(2):
            policy.admit([A, B], free_rows=2, pinned=set())
        policy.record_hits([A])
        for _ in range(2):
            decision = policy.admit([C], free_rows=0, pinned=set())
        assert decision.evicted == [B]

    def test_rejects_a_bound_below_one(self) -> None:
        with pytest.raises(ValueError, match="remembered_keys"):
            SecondOfferAdmission(FillUntilFull(), remembered_keys=0)
