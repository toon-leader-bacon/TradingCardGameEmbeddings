from src.data_retrieval.seventeenlands.known_files import list_known_refs
from src.data_retrieval.seventeenlands.refs import DataType, SeventeenLandsFileRef


class TestListKnownRefs:
    def test_includes_a_known_combination(self) -> None:
        refs = list_known_refs()

        assert (
            SeventeenLandsFileRef.from_known(DataType.GAME, "MSH", "PremierDraft")
            in refs
        )

    def test_excludes_a_combination_17lands_never_published(self) -> None:
        # STX+Sealed only ever got a GAME file, never DRAFT or REPLAY —
        # the motivating example for why this table can't just be the
        # cartesian product of every known expansion/format/data type.
        refs = list_known_refs()

        assert (
            SeventeenLandsFileRef.from_known(DataType.DRAFT, "STX", "Sealed")
            not in refs
        )

    def test_has_no_duplicate_triples(self) -> None:
        refs = list_known_refs()

        seen = {(ref.data_type, ref.expansion, ref.format_code) for ref in refs}
        assert len(seen) == len(refs)

    def test_sorted_by_data_type_then_expansion_then_format_code(self) -> None:
        refs = list_known_refs()

        keys = [(ref.data_type.value, ref.expansion, ref.format_code) for ref in refs]
        assert keys == sorted(keys)
