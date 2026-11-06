import pytest

from src.data_refinement.card_binder.tcgjson.game_spec import CatalogField, FieldKind
from src.data_refinement.card_binder.tcgjson.rules_text import clean_rules_text
from src.schema.json_types import JsonValue


def _field(kind: FieldKind) -> CatalogField:
    return CatalogField("attribute", "key", kind)


@pytest.mark.parametrize(
    "kind, value, expected",
    [
        (FieldKind.TEXT, "  Ruby \n", "Ruby"),
        (FieldKind.TEXT, "", None),
        (FieldKind.TEXT, None, None),
        (FieldKind.TEXT, ["a"], None),
        (FieldKind.INT, "8", 8),
        (FieldKind.INT, "0", 0),
        (FieldKind.INT, 5, 5),
        (FieldKind.INT, "X", None),
        (FieldKind.INT, "2.5", None),
        (FieldKind.INT, True, None),
        (FieldKind.LIST, "Dreamborn;Hero", ["Dreamborn", "Hero"]),
        (FieldKind.LIST, " Storyborn ; ;Villain", ["Storyborn", "Villain"]),
        (FieldKind.LIST, ["Character", 3], ["Character"]),
        (FieldKind.LIST, ";", None),
        (FieldKind.LIST, 7, None),
        (FieldKind.FLAG, "Yes", True),
        (FieldKind.FLAG, "No", False),
        (FieldKind.FLAG, "Unknown", None),
        (FieldKind.FLAG, None, None),
        (FieldKind.TEXT, "None", None),
        (FieldKind.LIST, "None", None),
        (FieldKind.LIST, ["Calm", "None"], ["Calm"]),
        (FieldKind.INT, "None", None),
        (FieldKind.RULES, "<strong>[Blocker]</strong> Block.<br>", "[Blocker] Block."),
        (FieldKind.RULES, "<br>", None),
        (FieldKind.RULES, ["a"], None),
    ],
)
def test_normalize_by_kind(
    kind: FieldKind, value: JsonValue, expected: JsonValue
) -> None:
    assert _field(kind).normalize(value) == expected


@pytest.mark.parametrize(
    "html, expected",
    [
        (
            "<strong>[Freeze]</strong>[Exert] — Exert it.<br>",
            "[Freeze] [Exert] — Exert it.",
        ),
        ("<strong>[Dragon Fire]</strong>When you play", "[Dragon Fire] When you play"),
        (
            "<strong>Evasive</strong> <em>(Only Evasive.)</em>",
            "Evasive (Only Evasive.)",
        ),
        ("<em>(Evasive</em>)", "(Evasive)"),
        ("Foo</strong>, bar", "Foo, bar"),
        ("<stong>Typo</stong> and <strrong>x</strrong>", "Typo and x"),
        ("<strong[I Wreck Things]</strong> Go.", "[I Wreck Things] Go."),
        ('<span style="color:red">Red</span>', "Red"),
        ("It’s “fine” now", 'It\'s "fine" now'),
        ("cost 2 or less", "cost 2 or less"),
        ("<br>", ""),
        # Store notes go, in a red span or on a line of their own
        (
            '<span style="color:red">This product comes in <strong>Three'
            " (3)</strong> serializations.</span><br>When Played: Draw.",
            "When Played: Draw.",
        ),
        (
            "Draw 1.<br><br><em>(Disclaimer: This card features a 2020"
            " Copyright date.)</em>",
            "Draw 1.",
        ),
        (
            "Draw 1.<br><strong>Official Tournament Stores must review their"
            " agreement.</strong>",
            "Draw 1.",
        ),
        (
            'Then, <span style="color:red">you may</span> play it.',
            "Then, you may play it.",
        ),
        (
            "This card cannot be Normal Summoned.<br>Draw 1.",
            "This card cannot be Normal Summoned. Draw 1.",
        ),
        ("This card is not affected by traps.", "This card is not affected by traps."),
    ],
)
def test_clean_rules_text(html: str, expected: str) -> None:
    assert clean_rules_text(html) == expected
