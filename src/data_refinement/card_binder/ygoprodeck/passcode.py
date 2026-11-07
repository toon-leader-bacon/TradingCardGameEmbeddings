"""How a Yu-Gi-Oh passcode is spelled as a DataSource.YGOPRODECK alias id.

A neutral module, not the alias stage's: the deck stage
(deck_box/ygoprodeck_decks/) imports it too, so both sides spell ids the
same without the deck stage depending on a binder stage's
implementation.
"""


def passcode_id(raw: str) -> str:
    """The alias id for a passcode: its decimal digits without leading
    zeros, so "08512558" and "8512558" are one id.

    Inputs: raw (a passcode as YGOPRODeck writes it; surrounding
        whitespace is ignored).
    Output: str (the digits, no leading zeros; "0" for zero).
    Side effects: none.
    Exceptions: ValueError if raw isn't a non-negative decimal integer.

    Example:
        >>> passcode_id("08512558")
        '8512558'
    """
    digits = raw.strip()
    if not digits.isdecimal() or not digits.isascii():
        raise ValueError(f"not a passcode: {raw!r}")
    return str(int(digits))
