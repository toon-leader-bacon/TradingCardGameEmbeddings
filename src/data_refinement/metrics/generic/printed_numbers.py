"""Shared check for a card's printed numeric stat, for masked-field
regression metrics whose source stores stats as strings ("3", "*", "X",
"1+*")."""


def is_small_whole_number(value: object, maximum: int) -> bool:
    """Whether value is a printed whole number from 0 to maximum: a real
    stat rather than a variable one ("*", "X") or a joke card's (99).

    Inputs: value (a raw field value, any type; str(value) is checked),
        maximum (int, inclusive upper bound).
    Output: bool.
    Side effects: none. Exceptions: none.

    Example:
        >>> is_small_whole_number("3", 20), is_small_whole_number("*", 20)
        (True, False)
    """
    text = str(value)
    return text.isdigit() and int(text) <= maximum
