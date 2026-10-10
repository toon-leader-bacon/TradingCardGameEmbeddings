"""Append (or replace) a "Runtime:" paragraph at the end of a class or
function docstring, creating a one-paragraph docstring if there is none.

Usage: python scratch/add_runtime_note.py <file> <ClassOrFunctionName> "<note>"
The note is wrapped to 79 columns and indented to the docstring's level.
An existing "Runtime:" paragraph in that docstring is replaced.
"""

import re
import sys
import textwrap

_WIDTH = 79


def _paragraph(note: str, indent: str) -> str:
    return textwrap.fill(
        f"Runtime: {note}",
        width=_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def main(path: str, name: str, note: str) -> None:
    text = open(path, encoding="utf-8").read()
    header = re.search(
        rf"^(?P<outer>[ \t]*)(?:class|def) {re.escape(name)}\b.*?:\n",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if header is None:
        sys.exit(f"no class or def {name} in {path}")
    indent = header.group("outer") + "    "
    rest = text[header.end():]
    docstring = re.match(r'([ \t]+)"""(.*?)"""', rest, re.DOTALL)

    # No docstring yet: insert one holding just the runtime note
    if docstring is None:
        paragraph = _paragraph(note, indent)
        new_doc = f'{indent}"""{paragraph[len(indent):]}\n{indent}"""\n'
        if len(paragraph.splitlines()) == 1 and len(paragraph) + 6 <= _WIDTH:
            new_doc = f'{indent}"""{paragraph[len(indent):]}"""\n'
        new_text = text[: header.end()] + new_doc + rest
        open(path, "w", encoding="utf-8").write(new_text)
        return

    # Existing docstring: drop any old Runtime paragraph, append the new one
    body = docstring.group(2)
    body = re.sub(r"\n\n\s*Runtime:.*?(?=\n\n|\s*$)", "", body, flags=re.DOTALL)
    new_body = body.rstrip() + "\n\n" + _paragraph(note, indent) + "\n" + indent
    start = header.end() + docstring.start(2)
    end = header.end() + docstring.end(2)
    open(path, "w", encoding="utf-8").write(text[:start] + new_body + text[end:])


if __name__ == "__main__":
    main(*sys.argv[1:4])
