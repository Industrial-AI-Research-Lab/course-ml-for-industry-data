"""Equipment tags in one standard form, by code.

The same pump is written "P-101", "Р-101" (a Cyrillic Р that looks the same), "P101", "p-101" or "P 101".
For a person these are one tag; for search they are different strings. The rule is applied to the chunks
when the index is built AND to the question when it is asked: normalizing only one side changes nothing.

Only tags of known equipment prefixes are touched, so "ТО-1", "G2.5" or "6312 C3" stay as they are.
"""

from __future__ import annotations

import re

# Cyrillic letters that look like Latin ones: the usual source of "twin" tags
LOOKALIKE = str.maketrans("АВЕКМНОРСТХУавекмнорстху", "ABEKMHOPCTXYabekmhopctxy")

# Prefixes of equipment tags at this plant (P pump, K compressor, B fan, V valve, T tank, ...).
TAG_PREFIXES = {"P", "K", "B", "V", "T", "E", "F", "M", "XV", "FV", "HV", "PSV", "FT", "PT", "VT", "TT", "LT",
                "CV", "GB"}

# letters, then an optional hyphen or space, then digits and an optional suffix letter: P-101, Р101, P 101, P-212A
_CANDIDATE = re.compile(r"(?<![\w\-./,])([A-Za-zА-Яа-яЁё]{1,3})([ \-–]?)(\d{1,4})([A-Z]?)(?!\w)")
TAG_RE = re.compile(r"\b(?:" + "|".join(sorted(TAG_PREFIXES, key=len, reverse=True)) + r")-\d{1,4}[A-Z]?\b")


def _as_tag(letters: str, sep: str, digits: str, suffix: str) -> str | None:
    latin = letters.translate(LOOKALIKE).upper()
    if not latin.isascii() or latin not in TAG_PREFIXES:
        return None
    if sep == "" and len(digits) < 3 and not (letters.isascii() and letters.isupper() and len(digits) == 2):
        return None  # "м3", "G2" are not tags; "P101", "V37" are
    if sep == " " and len(digits) < 3:  # "в 12 месяцев" is not a tag; "P 101" is
        return None
    if sep == " " and not letters.isupper():  # "в 120 случаях" is not a tag
        return None
    return f"{latin}-{digits}{suffix}"


def normalize_tags(text: str) -> str:
    """Rewrite every equipment tag in the text into the form "P-101"."""

    def repl(m: re.Match) -> str:
        tag = _as_tag(m.group(1), m.group(2), m.group(3), m.group(4))
        return tag if tag else m.group(0)

    return _CANDIDATE.sub(repl, text)


def find_tags(text: str) -> list[str]:
    """Equipment tags in the text, normalized, in order of appearance, without repeats."""
    return list(dict.fromkeys(TAG_RE.findall(normalize_tags(text))))
