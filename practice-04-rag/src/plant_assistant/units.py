"""Numbers and units: the same value is "4,5 мм/с" in a regulation and "4.5 mm/s" in an answer.

Numbers and the unit synonym table come from practice 3 (`metrics.numbers`, `qa.UNIT_SYNONYMS`), with the
units of practice 4 added (MPa, days). Grading and citation checks compare numbers by value, never as text.
"""

from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
# a clause number "4.3.3" is not the number 4.3: no digit group may follow
NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?: \d{3})+(?![\d])|\d+)(?:[.,]\d+)?(?![\w]|[.,]\d)")
ROMAN = {"I", "II", "III", "IV", "V", "VI"}


def plain(text: str) -> str:
    t = re.sub(r"<br\s*/?>", " ", text or "")
    t = html.unescape(_TAG.sub(" ", t))
    t = re.sub(r":?-{3,}:?", " ", t)
    t = re.sub(r"[|#*_`]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def numbers(text: str) -> list[str]:
    """Numbers as written: '4,5', '25 000', '2019'."""
    return NUMBER.findall(plain(text))


def number_value(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        return float(str(token).strip().replace(" ", "").replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def same_value(a: str | None, b: str | None) -> bool:
    """4,5 == 4.5 == '4.50'; roman numerals and other words compare as text."""
    if a is None or b is None:
        return False
    va, vb = number_value(a), number_value(b)
    if va is not None and vb is not None:
        return abs(va - vb) < 1e-9
    return str(a).strip().upper() == str(b).strip().upper()


def has_value(text: str, value: str) -> bool:
    """Is the value in the text? Numbers by value, a roman numeral as a whole word."""
    v = number_value(value)
    if v is None:
        return re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", text) is not None
    return any(abs((number_value(t) or -1e18) - v) < 1e-9 for t in numbers(text))


UNIT_SYNONYMS = {
    "mm/s": ["mm/s", "мм/с", "mm/sec", "mm/s rms"],
    "%": ["%", "percent", "процента", "процентов"],
    "months": ["months", "month", "мес", "месяц", "месяца", "месяцев"],
    "min": ["min", "minutes", "minute", "мин", "минут", "минуты"],
    "h": ["h", "hours", "hour", "hrs", "ч", "час", "часа", "часов", "operating hours", "часов наработки",
          "ч наработки", "моточасов", "h of operation"],
    "days": ["days", "day", "сутки", "суток", "дня", "дней", "calendar days"],
    "kW": ["kw", "квт"],
    "°C": ["°c", "c", "degc", "°с", "градусов", "℃", "deg c"],
    "mm": ["mm", "мм"],
    "years": ["years", "year", "лет", "года", "год"],
    "rpm": ["rpm", "об/мин", "r/min", "rev/min", "1/min", "min-1", "мин-1"],
    "Hz": ["hz", "гц"],
    "m³/h": ["m³/h", "m3/h", "м³/ч", "м3/ч", "m^3/h", "m3/hr", "m³/hr"],
    "m": ["m", "м"],
    "kg": ["kg", "кг"],
    "MPa": ["mpa", "мпа"],
}
_UNIT = {s.replace(" ", ""): canon for canon, syn in UNIT_SYNONYMS.items() for s in syn}


def canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = unit.strip().lower().replace(" ", "").rstrip(".")
    return _UNIT.get(u, unit.strip())


def unit_in_text(unit: str, text: str) -> bool:
    """Does the text name the unit (in any of its spellings)? Used by the citation check."""
    t = " " + plain(text).lower() + " "
    for s in UNIT_SYNONYMS.get(unit, [unit]):
        if len(s) <= 2:  # "h", "m", "ч": only as a separate word
            if re.search(rf"(?<![\wа-яё]){re.escape(s)}(?![\wа-яё])", t):
                return True
        elif s in t:
            return True
    return False
