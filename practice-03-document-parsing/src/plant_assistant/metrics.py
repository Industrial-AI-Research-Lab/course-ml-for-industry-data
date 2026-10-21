"""Component metrics of reading: CER, numbers, length, repetitions, tags.

CER is hygiene, not quality: the same 2% can be typos in words (harmless) or one wrong digit in a limit
(critical). So next to CER we count numbers, and for VLMs the length ratio and repeated lines: a VLM
fails with clean text (invented, rewritten, cut off, looped), and CER alone does not show that.
"""

from __future__ import annotations

import html
import re
from collections import Counter

from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------------------
_TAG = re.compile(r"<[^>]+>")
_RULE = re.compile(r":?-{3,}:?")
_MARKUP = re.compile(r"[|#*_`]")


def plain(text: str) -> str:
    """Markdown or HTML output -> plain words, one space between them. Used before comparing texts."""
    t = re.sub(r"<br\s*/?>", " ", text or "")
    t = _TAG.sub(" ", t)
    t = html.unescape(t)
    t = _RULE.sub(" ", t)
    t = _MARKUP.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after `plain()`: edits needed / length of the reference."""
    ref, hyp = plain(reference), plain(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def length_ratio(reference: str, hypothesis: str) -> float:
    """Output length / reference length. Far below 1: something is missing; far above 1: invented or looped."""
    ref = plain(reference)
    return len(plain(hypothesis)) / len(ref) if ref else 0.0


def repeated_share(text: str, min_len: int = 20) -> float:
    """Share of long lines that repeat an earlier line: the trace of a repetition loop."""
    lines = [plain(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if len(x) >= min_len]
    if not lines:
        return 0.0
    counts = Counter(lines)
    return sum(c - 1 for c in counts.values()) / len(lines)


# ---------------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------------
NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?: \d{3})+(?![\d])|\d+)(?:[.,]\d+)?(?![\w])")


def numbers(text: str) -> list[str]:
    """Numbers as written: '4,5', '25 000', '2019'. Clause numbers like 4.3.3 give '4.3'."""
    return NUMBER.findall(plain(text))


def number_value(token: str) -> float | None:
    try:
        return float(token.replace(" ", "").replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def number_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """How many numbers of the reference are in the output: as written, and by value.

    "by value" counts 4.5 for 4,5. The gap between the two is the share of numbers whose notation
    was changed: a VLM "rewrites" instead of reading.
    """
    ref, hyp = Counter(numbers(reference)), Counter(numbers(hypothesis))
    total = sum(ref.values())
    if not total:
        return {"numbers": 0, "numbers_exact": 1.0, "numbers_value": 1.0}
    exact = sum(min(c, hyp[t]) for t, c in ref.items())
    ref_v = Counter(number_value(t) for t in ref.elements())
    hyp_v = Counter(number_value(t) for t in hyp.elements())
    value = sum(min(c, hyp_v[v]) for v, c in ref_v.items())
    return {"numbers": total, "numbers_exact": exact / total, "numbers_value": value / total}


def page_scores(reference: str, hypothesis: str) -> dict[str, float]:
    return {"cer": cer(reference, hypothesis), **number_scores(reference, hypothesis),
            "length_ratio": length_ratio(reference, hypothesis), "repeated": repeated_share(hypothesis)}


# ---------------------------------------------------------------------------------------------
# Equipment tags
# ---------------------------------------------------------------------------------------------
# Cyrillic letters that look like Latin ones: OCR and models mix them up in tags (Р-101 vs P-101).
LOOKALIKES = str.maketrans("АВЕКМНОРСТХУ", "ABEKMHOPCTXY")


def normalize_tag(tag: str) -> str:
    """'pt 101', 'PT101', 'Р-101' (Cyrillic) -> 'PT-101', 'P-101'. Normalization that code can do, code does."""
    t = tag.strip().upper().translate(LOOKALIKES)
    t = re.sub(r"[\s‐-―_]+", "-", t)
    m = re.fullmatch(r"([A-Z]+)-?(\d+[A-Z]?)", t)
    return f"{m.group(1)}-{m.group(2)}" if m else t


def tag_scores(truth: list[str], found: list[str]) -> dict:
    t = {normalize_tag(x) for x in truth}
    f = {normalize_tag(x) for x in found}
    hit = t & f
    return {"precision": len(hit) / len(f) if f else 0.0, "recall": len(hit) / len(t) if t else 0.0,
            "found": len(f), "correct": len(hit), "invented": sorted(f - t), "missed": sorted(t - f)}
