"""Rules baseline: regular expressions, keyword dictionaries and a registry lookup. No model.

Every LLM solution is compared with this. For several fields (tag, date, duration) rules are
cheap, fast, deterministic and often as good as a model. A rule that is not sure returns None,
so we also see coverage: the share of fields the rules could fill at all.
"""

from __future__ import annotations

import re
from datetime import date

from .data import Equipment

# Cyrillic letters that look like Latin ones (typed on a Russian keyboard layout).
_TWINS = str.maketrans({"Р": "P", "К": "K", "М": "M", "Е": "E", "С": "C", "В": "B", "А": "A", "Х": "X",
                        "р": "P", "к": "K", "м": "M", "е": "E", "с": "C", "в": "B", "а": "A", "х": "X"})
_TAG = re.compile(r"(?<![A-Za-zА-Яа-я])([A-ZА-Я]{1,3})[-\s]?(\d{2,4})([A-ZА-Я])?(?![\dA-Za-zА-Яа-я])")

_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), ("y", "m", "d")),
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"), ("d", "m", "y")),
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2})\b"), ("d", "m", "yy")),
]

_HOURS_MINUTES = re.compile(r"(\d+)\s*(?:ч|h)\s*(\d+)\s*(?:м|мин|m|min)\b", re.IGNORECASE)
_HOURS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:ч\b|час|h\b|hr|hrs|hour)", re.IGNORECASE)
_MINUTES = re.compile(r"(\d+)\s*(?:мин|min|м\b)", re.IGNORECASE)

# Keyword dictionaries: first match wins, so order matters (more specific first).
WORK_KEYWORDS = [
    ("preventive_maintenance", r"ппр|плановое то|то-\d|то по графику|плановое обслуж|\bpm\b|scheduled|planned service|preventive"),
    ("lubrication", r"смазк|долив|grease|greasing|lubric|oil top"),
    ("adjustment", r"центровк|регулир|подтяж|соосност|alignment|adjust|tighten"),
    ("cleaning", r"чистк|промыв|продув|clean|flush"),
    ("inspection", r"осмотр|обход|проверка|inspection|walk-?down|routine check"),
    ("corrective_repair", r"ремонт|отказ|неисправн|авар|repair|breakdown|fault|failure|fixed"),
]
FAILURE_KEYWORDS = [
    ("seal_leak", r"теч[ьи]|подтек|утечк|капает|leak"),
    ("misalignment", r"расцентр|соосност|misalign"),
    ("imbalance", r"дисбаланс|unbalance|imbalance"),
    ("overheating", r"перегр|гре[ею]тся|температур|overheat|running hot|high temp"),
    ("corrosion", r"корроз|ржав|corros|rust"),
    ("electrical_fault", r"изоляц|клемм|обмотк|замыкан|insulation|terminal|winding|short circuit"),
    ("blockage", r"засор|забит|clog|blockage|plugged"),
    ("bearing_wear", r"подшипник\w* (?:шум|гуд)|гул подшип|люфт|износ подшип|bearing (?:worn|noise|play|wear)"),
    ("none", r"замечаний нет|без замечаний|в норме|всё в порядке|все в порядке|\bок\b|no issues|all normal|nothing found|\bok\b"),
]
PART_KEYWORDS = {
    "bearing": r"подшипник|bearing",
    "mechanical_seal": r"торцев\w* уплотн|торц\. уплотн|торцевик|mechanical seal",
    "gasket": r"прокладк|gasket",
    "coupling": r"муфт|coupling",
    "belt": r"ремень|ремни|ремн|\bbelts?\b",
    "impeller": r"рабоч\w* колес|крыльчатк|\bрк\b|impeller",
    "filter": r"фильтр|filter",
    "oil": r"масл|\boil\b",
    "valve_seat": r"седл|valve seat",
    "sensor": r"датчик|sensor",
}
REPLACE_WORDS = r"зам\.|замен|заменил|поменял|залил|долил|долив|установ|новый|новую|replac|changed|new\b|refill|top-?up|installed"
DOWNTIME_FALSE = r"без останов|без простоя|на ходу|не останавл|kept running|without stop|no downtime|while running"
DOWNTIME_TRUE = r"останов|остановлен|простой|вывели|выведен|stopp|shut ?down|downtime|\bост\.|offline"


def find_equipment_id(text: str, registry: dict[str, Equipment]) -> str | None:
    for m in _TAG.finditer(text):
        tag = f"{m.group(1)}-{m.group(2)}{m.group(3) or ''}".translate(_TWINS)
        if tag in registry:
            return tag
    return None


def find_date(text: str) -> str | None:
    for pattern, order in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        parts = dict(zip(order, m.groups()))
        year = int(parts["y"]) if "y" in parts else 2000 + int(parts["yy"])
        try:
            return date(year, int(parts["m"]), int(parts["d"])).isoformat()
        except ValueError:
            return None
    return None


def find_duration_hours(text: str) -> float | None:
    if m := _HOURS_MINUTES.search(text):
        return round(int(m.group(1)) + int(m.group(2)) / 60, 4)
    if m := _HOURS.search(text):
        return float(m.group(1).replace(",", "."))
    if m := _MINUTES.search(text):
        return round(int(m.group(1)) / 60, 4)
    return None


def _first(rules: list[tuple[str, str]], low: str) -> str | None:
    for label, pattern in rules:
        if re.search(pattern, low):
            return label
    return None


def find_replaced_parts(low: str) -> list[str]:
    if not re.search(REPLACE_WORDS, low):
        return []
    return sorted(part for part, pattern in PART_KEYWORDS.items() if re.search(pattern, low))


def find_downtime(low: str) -> bool | None:
    if re.search(DOWNTIME_FALSE, low):
        return False
    if re.search(DOWNTIME_TRUE, low):
        return True
    return None


def extract_with_rules(text: str, registry: dict[str, Equipment]) -> dict:
    """Same fields as MaintenanceRecord. None means: the rules could not decide."""
    low = text.lower()
    equipment_id = find_equipment_id(text, registry)
    return {
        "equipment_id": equipment_id,
        # A lookup is the best rule there is: never ask a model what the registry already knows.
        "equipment_type": registry[equipment_id].equipment_type if equipment_id else None,
        "work_type": _first(WORK_KEYWORDS, low),
        "failure_mode": _first(FAILURE_KEYWORDS, low),
        "defect_summary": None,
        "replaced_parts": find_replaced_parts(low),
        "duration_hours": find_duration_hours(text),
        "work_date": find_date(text),
        "downtime": find_downtime(low),
        "evidence": [],
    }
