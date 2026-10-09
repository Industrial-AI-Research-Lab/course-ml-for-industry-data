"""Generate the synthetic maintenance log with ground truth.

How it works:
1. Facts first. A seeded random generator samples structured records (this IS the ground truth):
   equipment from a fixed registry, work type, failure mode, parts, duration, date, downtime.
2. Text second. An LLM writes the entry in a technician's style from the facts. We give it the
   exact strings for the tag, the date and the duration, and the meaning of the other facts.
3. Verify. A checker makes sure the text states exactly these facts (tag, date, duration,
   parts, downtime and failure cues). On failure we ask again with the problems listed;
   after 3 failures we use a deterministic template, which is correct by construction.

The data is synthetic and the labels follow the generator. Numbers measured on it are
illustrative: on a real log you measure again.

Usage (from practice-01-structured-extraction/):
    uv run python scripts/generate_dataset.py --profile gpt-oss-120b --n 300
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_assistant import DATA_DIR  # noqa: E402
from plant_assistant.config import get_profile  # noqa: E402
from plant_assistant.llm import CallLogger, LLMClient  # noqa: E402

SEED = 2026

# ---------------------------------------------------------------- registry (fixed, not random)

REGISTRY = [
    # tag, type, area
    ("P-101", "pump", "Pump station 1"), ("P-101A", "pump", "Pump station 1"),
    ("P-102A", "pump", "Pump station 1"), ("P-102B", "pump", "Pump station 1"),
    ("P-103", "pump", "Pump station 1"), ("P-104", "pump", "Pump station 1"),
    ("P-201", "pump", "Pump station 2"), ("P-202A", "pump", "Pump station 2"),
    ("P-202B", "pump", "Pump station 2"), ("P-301", "pump", "Water treatment"),
    ("K-110", "compressor", "Compressor house"), ("K-120", "compressor", "Compressor house"),
    ("K-210", "compressor", "Compressor house"),
    ("F-130", "fan", "Cooling tower"), ("F-131", "fan", "Cooling tower"), ("F-230", "fan", "Boiler house"),
    ("M-101", "electric_motor", "Pump station 1"), ("M-102", "electric_motor", "Pump station 1"),
    ("M-201", "electric_motor", "Pump station 2"), ("M-310", "electric_motor", "Water treatment"),
    ("V-112", "valve", "Pump station 1"), ("V-113", "valve", "Pump station 1"),
    ("V-214", "valve", "Pump station 2"), ("XV-301", "valve", "Water treatment"),
    ("FV-305", "valve", "Water treatment"),
    ("E-140", "heat_exchanger", "Compressor house"), ("E-141", "heat_exchanger", "Compressor house"),
    ("E-240", "heat_exchanger", "Boiler house"),
    ("CV-401", "conveyor", "Warehouse"), ("CV-402", "conveyor", "Warehouse"),
    ("GB-401", "gearbox", "Warehouse"), ("GB-402", "gearbox", "Warehouse"),
]

TYPE_NAMES = {
    "pump": (["насос", "нас.", "насосный агрегат"], ["pump"]),
    "compressor": (["компрессор", "компр."], ["compressor"]),
    "fan": (["вентилятор", "вент.", "вентилятор градирни"], ["fan"]),
    "electric_motor": (["электродвигатель", "эл.дв.", "ЭД", "двигатель"], ["motor", "electric motor"]),
    "valve": (["задвижка", "клапан", "арматура"], ["valve"]),
    "heat_exchanger": (["теплообменник", "ТОА"], ["heat exchanger", "HX"]),
    "conveyor": (["конвейер", "транспортёр"], ["conveyor"]),
    "gearbox": (["редуктор"], ["gearbox"]),
}

# ---------------------------------------------------------------- meanings and verification cues

WORK_MEANING = {
    "corrective_repair": ("ремонт после неисправности (внеплановый)", "corrective repair after a fault"),
    "preventive_maintenance": ("плановое ТО / ППР", "scheduled preventive maintenance"),
    "inspection": ("осмотр / обход / диагностика, без ремонта", "inspection / walk-down, no repair"),
    "lubrication": ("смазка / замена или долив масла как отдельная работа", "lubrication work"),
    "adjustment": ("регулировка / центровка / подтяжка", "adjustment / alignment / tightening"),
    "cleaning": ("чистка / промывка", "cleaning / flushing"),
}

FAILURE_MEANING = {
    "bearing_wear": ("износ подшипника (шум, люфт, вибрация)", "bearing wear (noise, play, vibration)"),
    "seal_leak": ("течь по уплотнению / прокладке", "leak at the seal or gasket"),
    "misalignment": ("расцентровка валов", "shaft misalignment"),
    "imbalance": ("дисбаланс ротора / рабочего колеса", "rotor or impeller imbalance"),
    "overheating": ("перегрев", "overheating"),
    "corrosion": ("коррозия", "corrosion"),
    "electrical_fault": ("электрическая неисправность (изоляция, клеммы, обмотка)", "electrical fault (insulation, terminals, winding)"),
    "blockage": ("засор / забит", "blockage / clogging"),
    "none": ("замечаний нет, всё в норме", "no issues found, all normal"),
}

FAILURE_CUES = {
    "bearing_wear": r"подшип|подш\.|bearing|сепаратор|стук|гул|гуд[ие]т|шум|люфт",
    "seal_leak": r"теч|утеч|уплотн|сальник|прокладк|торц|травит|подтек|капает|сочит|leak|seal|gasket|drip",
    "misalignment": r"центр|соосн|перекос|misalign|alignment",
    "imbalance": r"дисбаланс|баланс|биени|вибрац|imbalance|unbalance|balanc|vibrat",
    "overheating": r"перегр|нагрев|гре[ею]|температур|горяч|overheat|hot\b|temperature",
    "corrosion": r"корроз|ржав|окисл|corros|rust",
    # not "электр"/"electric": they match the equipment name "электродвигатель" / "electric motor"
    "electrical_fault": r"изоляц|клемм|обмот|замыкан|\bкз\b|электрич|пробо|подгор|искр|insulation|terminal|winding|short circuit|electrical|spark",
    "blockage": r"засор|забит|забив|загрязн|грязь|отложен|накип|clog|block|plugged|dirty|fouled",
    "none": r"замечан|норм|в порядке|исправ|штатно|\bок\b|no issue|no defect|normal|\bok\b|good condition|nothing found|fine\b",
}

PART_NAMES = {
    "bearing": (["подшипник"], ["bearing"]),
    "mechanical_seal": (["торцевое уплотнение", "торц. уплотнение", "торцевик"], ["mechanical seal"]),
    "gasket": (["прокладка", "прокладку"], ["gasket"]),
    "coupling": (["муфта", "полумуфта", "муфту"], ["coupling"]),
    "belt": (["ремень", "ремни"], ["belt", "belts"]),
    "impeller": (["рабочее колесо", "крыльчатка", "РК"], ["impeller"]),
    "filter": (["фильтр", "фильтроэлемент"], ["filter"]),
    "oil": (["масло"], ["oil"]),
    "valve_seat": (["седло клапана", "седло"], ["valve seat", "seat"]),
    "sensor": (["датчик"], ["sensor"]),
}
PART_CUES = {
    "bearing": r"подшип|bearing", "mechanical_seal": r"торц|mechanical seal|mech\.? seal",
    "gasket": r"проклад|gasket", "coupling": r"муфт|coupling", "belt": r"ремн|ремен|belt",
    "impeller": r"колес|крыльчат|\bрк\b|impeller", "filter": r"фильтр|filter", "oil": r"масл|oil",
    "valve_seat": r"седл|seat", "sensor": r"датчик|sensor",
}

DOWNTIME_TRUE_CUES = r"останов|остановл|простой|простоя|выведен|вывели|отключ|stopp|shut ?down|downtime|taken out|offline"
DOWNTIME_FALSE_CUES = r"без останов|без простоя|на ходу|не останав|в работе|without stop|no downtime|while running|on the run|kept running|stayed online"
DATE_LIKE = r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|вчера|сегодня|yesterday|today"
DURATION_LIKE = r"\d+(?:[.,]\d+)?\s*(?:ч|час|h\b|hr|hrs|hour|мин|min|м\b)"

# Cyrillic letters that look exactly like Latin ones: technicians mix them up on Russian keyboards.
CYRILLIC_TWINS = {"P": "Р", "K": "К", "M": "М", "E": "Е", "C": "С", "B": "В", "A": "А", "X": "Х"}

FAILURES_BY_TYPE = {
    "pump": ["bearing_wear", "seal_leak", "misalignment", "imbalance", "overheating", "blockage"],
    "compressor": ["overheating", "seal_leak", "imbalance", "bearing_wear"],
    "fan": ["imbalance", "bearing_wear", "misalignment"],
    "electric_motor": ["overheating", "electrical_fault", "bearing_wear"],
    "valve": ["seal_leak", "corrosion", "blockage"],
    "heat_exchanger": ["blockage", "corrosion", "seal_leak"],
    "conveyor": ["misalignment", "bearing_wear"],
    "gearbox": ["overheating", "bearing_wear", "seal_leak"],
}


# ---------------------------------------------------------------- facts


@dataclass
class Facts:
    record_id: str
    equipment_id: str
    equipment_type: str
    work_type: str
    failure_mode: str | None
    replaced_parts: list[str]
    duration_hours: float | None
    work_date: date | None
    downtime: bool | None
    language: str  # ru / en / mixed
    style: str  # terse / normal / narrative
    wording: str  # hint (standard phrases) / free (own words, slang, symptoms)
    typos: int  # number of typos injected after verification
    tag_text: str
    date_text: str | None
    duration_text: str | None

    def ground_truth(self) -> dict:
        return {
            "equipment_id": self.equipment_id,
            "equipment_type": self.equipment_type,
            "work_type": self.work_type,
            "failure_mode": self.failure_mode,
            "replaced_parts": sorted(self.replaced_parts),
            "duration_hours": self.duration_hours,
            "work_date": self.work_date.isoformat() if self.work_date else None,
            "downtime": self.downtime,
        }


def _pick(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _parts_for(rng, eq_type, work_type, failure):
    if work_type == "corrective_repair":
        if failure is None:
            return []
        options = {
            "bearing_wear": [["bearing"], ["bearing", "oil"]],
            "seal_leak": [["gasket"]] if eq_type in ("valve", "heat_exchanger") else [["mechanical_seal"], ["gasket"]],
            "misalignment": [[], ["coupling"]],
            "imbalance": [["impeller"], []] if eq_type in ("pump", "fan") else [[]],
            "overheating": [["oil"], ["sensor"], []],
            "corrosion": [["valve_seat"], ["gasket"]] if eq_type == "valve" else [["gasket"], []],
            "electrical_fault": [["sensor"], []],
            "blockage": [["filter"], []],
        }[failure]
        return rng.choice(options)
    if work_type == "preventive_maintenance":
        return rng.choice([["filter"], ["oil"], ["filter", "oil"], ["belt"], []] if eq_type != "valve" else [[], ["gasket"]])
    if work_type == "lubrication":
        return ["oil"] if rng.random() < 0.8 else []
    if work_type == "cleaning":
        return ["filter"] if rng.random() < 0.4 else []
    return []


def _duration_text(rng, hours: float, language: str) -> str:
    minutes = round(hours * 60)
    ru = language in ("ru", "mixed") and rng.random() < 0.8
    if minutes < 60 or (minutes % 60 and rng.random() < 0.4):
        if minutes > 60 and rng.random() < 0.5:
            h, m = divmod(minutes, 60)
            return f"{h}ч{m}м" if ru else f"{h}h{m}m"
        return rng.choice([f"{minutes} мин", f"{minutes}мин"]) if ru else rng.choice([f"{minutes} min", f"{minutes}min"])
    value = f"{hours:g}"
    if ru:
        value = value.replace(".", ",") if rng.random() < 0.6 else value
        return rng.choice([f"{value} ч", f"{value}ч", f"{value} часа" if hours in (2, 3, 4) else f"{value} ч"])
    return rng.choice([f"{value} h", f"{value}h", f"{value} hrs"])


def _date_text(rng, d: date, language: str) -> str:
    if language == "en" and rng.random() < 0.5:
        return d.strftime("%Y-%m-%d")
    return rng.choice([d.strftime("%d.%m.%Y"), d.strftime("%d.%m.%Y"), d.strftime("%d.%m.%y"), d.strftime("%Y-%m-%d")])


def _tag_text(rng, tag: str, language: str) -> str:
    r = rng.random()
    if language != "en" and r < 0.12 and tag[0] in CYRILLIC_TWINS:
        return "".join(CYRILLIC_TWINS.get(ch, ch) for ch in tag)
    if r < 0.30:
        return tag.replace("-", "")
    if r < 0.35:
        return tag.replace("-", " ")
    return tag


def sample_facts(n: int, seed: int = SEED) -> list[Facts]:
    rng = random.Random(seed)
    start, end = date(2024, 1, 1), date(2026, 9, 30)
    facts = []
    for i in range(n):
        tag, eq_type, _ = rng.choice(REGISTRY)
        work = _pick(rng, {"corrective_repair": 40, "preventive_maintenance": 20, "inspection": 15,
                           "lubrication": 8, "adjustment": 8, "cleaning": 9})
        failures = FAILURES_BY_TYPE[eq_type]
        if work == "corrective_repair":
            failure = None if rng.random() < 0.1 else rng.choice(failures)
        elif work == "inspection":
            failure = "none" if rng.random() < 0.6 else rng.choice(failures)
        elif work == "adjustment":
            failure = "misalignment" if ("misalignment" in failures and rng.random() < 0.6) else "none"
        elif work == "cleaning":
            failure = "blockage" if ("blockage" in failures and rng.random() < 0.5) else "none"
        else:
            failure = "none" if rng.random() < 0.85 else rng.choice(failures)
        parts = _parts_for(rng, eq_type, work, failure)
        duration = None if rng.random() < 0.12 else rng.choice({
            "corrective_repair": [1, 1.5, 2, 2.5, 3, 4, 6, 8, 12],
            "preventive_maintenance": [1, 2, 3, 4, 6],
            "inspection": [0.25, 0.5, 0.75, 1],
            "lubrication": [0.25, 0.5, 0.75, 1],
            "adjustment": [0.5, 1, 1.5, 2],
            "cleaning": [0.5, 1, 2, 3, 4],
        }[work])
        work_date = None if rng.random() < 0.15 else start + timedelta(days=rng.randrange((end - start).days + 1))
        p_down = {"corrective_repair": 0.8, "preventive_maintenance": 0.5, "inspection": 0.05,
                  "lubrication": 0.2, "adjustment": 0.6, "cleaning": 0.6}[work]
        downtime = None if rng.random() < 0.2 else (rng.random() < p_down)
        language = _pick(rng, {"ru": 60, "en": 25, "mixed": 15})
        style = _pick(rng, {"terse": 50, "normal": 35, "narrative": 15})
        wording = "free" if rng.random() < 0.5 else "hint"
        typos = 0 if rng.random() < 0.75 else rng.choice([1, 1, 2])
        facts.append(Facts(
            record_id=f"ML-{i + 1:04d}", equipment_id=tag, equipment_type=eq_type, work_type=work,
            failure_mode=failure, replaced_parts=parts, duration_hours=duration, work_date=work_date,
            downtime=downtime, language=language, style=style, wording=wording, typos=typos,
            tag_text=_tag_text(rng, tag, language),
            date_text=_date_text(rng, work_date, language) if work_date else None,
            duration_text=_duration_text(rng, duration, language) if duration else None,
        ))
    return facts


# ---------------------------------------------------------------- verification


def verify(text: str, f: Facts) -> list[str]:
    problems = []
    low = text.lower()
    if f.tag_text not in text:
        problems.append(f"The equipment tag must appear exactly as '{f.tag_text}'.")
    for other_tag, _, _ in REGISTRY:
        if other_tag != f.equipment_id and re.search(rf"\b{re.escape(other_tag)}\b", text):
            problems.append(f"Do not mention other equipment ({other_tag}).")
    dates_found = re.findall(DATE_LIKE, low)
    if f.date_text:
        if f.date_text not in text:
            problems.append(f"The date must appear exactly as '{f.date_text}'.")
        if len(dates_found) != 1:
            problems.append("Mention exactly one date and no relative dates (yesterday, today).")
    elif dates_found:
        problems.append("Do not mention any date.")
    durations_found = re.findall(DURATION_LIKE, low)
    if f.duration_text:
        if f.duration_text not in text:
            problems.append(f"The duration must appear exactly as '{f.duration_text}'.")
        if len(durations_found) > 1 and not (len(durations_found) == 2 and re.fullmatch(r"\d+ч\d+м|\d+h\d+m", f.duration_text)):
            problems.append("Mention only one duration and no other time amounts.")
    elif durations_found:
        problems.append("Do not mention how long the work took, and no other time amounts.")
    for part in f.replaced_parts:
        if not re.search(PART_CUES[part], low):
            problems.append(f"Say that the {part} was replaced or refilled.")
    if f.downtime is True and (not re.search(DOWNTIME_TRUE_CUES, low) or re.search(DOWNTIME_FALSE_CUES, low)):
        problems.append("Say that the equipment was stopped for the work (and do not say it kept running).")
    if f.downtime is False and not re.search(DOWNTIME_FALSE_CUES, low):
        problems.append("Say that the equipment was not stopped (kept running).")
    if f.downtime is None and (re.search(DOWNTIME_TRUE_CUES, low) or re.search(DOWNTIME_FALSE_CUES, low)):
        problems.append("Do not say whether the equipment was stopped or kept running.")
    if f.failure_mode and not re.search(FAILURE_CUES[f.failure_mode], low):
        problems.append(f"State the finding clearly: {FAILURE_MEANING[f.failure_mode][1]}.")
    if f.failure_mode is None:
        for mode, cue in FAILURE_CUES.items():
            if mode != "none" and re.search(cue, low):
                problems.append("Do not name the cause of the fault; only say it was fixed.")
                break
    if len(text) > 600:
        problems.append("Too long: at most 3 short sentences.")
    return problems


# ---------------------------------------------------------------- text generation


STYLE_HINT = {
    "terse": "very short and telegraphic: abbreviations, almost no punctuation, like a quick note in a CMMS field",
    "normal": "short and plain, one or two sentences",
    "narrative": "two or three sentences with a bit more context about what was observed and done",
}
LANG_HINT = {
    "ru": "Russian",
    "en": "English (the technician is not a native speaker)",
    "mixed": "Russian mixed with English technical words (code-switching), as Russian technicians often write",
}

# Phrase pools: one phrase is picked per record, so the texts do not repeat our labels.
WORK_HINTS = {
    "corrective_repair": (["внеплановый ремонт", "ремонт по заявке оператора", "аварийный ремонт",
                           "устранение неисправности", "отказ, ремонт"],
                          ["repair after failure", "breakdown repair", "fixed after operator call", "unplanned repair"]),
    "preventive_maintenance": (["ППР", "плановое ТО", "ТО по графику", "плановое обслуживание"],
                               ["PM", "scheduled maintenance", "planned service"]),
    "inspection": (["осмотр", "обход", "плановая проверка", "проверка при обходе"],
                   ["inspection", "walk-down check", "routine check"]),
    "lubrication": (["смазка", "долив масла", "смазка узлов"], ["lubrication", "greasing", "oil top-up"]),
    "adjustment": (["центровка", "регулировка", "подтяжка креплений", "выставили соосность"],
                   ["alignment", "adjustment", "re-tightening"]),
    "cleaning": (["чистка", "промывка", "продувка и чистка"], ["cleaning", "flushing"]),
}
FAILURE_HINTS = {
    "bearing_wear": (["износ подшипника", "гул подшипника", "люфт в подшипниковом узле", "подшипник шумит"],
                     ["bearing worn", "bearing noise", "bearing play"]),
    "seal_leak": (["течь по торцу", "подтекание по уплотнению", "течь по прокладке", "капает с сальника"],
                  ["seal leak", "leaking seal", "gasket leak"]),
    "misalignment": (["расцентровка", "нарушена соосность валов"], ["misalignment", "shafts misaligned"]),
    "imbalance": (["дисбаланс", "дисбаланс рабочего колеса", "биение ротора из-за дисбаланса"],
                  ["imbalance", "rotor unbalance"]),
    "overheating": (["перегрев", "греется", "повышенная температура корпуса"],
                    ["overheating", "running hot", "high temperature"]),
    "corrosion": (["коррозия", "ржавчина на корпусе", "коррозия седла"], ["corrosion", "rust"]),
    "electrical_fault": (["пробой изоляции", "подгорели клеммы", "межвитковое замыкание обмотки",
                          "низкое сопротивление изоляции"],
                         ["insulation fault", "burnt terminals", "winding short circuit"]),
    "blockage": (["засор", "забит фильтр", "забиты трубки"], ["clogged", "blockage", "plugged strainer"]),
    "none": (["замечаний нет", "в норме", "без замечаний", "ок"], ["no issues", "OK", "all normal", "nothing found"]),
}
DOWNTIME_HINTS = {
    True: (["с остановом", "агрегат остановлен", "вывели в ремонт", "простой"],
           ["unit stopped", "shut down for the job", "downtime"]),
    False: (["без останова", "на ходу", "не останавливали"], ["without stop", "kept running", "no downtime"]),
}
SURNAMES = ["Иванов", "Петров", "Сидоренко", "Ковалёв", "Смирнов", "Захаров", "Гусев", "Орлов"]

STYLE_EXAMPLES = """Examples of the style (other equipment, other facts; do not copy their facts):
X-000 зам. подш. со стор. привода, после пуска норм, 3ч, ост.
Насос X-000: подтекание по торцу, поменяли торцевое уплотнение. 2,5 ч, агрегат останавливали.
X-000 PM done, oil changed, 1h, kept running
X-000 греется, проверили, заменён датчик, 40 мин
при обходе стук со стор. привода X-000, разобрали - сепаратор в хлам, поставили новый подш.
X-000 сальник травит, перебили набивку
ЭД X-000 мегаомметр показал низкое сопр. изол., сушили обмотку"""


def _hint(rng: random.Random, pool: tuple[list[str], list[str]], language: str) -> str:
    ru, en = pool
    if language == "ru":
        return rng.choice(ru)
    if language == "en":
        return rng.choice(en)
    return rng.choice(ru + en)


def _facts_prompt(f: Facts, rng: random.Random) -> str:
    ru_names, en_names = TYPE_NAMES[f.equipment_type]
    name = rng.choice(en_names if f.language == "en" else ru_names)
    free = f.wording == "free"
    work = WORK_MEANING[f.work_type][1] if free else _hint(rng, WORK_HINTS[f.work_type], f.language)
    lines = [
        f"- equipment: {name}, tag written EXACTLY as '{f.tag_text}'",
        f"- work: {work}",
    ]
    if f.failure_mode is None:
        lines.append("- a fault was fixed, but do NOT say what the fault or its cause was")
    elif free:
        lines.append(f"- finding: {FAILURE_MEANING[f.failure_mode][1]} (describe it in your own words: "
                     "a symptom, slang or an abbreviation, as a technician would)")
    else:
        lines.append(f"- finding: {_hint(rng, FAILURE_HINTS[f.failure_mode], f.language)}")
    if f.replaced_parts:
        ru = f.language != "en"
        parts = ", ".join(rng.choice(PART_NAMES[p][0 if ru else 1]) for p in f.replaced_parts)
        lines.append(f"- replaced or refilled: {parts} (only these, do not mention other replaced parts)")
    else:
        lines.append("- nothing was replaced or refilled (do not write that anything was replaced)")
    lines.append(f"- date: write EXACTLY '{f.date_text}'" if f.date_text else "- no date (do not mention any date)")
    lines.append(f"- duration: write EXACTLY '{f.duration_text}'" if f.duration_text else "- no duration (do not mention how long it took)")
    if f.downtime is None:
        lines.append("- do NOT say whether the equipment was stopped or kept running")
    elif free:
        lines.append("- the equipment was stopped for the work" if f.downtime else "- the equipment kept running during the work")
    else:
        lines.append(f"- stop: {_hint(rng, DOWNTIME_HINTS[f.downtime], f.language)}")
    if rng.random() < 0.25:
        lines.append(f"- work order number: WO-{rng.randint(10000, 99999)}")
    if rng.random() < 0.2:
        lines.append(f"- technician: {rng.choice(SURNAMES)} {rng.choice('АВГДЕКМНС')}.")
    return "\n".join(lines)


GEN_SYSTEM = (
    "You write realistic maintenance log entries as a plant technician types them into a CMMS. "
    "Use only the facts given, in your own words; vary the wording; do not copy the fact list literally; "
    "write in the requested language and do not copy English words from the facts into a Russian entry. "
    "Do not add any other numbers, dates, tags, parts, measurements or causes. "
    "Return only the entry text, no quotes, no comments.\n\n" + STYLE_EXAMPLES
)


def _template_text(f: Facts, rng: random.Random) -> str:
    """Deterministic fallback that satisfies verify() by construction."""
    ru = f.language != "en"
    ru_names, en_names = TYPE_NAMES[f.equipment_type]
    name = (ru_names if ru else en_names)[0]
    work = {
        "corrective_repair": ("ремонт", "repair"), "preventive_maintenance": ("плановое ТО", "scheduled PM"),
        "inspection": ("осмотр", "inspection"), "lubrication": ("смазка", "lubrication"),
        "adjustment": ("регулировка", "adjustment"), "cleaning": ("чистка", "cleaning"),
    }[f.work_type][0 if ru else 1]
    bits = [f"{name} {f.tag_text}", work]
    if f.failure_mode is None:
        bits.append("неисправность устранена" if ru else "fault fixed")
    elif f.failure_mode == "none":
        bits.append("замечаний нет" if ru else "no issues")
    else:
        bits.append({
            "bearing_wear": ("износ подшипника", "bearing worn"), "seal_leak": ("течь по уплотнению", "seal leak"),
            "misalignment": ("расцентровка", "misalignment"), "imbalance": ("дисбаланс", "imbalance"),
            "overheating": ("перегрев", "overheating"), "corrosion": ("коррозия", "corrosion"),
            "electrical_fault": ("неисправность изоляции", "insulation fault"), "blockage": ("засор", "clogged"),
        }[f.failure_mode][0 if ru else 1])
    if f.replaced_parts:
        names = ", ".join(PART_NAMES[p][0 if ru else 1][0] for p in f.replaced_parts)
        bits.append(("замена: " if ru else "replaced: ") + names)
    if f.duration_text:
        bits.append(f.duration_text)
    if f.downtime is True:
        bits.append("с остановом" if ru else "unit stopped")
    elif f.downtime is False:
        bits.append("без останова" if ru else "without stop")
    text = ", ".join(bits)
    return f"{f.date_text} {text}" if f.date_text else text


_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{5,}")


def add_typos(text: str, f: Facts, rng: random.Random) -> str:
    """Typos as technicians make them. Never inside the tag, the date or the duration."""
    protected = [x for x in (f.tag_text, f.date_text, f.duration_text) if x]
    spans = [(m.start(), m.end()) for x in protected for m in re.finditer(re.escape(x), text)]
    words = [m for m in _WORD.finditer(text) if not any(a < m.end() and m.start() < b for a, b in spans)]
    chosen = rng.sample(words, min(f.typos, len(words)))
    # right to left, so that earlier positions stay valid after each change
    for m in sorted(chosen, key=lambda m: m.start(), reverse=True):
        w = m.group(0)
        i = rng.randrange(1, len(w) - 2)
        kind = rng.choice(["swap", "drop", "double"])
        if kind == "swap":
            new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        elif kind == "drop":
            new = w[:i] + w[i + 1:]
        else:
            new = w[:i] + w[i] + w[i:]
        text = text[:m.start()] + new + text[m.end():]
    return text


def write_entry(client: LLMClient, f: Facts, max_attempts: int = 3) -> tuple[str, dict]:
    rng = random.Random(f"{SEED}-{f.record_id}")
    user = (
        f"Write one maintenance log entry. Language: {LANG_HINT[f.language]}. Style: {STYLE_HINT[f.style]}.\n"
        f"Facts:\n{_facts_prompt(f, rng)}"
    )
    messages = [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]
    history = []
    for attempt in range(1, max_attempts + 1):
        response = client.chat(messages, meta={"record_id": f.record_id, "attempt": attempt, "task": "generate"})
        text = (response.content or "").strip().strip('"').strip()
        problems = verify(text, f) if text else ["Empty answer."]
        history.append({"attempt": attempt, "problems": problems})
        if not problems:
            if f.typos:
                text = add_typos(text, f, rng)
            return text, {"source": "llm", "attempts": attempt, "history": history}
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Fix these problems and return only the entry:\n" + "\n".join(f"- {p}" for p in problems)},
        ]
    text = _template_text(f, rng)
    assert not verify(text, f), (f.record_id, verify(text, f), text)
    return text, {"source": "template", "attempts": max_attempts, "history": history}


# ---------------------------------------------------------------- main


def make_splits(record_ids: list[str], seed: int = SEED) -> dict[str, list[str]]:
    ids = list(record_ids)
    random.Random(seed + 1).shuffle(ids)
    eval_50 = sorted(ids[:50])
    dev = sorted(ids[50:])
    return {"eval_50": eval_50, "dev_20": dev[:20], "dev": dev}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="gpt-oss-120b", help="model profile that writes the texts")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    profile = get_profile(args.profile)
    profile.temperature = args.temperature
    profile.max_tokens = max(profile.max_tokens, 3000)
    logger = CallLogger(args.out / "generation_calls.jsonl")
    client = LLMClient(profile, logger=logger)

    facts = sample_facts(args.n)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(lambda f: write_entry(client, f), facts), total=len(facts), desc="generate"))

    with open(args.out / "maintenance_log.jsonl", "w", encoding="utf-8") as out:
        for f, (text, gen) in zip(facts, results):
            out.write(json.dumps({
                "record_id": f.record_id, "text": text, "language": f.language,
                "ground_truth": f.ground_truth(),
                "meta": {"style": f.style, "wording": f.wording, "typos": f.typos, "tag_text": f.tag_text,
                         "generator": {"profile": args.profile, **gen}},
            }, ensure_ascii=False) + "\n")

    with open(args.out / "equipment_registry.csv", "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["equipment_id", "equipment_type", "area", "name"])
        for tag, eq_type, area in REGISTRY:
            w.writerow([tag, eq_type, area, f"{TYPE_NAMES[eq_type][1][0]} {tag}"])

    splits = make_splits([f.record_id for f in facts])
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    sources = Counter(g["source"] for _, g in results)
    attempts = Counter(g["attempts"] for _, g in results if g["source"] == "llm")
    card = {
        "created_with": {"profile": args.profile, "model": profile.model, "temperature": args.temperature, "seed": SEED},
        "wording": dict(Counter(f.wording for f in facts)),
        "records_with_typos": sum(1 for f in facts if f.typos),
        "n_records": len(facts),
        "text_source": dict(sources),
        "llm_attempts_needed": dict(sorted(attempts.items())),
        "languages": dict(Counter(f.language for f in facts)),
        "work_types": dict(Counter(f.work_type for f in facts)),
        "splits": {k: len(v) for k, v in splits.items()},
    }
    (args.out / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
