"""Applicability through the equipment registry: tag -> station, type, power, machine class.

The question "P-101, 4,8 мм/с" has no class and no station. The registry has them, and they decide which
document applies (R-12 for station 2, R-14 for station 3) and which row of Table 3 to read. This is a lookup
by code, not a search: the first step to the "equipment card" tool of topic 5.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import DATA_DIR
from .query import QueryFields
from .store import Chunk

REGISTRY_FILE = DATA_DIR / "equipment_registry.csv"
# a tag that is not in the registry still tells the type of equipment by its prefix
PREFIX_TYPE = {"P": "pump", "K": "compressor", "B": "fan", "F": "fan", "V": "valve", "XV": "valve", "FV": "valve",
               "HV": "valve", "T": "tank", "E": "heat_exchanger", "M": "electric_motor"}
TYPE_WORDS = {"pump": re.compile(r"насос|\bpump", re.I)}


@dataclass
class EquipmentCard:
    tag: str
    equipment_type: str
    area: str = ""
    station: str = ""
    power_kw: str = ""
    machine_class: str = ""
    datasheet: str = ""
    in_registry: bool = True

    def describe(self) -> str:
        if not self.in_registry:
            return f"{self.tag}: not in the equipment registry (type by prefix: {self.equipment_type})"
        parts = [f"{self.tag}: {self.equipment_type}", self.area or "area unknown"]
        if self.power_kw:
            parts.append(f"motor {self.power_kw} kW")
        if self.machine_class:
            parts.append(f"machine class {self.machine_class}")
        if self.datasheet:
            parts.append(f"datasheet {self.datasheet}")
        return ", ".join(parts)


def load_registry(path: str | Path = REGISTRY_FILE) -> dict[str, EquipmentCard]:
    with open(path, encoding="utf-8") as f:
        return {r["equipment_id"]: EquipmentCard(r["equipment_id"], r["equipment_type"], r["area"], r["station"],
                                                 r["power_kw"], r["machine_class"], r["datasheet"])
                for r in csv.DictReader(f)}


def card(tag: str, registry: dict[str, EquipmentCard]) -> EquipmentCard:
    if tag in registry:
        return registry[tag]
    prefix = tag.split("-")[0]
    return EquipmentCard(tag, PREFIX_TYPE.get(prefix, "unknown"), in_registry=False)


@dataclass
class Scope:
    """What the question is about. Empty sets mean "no restriction"."""
    stations: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    doc_ids: set[str] = field(default_factory=set)
    cards: list[EquipmentCard] = field(default_factory=list)

    def describe(self) -> str:
        parts = []
        for name in ("doc_ids", "stations", "types", "tags"):
            v = getattr(self, name)
            if v:
                parts.append(f"{name}={sorted(v)}")
        return ", ".join(parts) or "no restriction"


def scope_for(fields: QueryFields, question: str, registry: dict[str, EquipmentCard]) -> Scope:
    cards = [card(t, registry) for t in fields.tags]
    stations = {c.station for c in cards if c.station} | set(fields.stations)
    types = {c.equipment_type for c in cards if c.equipment_type != "unknown"}
    if not types:
        types = {t for t, rx in TYPE_WORDS.items() if rx.search(question)}
    return Scope(stations=stations, types=types, tags=set(fields.tags), doc_ids=set(fields.doc_ids), cards=cards)


def in_scope(c: Chunk, scope: Scope, use_doc_ids: bool = True) -> bool:
    """Does the chunk apply to what the question is about? A document without scope fields applies everywhere."""
    if use_doc_ids and scope.doc_ids and c.doc_id not in scope.doc_ids:
        return False
    cs = set(c.scope_stations)
    if scope.stations and cs and "all" not in cs and not cs & scope.stations:
        return False
    ct = set(c.scope_equipment_types)
    if scope.types and ct and "all" not in ct and not ct & scope.types:
        return False
    if scope.tags and c.scope_tags and not set(c.scope_tags) & scope.tags:
        return False
    return True


def equipment_context(scope: Scope) -> str:
    """The equipment cards that go into the prompt next to the found text."""
    if not scope.cards:
        return ""
    return "Equipment registry:\n" + "\n".join(f"- {c.describe()}" for c in scope.cards)
