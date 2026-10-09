"""The contract between the model and the rest of the system.

One Pydantic model is the single source of truth: we generate the JSON Schema for the
model from it, and we validate the model's answer with it.
"""

from __future__ import annotations

import copy
from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class EquipmentType(str, Enum):
    pump = "pump"
    compressor = "compressor"
    fan = "fan"
    electric_motor = "electric_motor"
    valve = "valve"
    heat_exchanger = "heat_exchanger"
    conveyor = "conveyor"
    gearbox = "gearbox"
    other = "other"


class WorkType(str, Enum):
    corrective_repair = "corrective_repair"
    preventive_maintenance = "preventive_maintenance"
    inspection = "inspection"
    lubrication = "lubrication"
    adjustment = "adjustment"
    cleaning = "cleaning"
    other = "other"


class FailureMode(str, Enum):
    bearing_wear = "bearing_wear"
    seal_leak = "seal_leak"
    misalignment = "misalignment"
    imbalance = "imbalance"
    overheating = "overheating"
    corrosion = "corrosion"
    electrical_fault = "electrical_fault"
    blockage = "blockage"
    none = "none"  # the work was done and no failure was found
    other = "other"


class Part(str, Enum):
    bearing = "bearing"
    mechanical_seal = "mechanical_seal"
    gasket = "gasket"
    coupling = "coupling"
    belt = "belt"
    impeller = "impeller"
    filter = "filter"
    oil = "oil"
    valve_seat = "valve_seat"
    sensor = "sensor"


class MaintenanceRecord(BaseModel):
    """Structured version of one free-text maintenance log entry."""

    model_config = ConfigDict(extra="forbid")

    equipment_id: Optional[str] = Field(
        description="Equipment tag from the plant registry, Latin letters, format like 'P-101' or 'P-101A'. "
        "null if no equipment is named."
    )
    equipment_type: EquipmentType = Field(description="Type of the equipment.")
    work_type: WorkType = Field(description="What kind of work was done.")
    failure_mode: Optional[FailureMode] = Field(
        description="How the equipment failed. 'none' if the text says that no problem was found. "
        "null if the text does not say what the problem was."
    )
    defect_summary: Optional[str] = Field(
        description="Short English summary of the defect, at most 15 words. null if there was no defect."
    )
    replaced_parts: list[Part] = Field(description="Parts that were replaced or refilled. Empty list if none.")
    duration_hours: Optional[float] = Field(
        description="Duration of the work in hours (45 minutes = 0.75). null if not stated."
    )
    work_date: Optional[date] = Field(description="Date of the work, ISO format YYYY-MM-DD. null if not stated.")
    downtime: Optional[bool] = Field(
        description="true if the equipment was stopped for the work, false if it kept running. null if not stated."
    )
    evidence: list[str] = Field(
        description="Short exact quotes copied from the source text that support the extracted values."
    )


# Fields compared with the ground truth. defect_summary is free text: it is evaluated in topic 2
# (LLM-as-judge), evidence is checked by business rules, not by accuracy.
SCORED_FIELDS = [
    "equipment_id",
    "equipment_type",
    "work_type",
    "failure_mode",
    "replaced_parts",
    "duration_hours",
    "work_date",
    "downtime",
]


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            resolved = _inline_refs(copy.deepcopy(defs[name]), defs)
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**resolved, **extra}
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _strictify(node: Any) -> Any:
    """Every property required, no extra properties, no defaults/titles: what strict mode expects."""
    if isinstance(node, dict):
        node = {k: _strictify(v) for k, v in node.items() if k not in ("title", "default")}
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"].keys())
            node["additionalProperties"] = False
        return node
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    return node


def strict_json_schema(model: type[BaseModel] = MaintenanceRecord) -> dict[str, Any]:
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    return _strictify(_inline_refs(schema, defs))


def response_format(model: type[BaseModel] = MaintenanceRecord) -> dict[str, Any]:
    """`response_format` argument for an OpenAI-compatible chat completion."""
    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__, "schema": strict_json_schema(model), "strict": True},
    }


def fields_description(model: type[BaseModel] = MaintenanceRecord) -> str:
    """Plain-text field list for the prompt (used by the baseline that has no schema)."""
    lines = []
    for name, f in model.model_fields.items():
        annotation = f.annotation
        enum_cls = None
        for candidate in (EquipmentType, WorkType, FailureMode, Part):
            if candidate.__name__ in str(annotation):
                enum_cls = candidate
        values = f" One of: {', '.join(e.value for e in enum_cls)}." if enum_cls else ""
        lines.append(f"- {name}: {f.description}{values}")
    return "\n".join(lines)
